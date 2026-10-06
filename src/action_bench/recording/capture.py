"""Pinned-buffer state capture; synchronization and disk I/O stay on a writer thread."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .io import write_capture
from .schema import validate_manifest


class TrainingStateCapture:
    """One bounded window at a time, with partial-window publication on shutdown."""

    def __init__(
        self,
        channel_specs,
        channel_dtypes,
        window_steps,
        max_start_delay_steps,
        manifest_template,
        output_root,
        source_device,
    ):
        for value, minimum in ((window_steps, 1), (max_start_delay_steps, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(
                    "capture window/delay must be positive/nonnegative integers"
                )
        self._template = dict(manifest_template)
        self._specs = {name: tuple(shape) for name, shape in channel_specs.items()}
        self._dtypes = {
            name: channel_dtypes.get(name, torch.float32) for name in channel_specs
        }
        expected = {
            name: {
                "shape": list(shape),
                "dtype": str(self._dtypes[name]).split(".", 1)[1],
            }
            for name, shape in self._specs.items()
        }
        declared = {
            name: {"shape": list(spec["shape"]), "dtype": spec["dtype"]}
            for name, spec in self._template["channels"].items()
        }
        if expected != declared or set(channel_dtypes) - set(channel_specs):
            raise ValueError(
                "capture manifest channels differ from tensor specifications"
            )
        validate_manifest(
            dict(self._template, epoch=0, global_step=0, num_frames=window_steps)
        )
        self._window = window_steps
        self._delay = max_start_delay_steps
        self._capacity = window_steps + max_start_delay_steps
        self._root = Path(output_root)
        self._device = torch.device(source_device)
        self._buffers = {
            name: torch.empty(
                (self._capacity, *shape),
                dtype=self._dtypes[name],
                pin_memory=self._device.type == "cuda",
            )
            for name, shape in self._specs.items()
        }
        self._state = "idle"
        self._row = 0
        self._writer = None
        self._error = None

    @property
    def active(self):
        return self._state in {"armed", "recording"}

    def arm(self, epoch, global_step):
        self.raise_if_failed()
        if self._state != "idle":
            return False
        self._epoch, self._global_step, self._row = int(epoch), int(global_step), 0
        self._state = "armed"
        return True

    def append(self, rows):
        if not self.active:
            return
        if self._state == "armed":
            if set(rows) != set(self._specs):
                raise ValueError("capture row channel set mismatch")
            for name, source in rows.items():
                if (
                    source.shape != self._specs[name]
                    or source.dtype != self._dtypes[name]
                    or source.device != self._device
                ):
                    raise ValueError(
                        f"capture row {name!r} shape, dtype or device mismatch"
                    )
            self._state = "recording"
        for name, buffer in self._buffers.items():
            buffer[self._row].copy_(rows[name], non_blocking=True)
        self._row += 1
        if self._row == self._capacity:
            self._start_writer()

    def _start_writer(self):
        self._state = "writing"
        completion = None
        if self._device.type == "cuda":
            completion = torch.cuda.Event()
            completion.record(torch.cuda.current_stream(self._device))
        self._writer = threading.Thread(
            target=self._write,
            args=(completion,),
            name="action-bench-capture",
            daemon=True,
        )
        self._writer.start()

    def _write(self, completion):
        try:
            if completion is not None:
                completion.synchronize()
            reset = self._buffers["reset"][: self._row].numpy()
            reset = reset if reset.ndim == 1 else reset[:, 0]
            # Only trim when a whole requested window remains after the reset.
            candidates = np.flatnonzero(
                reset[: min(self._delay, max(0, self._row - self._window + 1))]
            )
            start = int(candidates[0]) if candidates.size else 0
            stop = min(start + self._window, self._row)
            manifest = dict(
                self._template,
                epoch=self._epoch,
                global_step=self._global_step,
                num_frames=stop - start,
                utc_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            manifest["extras_meta"] = dict(
                manifest.get("extras_meta", {}),
                trim="first_reset" if candidates.size else "none",
            )
            write_capture(
                self._root,
                manifest,
                {
                    name: buffer[start:stop].numpy()
                    for name, buffer in self._buffers.items()
                },
            )
        except BaseException as error:  # noqa: BLE001 -- deliver writer failures to trainer
            self._error = error
        finally:
            self._state = "idle"

    def raise_if_failed(self):
        if self._error is not None:
            raise self._error

    def close(self, timeout=60.0):
        if self.active and self._row:
            self._start_writer()
        if self._writer is not None:
            self._writer.join(timeout)
            if self._writer.is_alive():
                raise TimeoutError("capture writer did not finish before timeout")
        self._state = "closed"
        self.raise_if_failed()
