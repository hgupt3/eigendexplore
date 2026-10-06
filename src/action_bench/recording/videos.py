"""Render committed capture bundles beside a running trainer, using CPU only."""

import json
import sys
import threading
from pathlib import Path


def asset_roots(installation):
    name = {
        "dextreme": "isaacgymenvs",
        "simtoolreal": "simtoolreal",
    }[installation["benchmark"]]
    return {name: Path(installation["host"])}


class VideoWorker:
    """One local renderer; state bundles remain available if rendering fails."""

    def __init__(self, root, roots, *, render=None):
        if render is None:
            from .renderer import render_capture

            render = render_capture
        self.root = Path(root)
        self.roots = roots
        self.render = render
        self.processed = set()
        self.errors = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="capture-videos")
        self.thread.start()

    def _scan(self):
        for manifest in sorted(self.root.glob("step_*/manifest.json")):
            capture = manifest.parent
            if capture in self.processed:
                continue
            self.processed.add(capture)
            output = capture.with_suffix(".mp4")
            try:
                self.render(capture, output, asset_roots=self.roots)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("renderer produced no video")
                print(f"Recording video: {output}", flush=True)
            except Exception as error:
                self.errors[capture.name] = str(error)
                (self.root / "video-errors.json").write_text(
                    json.dumps(self.errors, indent=2) + "\n"
                )
                print(
                    f"Recording render failed for {capture}: {error}", file=sys.stderr
                )

    def _run(self):
        while not self.stop.is_set():
            self._scan()
            self.stop.wait(2)
        self._scan()  # The trainer flushes its final partial capture before exiting.

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.errors:
            raise RuntimeError(
                f"Video rendering failed; captures preserved. See {self.root / 'video-errors.json'}"
            )
