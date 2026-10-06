"""Cold initialization, common Eigen noise authority, and optimizer-center contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from action_bench.artifacts import PCAArtifact

from .rank import retained_rank_from_spectrum


def _expected_joint_rms(components, scales):
    basis = components.detach().to(device="cpu", dtype=torch.float64)
    scales = scales.detach().to(device="cpu", dtype=torch.float64)
    rms = float(((scales[:, None] * basis).square().sum() / basis.shape[1]).sqrt())
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("Eigen basis must have finite positive joint RMS")
    return rms


def random_orthonormal_rows(joint_dim, k):
    """Fixed random orthonormal directions, nested across k (random-basis control)."""
    generator = torch.Generator().manual_seed(20260820)
    basis, triangular = torch.linalg.qr(
        torch.randn(joint_dim, joint_dim, generator=generator)
    )
    signs = torch.sign(torch.diag(triangular))
    signs[signs == 0] = 1
    return (basis * signs).T[:k]


def _powered_component_scale(artifact, *, k, like):
    variance = artifact.explained_variance[:k].to(like)
    std = artifact.component_std[:k].to(like)
    # Spider's recorded convention uses eigenvalue power 1 (not std power 1).
    return std[0] * (variance / variance[0])


def normalize_basis(
    adapter, *, artifact_path, target_rms, retained_variance, artifact_layout
):
    artifact = PCAArtifact.load(artifact_path, hand_layout=artifact_layout)
    derived_k = retained_rank_from_spectrum(
        artifact.explained_variance, retained_variance
    )
    if adapter.k != derived_k:
        raise ValueError("adapter rank differs from retained-variance rank")
    source_std = artifact.component_std[: adapter.k].to(adapter.component_std)
    if not torch.equal(source_std, adapter.component_std):
        raise ValueError("adapter component scales differ from artifact")
    target_rms = float(target_rms)
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("Eigen RMS target must be finite and positive")
    scale = _powered_component_scale(artifact, k=adapter.k, like=adapter.component_std)
    raw_rms = _expected_joint_rms(adapter.components, scale)
    factor = target_rms / raw_rms
    applied = scale * factor
    normalized_rms = _expected_joint_rms(adapter.components, applied)
    error = abs(normalized_rms - target_rms)
    tolerance = max(
        1e-12, 8 * torch.finfo(applied.dtype).eps * max(normalized_rms, target_rms)
    )
    if error > tolerance:
        raise ValueError("Eigen RMS normalization failed its postcondition")
    adapter.component_std.copy_(applied)
    return {
        "power": 1.0,
        "derived_k": derived_k,
        "applied_component_scale": applied.cpu().tolist(),
        "rms_normalization": {
            "raw_rms": raw_rms,
            "target_rms": target_rms,
            "factor": factor,
            "normalized_rms_relative_error": error / target_rms,
        },
    }


def _max_error(left, right):
    return float((left - right).abs().max())


def install_cold_start(host, *, binding, audit):
    native_load_data, native_setup_env = host.load_data, host.setup_env
    controls, positions, velocities = (
        binding[k] for k in ("control_indices", "qpos_indices", "qvel_indices")
    )
    mean = np.asarray(binding["mean"], dtype=np.float64)

    def cold_load_data(*args, **kwargs):
        qpos, qvel, ctrl, contact, contact_pos = native_load_data(*args, **kwargs)
        target = torch.as_tensor(mean, device=ctrl.device, dtype=ctrl.dtype)
        cold = ctrl.clone()
        cold[:, controls] = target
        other = [i for i in range(ctrl.shape[1]) if i not in controls]
        audit.update(
            control_center_all_frames_max_abs_error=_max_error(
                cold[:, controls], target
            ),
            nonfinger_control_max_abs_change=_max_error(cold[:, other], ctrl[:, other]),
        )
        return qpos, qvel, cold, contact, contact_pos

    def cold_setup_env(config, data):
        qpos, qvel, ctrl, contact, contact_pos = data
        cold_pos, cold_vel = qpos.clone(), qvel.clone()
        target = torch.as_tensor(mean, device=qpos.device, dtype=qpos.dtype)
        cold_pos[0, positions] = target
        cold_vel[0, velocities] = 0
        other = [i for i in range(qpos.shape[1]) if i not in positions]
        audit.update(
            initial_finger_qpos_max_abs_error=_max_error(
                cold_pos[0, positions], target
            ),
            initial_nonfinger_qpos_max_abs_change=_max_error(
                cold_pos[0, other], qpos[0, other]
            ),
            initial_finger_qvel_max_abs=float(cold_vel[0, velocities].abs().max()),
        )
        return native_setup_env(
            config, (cold_pos, cold_vel, ctrl, contact, contact_pos)
        )

    host.load_data, host.setup_env = cold_load_data, cold_setup_env


class PersistentAugmentedCenter:
    """Maintain separate stock-joint and PCA means within each MPC commit.

    SPIDER still scores physical controls and supplies its unchanged selection
    weights. At a receding-horizon boundary, the selected physical plan is
    shifted exactly as stock SPIDER does, then the latent decomposition is
    rebased to ``joint_center=shifted_plan, eigen_center=0``. This preserves the
    physical center without inventing a latent history for the appended tail.
    """

    def __init__(
        self,
        base,
        *,
        updates_per_commit: int,
        iid_authority: float,
        eigen_authority: float,
    ) -> None:
        self.base = base
        self.adapter = base.adapter
        self.updates_per_commit = int(updates_per_commit)
        self.iid_authority = float(iid_authority)
        self.eigen_authority = float(eigen_authority)
        self.joint_center: torch.Tensor | None = None
        self.eigen_center: torch.Tensor | None = None
        self.expected_physical_center: torch.Tensor | None = None
        self.last_joint_delta: torch.Tensor | None = None
        self.last_eigen_coefficients: torch.Tensor | None = None
        self.last_eigen_delta: torch.Tensor | None = None
        self.last_config = None
        self.commit_index = 0
        self.update_in_commit = 0
        self.history: list[dict[str, float | int]] = []
        self.rebase_history: list[dict[str, float | int]] = []

    def _zero_eigen_center(self, config, ctrls: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            config.noise_scale.shape[1],
            self.adapter.hand_count,
            self.adapter.k,
            device=ctrls.device,
            dtype=ctrls.dtype,
        )

    def _decode_center(self, config, coefficients: torch.Tensor) -> torch.Tensor:
        knot_delta = self.base._decode_coefficients(coefficients.unsqueeze(0))[0]
        from spider.interp import interp

        return interp(knot_delta.unsqueeze(0), config.knot_steps)[0]

    def _initialize(self, config, ctrls: torch.Tensor) -> None:
        self.joint_center = ctrls.detach().clone()
        self.eigen_center = self._zero_eigen_center(config, ctrls)
        self.expected_physical_center = ctrls.detach().clone()
        self.update_in_commit = 0

    def _validate_or_rebase(self, config, ctrls: torch.Tensor) -> None:
        if self.joint_center is None:
            self._initialize(config, ctrls)
            return
        assert self.expected_physical_center is not None
        if self.update_in_commit == self.updates_per_commit:
            shift = int(config.ctrl_steps)
            if not 0 < shift < ctrls.shape[0]:
                raise RuntimeError("invalid receding-horizon shift")
            prefix_error = float(
                (ctrls[:-shift] - self.expected_physical_center[shift:])
                .abs()
                .max()
                .detach()
                .cpu()
            )
            if prefix_error > 2.0e-5:
                raise RuntimeError(
                    "SPIDER receding-horizon prefix differs from the selected "
                    f"physical center: max error={prefix_error:.9g}"
                )
            self.commit_index += 1
            self._initialize(config, ctrls)
            self.rebase_history.append(
                {
                    "commit_index": self.commit_index,
                    "shift_steps": shift,
                    "prefix_max_abs_error": prefix_error,
                }
            )
            return
        incoming_error = float(
            (ctrls - self.expected_physical_center).abs().max().detach().cpu()
        )
        if incoming_error > 2.0e-5:
            raise RuntimeError(
                "SPIDER physical center diverged from augmented latent centers: "
                f"max error={incoming_error:.9g}"
            )

    def __call__(self, config, ctrls, stock_controls, sample_params=None):
        self._validate_or_rebase(config, ctrls)
        scaled_stock = stock_controls
        if self.iid_authority != 1.0:
            scaled_stock = stock_controls.clone()
            finger = self.adapter.finger_control_indices
            center = ctrls.unsqueeze(0).expand_as(stock_controls)
            scaled_stock[..., finger] = center[..., finger] + self.iid_authority * (
                stock_controls[..., finger] - center[..., finger]
            )
        proposal_params = dict(sample_params or {})
        proposal_params["eigen_global_noise_scale"] = self.eigen_authority
        proposal = self.base(config, ctrls, scaled_stock, proposal_params)
        if proposal.eigen_coefficients is None:
            raise RuntimeError("additive PCA proposal omitted its coefficients")
        self.last_joint_delta = proposal.joint_delta.detach()
        self.last_eigen_coefficients = proposal.eigen_coefficients.detach()
        self.last_eigen_delta = proposal.eigen_delta.detach()
        self.last_config = config
        return proposal

    def observe(self, weights, controls, ctrls) -> None:
        del ctrls
        if any(
            value is None
            for value in (
                self.joint_center,
                self.eigen_center,
                self.last_joint_delta,
                self.last_eigen_coefficients,
                self.last_eigen_delta,
                self.last_config,
            )
        ):
            raise RuntimeError("observe called before an augmented proposal")
        assert self.joint_center is not None
        assert self.eigen_center is not None
        assert self.last_joint_delta is not None
        assert self.last_eigen_coefficients is not None
        assert self.last_eigen_delta is not None
        config = self.last_config

        joint_update = torch.einsum("n,nht->ht", weights, self.last_joint_delta)
        coefficient_update = torch.einsum(
            "n,nlhk->lhk", weights, self.last_eigen_coefficients
        )
        eigen_physical_update = torch.einsum(
            "n,nht->ht", weights, self.last_eigen_delta
        )
        self.joint_center = self.joint_center + joint_update
        self.eigen_center = self.eigen_center + coefficient_update
        decoded_eigen_center = self._decode_center(config, self.eigen_center)
        latent_physical_center = self.joint_center + decoded_eigen_center
        selected_physical_center = torch.einsum("n,nht->ht", weights, controls)
        equivalence_error = float(
            (latent_physical_center - selected_physical_center)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        decode_update_error = float(
            (self._decode_center(config, coefficient_update) - eigen_physical_update)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        if equivalence_error > 2.0e-5 or decode_update_error > 2.0e-5:
            raise RuntimeError(
                "augmented-center decode identity failed: "
                f"center={equivalence_error:.9g}, "
                f"update={decode_update_error:.9g}"
            )
        self.history.append(
            {
                "commit_index": self.commit_index,
                "update_in_commit": self.update_in_commit,
                "equivalence_max_abs_error": equivalence_error,
                "decode_update_max_abs_error": decode_update_error,
            }
        )
        self.update_in_commit += 1
        self.expected_physical_center = selected_physical_center.detach().clone()

    def save(self, output_dir: Path) -> None:
        (output_dir / "latent_center_trace.json").write_text(
            json.dumps(
                {"updates": self.history, "rebases": self.rebase_history},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
