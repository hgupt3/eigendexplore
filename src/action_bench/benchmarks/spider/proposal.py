"""Add independent Eigen innovations to Spider's native IID proposals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from spider.interp import interp


@dataclass
class ControlProposalBatch:
    controls: torch.Tensor
    joint_delta: torch.Tensor
    eigen_delta: torch.Tensor
    eigen_coefficients: torch.Tensor


def _generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


class EigenProposal:
    """Add an independent Action-Bench eigen head to stock SPIDER samples."""

    def __init__(
        self,
        *,
        adapter,
        device: torch.device | str,
        seed: int,
    ) -> None:
        self.device = torch.device(device)
        self.adapter = adapter
        self.reference_finger_index = int(
            self.adapter.finger_control_indices[0].detach().cpu()
        )
        self.eigen_generator = _generator(self.device, seed + 1_000_003)

    def _decode_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Decode already-scaled coefficients through the selected directions."""
        return self.adapter.decode_coefficients(coefficients).full_delta

    def _standard_noise(self, config):
        # Draw the full joint width so different retained ranks share the same
        # random prefix at a paired seed.
        return torch.randn(
            (
                config.num_samples,
                config.noise_scale.shape[1],
                self.adapter.hand_count,
                self.adapter.joint_dim,
            ),
            generator=self.eigen_generator,
            device=self.device,
            dtype=config.noise_scale.dtype,
        )[..., : self.adapter.k]

    def __call__(
        self,
        config,
        ctrls: torch.Tensor,
        stock_controls: torch.Tensor,
        sample_params: dict | None = None,
    ) -> ControlProposalBatch:
        """Add eigen samples without advancing SPIDER's native RNG stream."""
        if ctrls.device != self.device or stock_controls.device != self.device:
            raise ValueError("SPIDER controls moved away from proposal device")
        if stock_controls.shape[1:] != ctrls.shape:
            raise ValueError(
                "stock SPIDER proposal shape does not match control center"
            )
        if sample_params is None:
            sample_params = {}
        eigen_global_noise_scale = float(
            sample_params.get(
                "eigen_global_noise_scale",
                sample_params.get("global_noise_scale", 1.0),
            )
        )
        standard_noise = self._standard_noise(config)
        denominator = config.joint_noise_scale * config.last_ctrl_noise_scale
        if denominator <= 0.0:
            raise ValueError(
                "joint_noise_scale and last_ctrl_noise_scale must be positive "
                "for joint-plus-eigen sampling"
            )
        sample_scale = (
            config.noise_scale[:, :, self.reference_finger_index]
            / denominator
            * eigen_global_noise_scale
        )
        eigen_knots = self.adapter(standard_noise, sample_scale)
        knot_delta = eigen_knots.full_delta
        eigen_delta = interp(knot_delta, config.knot_steps)
        joint_delta = stock_controls - ctrls.unsqueeze(0)
        controls = stock_controls + eigen_delta
        return ControlProposalBatch(
            controls=controls,
            joint_delta=joint_delta,
            eigen_delta=eigen_delta,
            eigen_coefficients=eigen_knots.coefficients,
        )
