"""One implementation of IID plus low-rank Gaussian exploration."""

import math
from typing import Optional

import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal, kl_divergence


class EigenDExplore(nn.Module):
    """Fixed directions with a learnable log scale per direction.

    ``basis`` has shape (rank, action_dim) in policy action units. The host
    supplies its existing mean and diagonal standard deviation on each call.
    Construction validates tensors; rollout calls only check shape/placement.

    ``correlation`` rho > 0 makes the Eigen coefficients an AR(1) process,
    ``eta_t = rho * eta_{t-1} + sqrt(1 - rho**2) * xi_t``, with the same
    per-step spread. Zero (the default) is the paper's i.i.d. noise.
    """

    def __init__(
        self,
        basis: torch.Tensor,
        eigen_sigma: torch.Tensor,
        *,
        correlation: float = 0.0,
    ):
        super().__init__()
        if not math.isfinite(correlation) or not 0 <= correlation < 1:
            raise ValueError("correlation must be in [0, 1)")
        if basis.ndim != 2 or not 0 < basis.shape[0] <= basis.shape[1]:
            raise ValueError(
                "basis must have shape (rank, action_dim), 0 < rank <= action_dim"
            )
        if basis.dtype not in (torch.float32, torch.float64):
            raise TypeError("basis must be float32 or float64")
        if (
            eigen_sigma.ndim not in (1, 2)
            or eigen_sigma.shape[-1] != basis.shape[0]
            or eigen_sigma.numel() == 0
        ):
            raise ValueError("eigen_sigma must have one entry per direction and policy")
        if eigen_sigma.device != basis.device or eigen_sigma.dtype != basis.dtype:
            raise ValueError("basis and eigen_sigma must share device and dtype")
        if not bool(torch.isfinite(basis).all()) or not bool(
            torch.isfinite(eigen_sigma).all()
        ):
            raise ValueError("basis and eigen_sigma must be finite")
        if not bool((eigen_sigma > 0).all()):
            raise ValueError("eigen_sigma must be positive")
        sv = torch.linalg.svdvals(basis.double())
        if bool(sv[-1] <= sv[0] * 1e-6):
            raise ValueError("basis must have full row rank (relative tolerance 1e-6)")
        self.register_buffer("basis", basis.detach().clone())
        self.log_scale = nn.Parameter(eigen_sigma.detach().log().clone())
        self.correlation = float(correlation)

    @classmethod
    def from_covariance(
        cls,
        covariance: torch.Tensor,
        *,
        rank: Optional[int] = None,
        eigen_rms: Optional[float] = None,
        spectrum_power: float = 1.0,
        coordinate_scale: Optional[torch.Tensor] = None,
        correlation: float = 0.0,
    ):
        """Create directions from a positive-semidefinite covariance.

        Covariance is in action units unless ``coordinate_scale`` gives the
        positive native-units-per-action-unit vector (e.g. joint half ranges).
        Rank truncation, spectrum shaping, and RMS matching are explicit.
        With all defaults, the added covariance equals the supplied matrix.
        """
        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] == 0
        ):
            raise ValueError("covariance must be square")
        if covariance.dtype not in (torch.float32, torch.float64):
            raise TypeError("covariance must be float32 or float64")
        if not bool(torch.isfinite(covariance).all()):
            raise ValueError("covariance must be finite")
        if not torch.allclose(covariance, covariance.T, rtol=1e-6, atol=0):
            raise ValueError("covariance must be symmetric")
        if not math.isfinite(spectrum_power) or spectrum_power < 0:
            raise ValueError("spectrum_power must be finite and nonnegative")
        values, vectors = torch.linalg.eigh(covariance.double())
        tolerance = (
            values.abs().max()
            * torch.finfo(covariance.dtype).eps
            * covariance.shape[0]
            * 4
        )
        if bool(values[0] < -tolerance):
            raise ValueError("covariance must be positive semidefinite")
        available = int((values > tolerance).sum())
        if rank is None:
            rank = available
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 < rank <= available
        ):
            raise ValueError("rank must select positive covariance eigenvalues")
        values = values.flip(0)[:rank]
        basis = vectors.flip(1)[:, :rank].T.to(covariance)
        return cls.from_pca(
            basis,
            values.sqrt().to(covariance),
            eigen_rms=eigen_rms,
            spectrum_power=spectrum_power,
            coordinate_scale=coordinate_scale,
            correlation=correlation,
        )

    @classmethod
    def from_pca(
        cls,
        components: torch.Tensor,
        component_std: torch.Tensor,
        *,
        rank: Optional[int] = None,
        eigen_rms: Optional[float] = None,
        spectrum_power: float = 1.0,
        coordinate_scale: Optional[torch.Tensor] = None,
        correlation: float = 0.0,
    ):
        """Use existing PCA rows and coefficient standard deviations directly.

        Rows are ordered by the caller, usually highest variance first. Columns
        must already follow policy action order. ``coordinate_scale`` gives
        positive native-units-per-action-unit (joint half ranges for normalized
        absolute targets). PCA means are not used: exploration is zero-mean.
        With default shaping, covariance is C.T @ diag(component_std**2) @ C.
        """
        if components.ndim != 2 or not 0 < components.shape[0] <= components.shape[1]:
            raise ValueError("components must have shape (rank, action_dim)")
        if components.dtype not in (torch.float32, torch.float64):
            raise TypeError("components must be float32 or float64")
        if (
            component_std.shape != (components.shape[0],)
            or component_std.device != components.device
            or component_std.dtype != components.dtype
        ):
            raise ValueError("component_std must match PCA rows, device and dtype")
        if not bool(torch.isfinite(components).all()) or not bool(
            torch.isfinite(component_std).all() & (component_std >= 0).all()
        ):
            raise ValueError(
                "PCA components must be finite and component_std nonnegative"
            )
        rank = components.shape[0] if rank is None else rank
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 < rank <= components.shape[0]
        ):
            raise ValueError("rank must select existing PCA rows")
        if not math.isfinite(spectrum_power) or spectrum_power < 0:
            raise ValueError("spectrum_power must be finite and nonnegative")
        basis = components[:rank]
        if not bool((component_std[:rank] > 0).all()):
            raise ValueError("selected PCA standard deviations must be positive")
        sigma = component_std[:rank].pow(spectrum_power)
        if coordinate_scale is not None:
            if (
                coordinate_scale.shape != (components.shape[1],)
                or coordinate_scale.device != components.device
                or coordinate_scale.dtype != components.dtype
            ):
                raise ValueError(
                    "coordinate_scale must match PCA width, device and dtype"
                )
            if not bool(
                torch.isfinite(coordinate_scale).all() & (coordinate_scale > 0).all()
            ):
                raise ValueError("coordinate_scale must be finite and positive")
            basis = basis / coordinate_scale
        if eigen_rms is not None:
            if not math.isfinite(eigen_rms) or eigen_rms <= 0:
                raise ValueError("eigen_rms must be finite and positive")
            rms = (basis * sigma[:, None]).square().sum(0).mean().sqrt()
            if not bool(torch.isfinite(rms) & (rms > 0)):
                raise ValueError("PCA must have finite positive noise power")
            sigma = sigma * (eigen_rms / rms)
        return cls(basis, sigma, correlation=correlation)

    def distribution(
        self,
        mean: torch.Tensor,
        iid_sigma: torch.Tensor,
        *,
        eigen_sigma: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ):
        """Return the exact action-space marginal, with differentiable entropy.

        With ``correlation`` > 0, pass the Eigen state the action was sampled
        from (see ``sample``) to get its likelihood. The mean then shifts by
        ``rho * B.T @ (eigen_sigma * state)`` and the Eigen scales shrink by
        ``sqrt(1 - rho**2)``. Without ``state`` this is the per-step marginal.
        The caller guarantees finite means and positive finite IID sigmas.
        Disabling distribution value validation avoids GPU synchronization.
        """
        eigen_sigma = self._eigen_sigma(mean, iid_sigma, eigen_sigma)
        with torch.autocast(device_type=mean.device.type, enabled=False):
            if state is not None and self.correlation > 0:
                self._check_state(mean, state)
                mean = mean + (self.correlation * eigen_sigma * state) @ self.basis
                eigen_sigma = eigen_sigma * math.sqrt(1 - self.correlation**2)
            factor = self.basis.T * eigen_sigma.unsqueeze(-2)
            return LowRankMultivariateNormal(
                mean, factor, iid_sigma.square(), validate_args=False
            )

    def initial_state(self, batch_shape=()) -> torch.Tensor:
        """Stationary Eigen state for new episodes, shape (*batch_shape, rank)."""
        return torch.randn(
            (*batch_shape, self.basis.shape[0]),
            device=self.basis.device,
            dtype=self.basis.dtype,
        )

    def sample(
        self,
        mean: torch.Tensor,
        iid_sigma: torch.Tensor,
        state: torch.Tensor,
        *,
        eigen_sigma: Optional[torch.Tensor] = None,
    ):
        """Sample one action given the previous Eigen state.

        Returns ``(action, next_state)``. Store ``state`` with the action and
        pass it to ``distribution`` for the action's likelihood.
        """
        eigen_sigma = self._eigen_sigma(mean, iid_sigma, eigen_sigma)
        self._check_state(mean, state)
        with torch.no_grad(), torch.autocast(
            device_type=mean.device.type, enabled=False
        ):
            rho = self.correlation
            next_state = rho * state + math.sqrt(1 - rho**2) * torch.randn_like(state)
            iid = iid_sigma * torch.randn_like(mean)
            action = mean + iid + (eigen_sigma * next_state) @ self.basis
        return action, next_state

    def _eigen_sigma(self, mean, iid_sigma, eigen_sigma):
        if mean.ndim < 1 or mean.shape[-1] != self.basis.shape[1]:
            raise ValueError("mean width differs from basis action dimension")
        if iid_sigma.shape != mean.shape and iid_sigma.shape != (mean.shape[-1],):
            raise ValueError("iid_sigma must have mean shape or one value per action")
        if eigen_sigma is None:
            eigen_sigma = self.log_scale.exp()
        if eigen_sigma.shape not in (
            (self.basis.shape[0],),
            (*mean.shape[:-1], self.basis.shape[0]),
        ):
            raise ValueError("eigen_sigma must match the mean batch and basis rank")
        for value in (mean, iid_sigma, eigen_sigma):
            if value.device != self.basis.device or value.dtype != self.basis.dtype:
                raise ValueError("inputs must match EigenDExplore device and dtype")
        return eigen_sigma

    def _check_state(self, mean, state):
        if state.shape != (*mean.shape[:-1], self.basis.shape[0]):
            raise ValueError("state must match the mean batch and basis rank")
        if state.device != self.basis.device or state.dtype != self.basis.dtype:
            raise ValueError("state must match EigenDExplore device and dtype")

    def policy_kl(
        self,
        mean,
        iid_sigma,
        eigen_sigma,
        reference_mean,
        reference_iid_sigma,
        reference_eigen_sigma,
        reduce=True,
        state=None,
    ):
        """Exact KL(current || reference), including both covariance terms.

        Hosts retain per-sample reference scales alongside their mean and IID
        scale snapshots. Recurrent hosts request unreduced values for masking.
        With ``correlation`` > 0, both policies condition on the stored state.
        """
        with torch.autocast(device_type=mean.device.type, enabled=False):
            current = self.distribution(
                mean, iid_sigma, eigen_sigma=eigen_sigma, state=state
            )
            reference = self.distribution(
                reference_mean,
                reference_iid_sigma,
                eigen_sigma=reference_eigen_sigma,
                state=state,
            )
            result = kl_divergence(current, reference)
        return result.mean() if reduce else result
