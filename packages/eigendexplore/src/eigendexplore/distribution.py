"""One implementation of IID plus low-rank Gaussian exploration."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, constraints, kl_divergence, register_kl
from torch.distributions.utils import lazy_property

_LOG_2PI = math.log(2 * math.pi)


def _eigen_gram(weights, basis):
    """``B @ diag(weights) @ B.T`` for each row of weights, as one matrix product."""
    rank = basis.shape[0]
    outer = (basis.T.unsqueeze(-1) * basis.T.unsqueeze(-2)).reshape(-1, rank * rank)
    return (weights @ outer).unflatten(-1, (rank, rank))


def _invert_lower(tril):
    eye = torch.eye(tril.shape[-1], device=tril.device, dtype=tril.dtype)
    return torch.linalg.solve_triangular(tril, eye, upper=False)


class EigenDExploreDistribution(Distribution):
    """Gaussian with covariance ``diag(iid_sigma**2) + B.T @ diag(eigen_sigma**2) @ B``.

    Likelihood, entropy and KL use the Woodbury identity, so every solve has the
    basis rank. The rank-by-rank factorization follows the scales' own batch
    shape: scales shared across a batch are factorized once, and with ``group``
    each row of per-group scales is factorized once and looked up per sample.
    Nothing synchronizes with the host. Construct it through
    ``EigenDExplore.distribution``.
    """

    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, iid_sigma, eigen_sigma, basis, group=None):
        rank, width = basis.shape
        self.loc, self.basis, self._group = loc, basis, group
        variance = iid_sigma.square()
        # Eigen-space precision of the IID part, B @ diag(iid_sigma**-2) @ B.T.
        self._scale_precision = _eigen_gram(variance.reciprocal(), basis)
        capacitance = (
            eigen_sigma.unsqueeze(-1)
            * self._scale_precision
            * eigen_sigma.unsqueeze(-2)
        )
        capacitance = capacitance + torch.eye(
            rank, device=basis.device, dtype=basis.dtype
        )
        # Identity plus a positive semidefinite matrix: always positive definite.
        self._scale_tril = torch.linalg.cholesky_ex(capacitance).L
        # log det(cov) = sum(log iid variance) + log det(capacitance).
        low_rank = 2 * self._scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        if group is not None:
            iid_sigma, eigen_sigma, low_rank, inverse = self._select(
                iid_sigma,
                eigen_sigma,
                low_rank.unsqueeze(-1),
                _invert_lower(self._scale_tril).flatten(-2),
            )
            low_rank = low_rank.squeeze(-1)
            self._tril_inverse = inverse.unflatten(-1, (rank, rank))
        self.iid_sigma, self.eigen_sigma = iid_sigma, eigen_sigma
        self._iid_variance = iid_sigma.square()
        self._low_rank_log_det = low_rank
        self._log_det = self._iid_variance.log().sum(-1) + low_rank
        batch_shape = torch.broadcast_shapes(
            loc.shape[:-1], iid_sigma.shape[:-1], eigen_sigma.shape[:-1]
        )
        super().__init__(batch_shape, (width,), validate_args=False)

    def _select(self, *tables):
        """Each sample's rows of per-group tables, with one lookup."""
        rows = F.embedding(self._group, torch.cat(tables, -1))
        return rows.split([table.shape[-1] for table in tables], -1)

    @lazy_property
    def _tril_inverse(self):
        return _invert_lower(self._scale_tril)

    @lazy_property
    def _precision(self):
        if self._group is None:
            return self._scale_precision
        rank = self.basis.shape[0]
        (rows,) = self._select(self._scale_precision.flatten(-2))
        return rows.unflatten(-1, (rank, rank))

    def _mahalanobis(self, diff):
        scaled = diff / self._iid_variance
        eigen = self.eigen_sigma * (scaled @ self.basis.T)
        inverse = self._tril_inverse
        if inverse.ndim == 2:
            eigen = eigen @ inverse.mT
        else:  # tiny per-sample matrices: elementwise beats batched products
            eigen = (inverse * eigen.unsqueeze(-2)).sum(-1)
        return (diff * scaled).sum(-1) - eigen.square().sum(-1)

    @property
    def mean(self):
        return self.loc.expand(self.batch_shape + self.event_shape)

    @property
    def mode(self):
        return self.mean

    @property
    def variance(self):
        eigen = self.eigen_sigma.square() @ self.basis.square()
        return (self._iid_variance + eigen).expand(self.batch_shape + self.event_shape)

    @property
    def covariance_matrix(self):
        factor = self.basis.T * self.eigen_sigma.unsqueeze(-2)
        covariance = torch.diag_embed(self._iid_variance) + factor @ factor.mT
        return covariance.expand(self.batch_shape + self.event_shape * 2)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        options = {"device": self.loc.device, "dtype": self.loc.dtype}
        eta = torch.randn(shape[:-1] + self.basis.shape[:1], **options)
        iid = torch.randn(shape, **options)
        return self.loc + (self.eigen_sigma * eta) @ self.basis + self.iid_sigma * iid

    def log_prob(self, value):
        return -0.5 * (
            self.event_shape[0] * _LOG_2PI
            + self._log_det
            + self._mahalanobis(value - self.loc)
        )

    def entropy(self):
        entropy = 0.5 * (self.event_shape[0] * (1 + _LOG_2PI) + self._log_det)
        return entropy.expand(self.batch_shape)


@register_kl(EigenDExploreDistribution, EigenDExploreDistribution)
def _kl_eigendexplore(p, q):
    """Exact KL(p || q) for two distributions over the same basis."""
    if p.basis is not q.basis:
        raise ValueError("KL needs both distributions from the same EigenDExplore")
    # With q_cov**-1 = D**-1 - D**-1 W C**-1 W.T D**-1 (Woodbury), the trace and
    # Mahalanobis terms share one rank-by-rank correction,
    # tr(C**-1 @ W.T D**-1 (p_cov + diff diff.T) D**-1 W), all in q's scales.
    inverse = q._tril_inverse
    diff = p.loc - q.loc
    scaled = diff / q._iid_variance
    eigen = scaled @ q.basis.T
    cross = q._precision * p.eigen_sigma.unsqueeze(-2)
    inner = _eigen_gram(p._iid_variance / q._iid_variance.square(), q.basis)
    inner = inner + cross @ cross.mT + eigen.unsqueeze(-1) * eigen.unsqueeze(-2)
    inner = q.eigen_sigma.unsqueeze(-1) * inner * q.eigen_sigma.unsqueeze(-2)
    correction = ((inverse.mT @ inverse) * inner).sum((-2, -1))
    # The IID terms pair per action as r - 1 - log r, which stays accurate in
    # float32 when the policies are close.
    ratio = p._iid_variance / q._iid_variance
    low_rank = (p.eigen_sigma.square() * q._precision.diagonal(dim1=-2, dim2=-1)).sum(
        -1
    )
    low_rank = low_rank - correction + q._low_rank_log_det - p._low_rank_log_det
    return 0.5 * (
        (ratio - 1 - ratio.log()).sum(-1) + (diff * scaled).sum(-1) + low_rank
    )


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
        group: Optional[torch.Tensor] = None,
    ):
        """Return the exact action-space marginal, with differentiable entropy.

        With ``correlation`` > 0, pass the Eigen state the action was sampled
        from (see ``sample``) to get its likelihood. The mean then shifts by
        ``rho * B.T @ (eigen_sigma * state)`` and the Eigen scales shrink by
        ``sqrt(1 - rho**2)``. Without ``state`` this is the per-step marginal.

        Scales shared across the batch (shape ``(action_dim,)`` and ``(rank,)``)
        cost one rank-by-rank factorization. For several policies, pass their
        scales as rows, shape ``(policies, action_dim)`` and ``(policies, rank)``,
        and ``group``, the integer row of each mean; each row is factorized once.
        The caller guarantees finite means, positive finite IID sigmas and valid
        rows; nothing is checked on the device or synchronizes with the host.
        """
        eigen_sigma = self._eigen_sigma(mean, iid_sigma, eigen_sigma, group)
        if group is not None:
            iid_sigma = iid_sigma.expand(len(eigen_sigma), -1)
        with torch.autocast(device_type=mean.device.type, enabled=False):
            if state is not None and self.correlation > 0:
                self._check_state(mean, state)
                selected = (
                    eigen_sigma if group is None else F.embedding(group, eigen_sigma)
                )
                mean = mean + (self.correlation * selected * state) @ self.basis
                eigen_sigma = eigen_sigma * math.sqrt(1 - self.correlation**2)
            return EigenDExploreDistribution(
                mean, iid_sigma, eigen_sigma, self.basis, group=group
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
        with (
            torch.no_grad(),
            torch.autocast(device_type=mean.device.type, enabled=False),
        ):
            rho = self.correlation
            next_state = rho * state + math.sqrt(1 - rho**2) * torch.randn_like(state)
            iid = iid_sigma * torch.randn_like(mean)
            action = mean + iid + (eigen_sigma * next_state) @ self.basis
        return action, next_state

    def _eigen_sigma(self, mean, iid_sigma, eigen_sigma, group=None):
        if mean.ndim < 1 or mean.shape[-1] != self.basis.shape[1]:
            raise ValueError("mean width differs from basis action dimension")
        if eigen_sigma is None:
            eigen_sigma = self.log_scale.exp()
        rank, width = self.basis.shape
        if group is None:
            if iid_sigma.shape not in (mean.shape, (width,)):
                raise ValueError(
                    "iid_sigma must have mean shape or one value per action"
                )
            if eigen_sigma.shape not in ((rank,), (*mean.shape[:-1], rank)):
                raise ValueError("eigen_sigma must match the mean batch and basis rank")
        else:
            if group.shape != mean.shape[:-1] or group.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("group must hold one integer row per mean")
            if group.device != self.basis.device:
                raise ValueError("group must be on the EigenDExplore device")
            if eigen_sigma.ndim == 1:
                eigen_sigma = eigen_sigma.unsqueeze(0)
            rows = len(eigen_sigma) if iid_sigma.ndim == 1 else len(iid_sigma)
            if iid_sigma.shape not in (
                (width,),
                (rows, width),
            ) or eigen_sigma.shape not in (
                (1, rank),
                (rows, rank),
            ):
                raise ValueError(
                    "with group, scales must be rows of shape (policies, action_dim)"
                    " and (policies, rank)"
                )
            eigen_sigma = eigen_sigma.expand(rows, -1)
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
