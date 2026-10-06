"""rl_games registry integration; no edits to its sampler are required."""

import copy

import torch
from rl_games.algos_torch import model_builder, models, network_builder

from .distribution import EigenDExplore


class EigenDExploreNetworkBuilder(network_builder.A2CBuilder):
    def build(self, name, **kwargs):
        params = copy.deepcopy(self.params)
        settings = params.pop("eigendexplore")
        if set(settings) != {"basis", "eigen_sigma", "iid_sigma"}:
            raise ValueError("EigenDExplore requires basis, eigen_sigma, and iid_sigma")
        continuous = params["space"]["continuous"]
        if any(key.startswith("noise_") for key in continuous):
            raise ValueError(
                "EigenDExplore cannot be combined with another noise mechanism"
            )
        if continuous.get("sigma_activation") not in (None, "None", "none"):
            raise ValueError(
                "EigenDExplore requires the host's log-standard-deviation output"
            )
        net = network_builder.A2CBuilder.Network(params, **kwargs)
        basis = torch.tensor(settings["basis"], dtype=torch.float32)
        if basis.shape[1] != kwargs["actions_num"]:
            raise ValueError("EigenDExplore basis width differs from host action width")
        eigen_sigma = torch.tensor(settings["eigen_sigma"], dtype=basis.dtype)
        if continuous["fixed_sigma"] == "coef_cond":
            eigen_sigma = eigen_sigma.expand(len(net.sigma_ids), -1).clone()
        net.eigendexplore = EigenDExplore(basis, eigen_sigma)
        iid = torch.tensor(settings["iid_sigma"], dtype=basis.dtype)
        if iid.shape != (basis.shape[1],) or not bool(
            torch.isfinite(iid).all() & (iid > 0).all()
        ):
            raise ValueError(
                "iid_sigma must be finite, positive, and match action width"
            )
        net.eigendexplore_batch_constant_sigma = (
            isinstance(net.sigma, torch.nn.Parameter) and net.sigma.ndim == 1
        )
        net.register_buffer("eigendexplore_iid_multiplier", iid)
        return net


class EigenDExploreModel(models.ModelA2CContinuousLogStd):
    class Network(models.ModelA2CContinuousLogStd.Network):
        def forward(self, input_dict):
            inputs = dict(input_dict)
            inputs["obs"] = self.norm_obs(inputs["obs"])
            mean, logstd, value, states = self.a2c_network(inputs)
            # AMP accelerates the policy network; covariance solves and probabilities
            # retain the module's dtype, with gradients through the explicit casts.
            with torch.autocast(device_type=mean.device.type, enabled=False):
                dtype = self.a2c_network.eigendexplore.basis.dtype
                mean = mean.to(dtype=dtype)
                logstd = logstd.to(dtype=dtype)
                # Fixed-sigma hosts share one factorization across the rollout batch.
                shared_logstd = (
                    logstd[0]
                    if self.a2c_network.eigendexplore_batch_constant_sigma
                    else logstd
                )
                sigma = (
                    shared_logstd.exp() * self.a2c_network.eigendexplore_iid_multiplier
                )
                net = self.a2c_network
                eigen_sigma = net.eigendexplore.log_scale.exp()
                if eigen_sigma.ndim == 2:
                    ids = (
                        (
                            inputs["obs"][:, net.sigma_id_idx].unsqueeze(-1)
                            == net.sigma_ids
                        )
                        .float()
                        .argmax(dim=-1)
                    )
                    eigen_sigma = eigen_sigma[ids]
                distribution = net.eigendexplore.distribution(
                    mean, sigma, eigen_sigma=eigen_sigma
                )
                result = {
                    "mus": mean,
                    "sigmas": sigma.expand_as(mean),
                    "eigen_sigmas": eigen_sigma.expand(
                        *mean.shape[:-1], net.eigendexplore.basis.shape[0]
                    ),
                    "rnn_states": states,
                }
                if inputs.get("is_train", True):
                    result.update(
                        prev_neglogp=-distribution.log_prob(inputs["prev_actions"]),
                        entropy=distribution.entropy(),
                        values=value,
                    )
                else:
                    action = distribution.sample()
                    result.update(
                        actions=action,
                        neglogpacs=-distribution.log_prob(action),
                        values=self.denorm_value(value),
                    )
            return result


def register():
    """Register before constructing the host's Runner or ModelBuilder."""
    model_builder.register_network("eigendexplore", EigenDExploreNetworkBuilder)
    model_builder.register_model("eigendexplore", EigenDExploreModel)


def configure(params, settings):
    """Configure rl_games with a settings dictionary before Runner.load()."""
    register()
    params["model"]["name"] = "eigendexplore"
    params["network"]["name"] = "eigendexplore"
    params["network"]["eigendexplore"] = settings
