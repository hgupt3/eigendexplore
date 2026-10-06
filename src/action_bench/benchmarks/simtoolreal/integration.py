"""Configure the existing host action/observation hooks before environment creation."""

from action_bench.exploration import policy_noise
from action_bench.settings import load_study

from .adapter import SimToolRealAdapter


def configure_training(env_cfg, agent_cfg, study_path):
    study = load_study(study_path)
    # IsaacLab rejects a Hydra int override for a config field initialized to None.
    # Set the physical environment seed from the resolved native trainer seed.
    env_cfg.seed = agent_cfg["params"]["seed"]
    adapter = SimToolRealAdapter(study).wire(env_cfg, device=env_cfg.sim.device)
    if study.method == "eigendexplore":
        from eigendexplore.rl_games import configure

        configure(agent_cfg["params"], policy_noise(study, prefix=7))
    return adapter


def validate_environment(env, adapter):
    inner = env.unwrapped
    actual = tuple(inner.robot.joint_names[i] for i in inner._hand_joint_ids)
    if actual != adapter.host_joint_names:
        raise ValueError("live hand joint order differs from adapter")
    if inner.single_action_space.shape != (7 + adapter.action_dim,):
        raise ValueError("live policy action width differs from adapter")
    if (
        inner.cfg.observation_space != 140 + adapter.latent_obs_dim
        or inner.cfg.state_space != 162 + adapter.latent_obs_dim
    ):
        raise ValueError("live actor/critic observation widths differ from adapter")
