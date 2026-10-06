"""Fixed-scale correlated proposals for a sampling optimizer, with wrist channels."""

import torch

from action_bench import load_eigendexplore, load_hand_layout


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    layout = load_hand_layout("allegro_right")
    # Policy/controller order can differ from the PCA and contain extra channels.
    names = ("wrist_pitch", *reversed(layout.joint_names), "wrist_yaw")
    noise = load_eigendexplore("allegro", action_joint_names=names)
    noise.requires_grad_(False)  # A sampling optimizer can keep scales fixed.
    center = torch.zeros(12, len(names))
    iid_sigma = torch.ones_like(center)
    iid_sigma[:, 1:-1] = 0.5**0.5  # Fingers; wrists have no Eigen noise and keep sigma 1.
    goal = torch.full_like(center, 0.2)
    with torch.no_grad():
        for update in range(3):
            proposals = noise.distribution(center, iid_sigma).sample((256,))
            # Replace this batched toy objective with your environment's rollout.
            costs = (proposals.clamp(-1, 1) - goal).square().mean(dim=(-2, -1))
            center = proposals[costs.argmin()]
            print(f"update={update + 1} best_cost={costs.min().item():.4f}")
    assert not noise.basis[:, [0, -1]].any()  # Wrists retain only IID exploration.


if __name__ == "__main__":
    main()
