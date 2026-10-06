"""Small CPU PPO run with a packaged hand PCA and a batched toy environment.

This demonstrates the distribution boundary; it is not a benchmark result.
Run from the checkout after installing both release packages.
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from action_bench import load_basis, load_eigendexplore


class Policy(nn.Module):
    def __init__(self, hand, k=None, correlation=0.0):
        super().__init__()
        self.noise = load_eigendexplore(hand, k=k, correlation=correlation)
        width = self.noise.basis.shape[1]
        self.actor = nn.Sequential(
            nn.Linear(2 * width, 64), nn.Tanh(), nn.Linear(64, width)
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * width, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        # IID sigma sqrt(0.5) with the loader defaults performs well across settings.
        self.log_std = nn.Parameter(torch.full((width,), 0.5 * math.log(0.5)))

    def forward(self, observations, noise_state):
        mean = self.actor(observations)
        distribution = self.noise.distribution(
            mean, self.log_std.exp(), state=noise_state
        )
        return mean, distribution, self.critic(observations).squeeze(-1)


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    policy = Policy(args.hand, k=args.k, correlation=args.correlation)
    # Includes the mean, value, IID log scales, and Eigen log scales.
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    width = policy.noise.basis.shape[1]
    state = torch.zeros(args.num_envs, width)
    goal = 0.6 * torch.rand_like(state) - 0.3
    # Eigen state of the last action. Redraw it with initial_state() at resets.
    # With --correlation 0 it carries nothing and the noise is i.i.d.
    noise_state = policy.noise.initial_state((args.num_envs,))
    gamma, gae_lambda = 0.99, 0.95
    for update in range(args.updates):
        observations, actions, log_probs, means, noise_states, values, rewards = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with torch.no_grad():
            # Snapshot scales before the rollout. Keep these through every PPO epoch.
            old_iid = policy.log_std.exp().clone()
            old_eigen = policy.noise.log_scale.exp().clone()
            for _ in range(args.steps):
                obs = torch.cat((state, goal), dim=-1)
                mean, distribution, value = policy(obs, noise_state)
                action, next_noise_state = policy.noise.sample(
                    mean, policy.log_std.exp(), noise_state
                )
                # Save the RAW sampled action, its likelihood and the state it was
                # drawn from. Bounds apply only at the environment boundary, never
                # to the action stored for PPO.
                observations.append(obs)
                actions.append(action)
                log_probs.append(distribution.log_prob(action))
                means.append(mean)
                noise_states.append(noise_state)
                values.append(value)
                noise_state = next_noise_state
                state = state + 0.25 * (action.clamp(-1, 1) - state)
                rewards.append(-(state - goal).square().mean(-1))
            _, _, next_value = policy(torch.cat((state, goal), dim=-1), noise_state)
            values = torch.stack(values)
            rewards = torch.stack(rewards)
            advantages = torch.empty_like(rewards)
            gae = torch.zeros_like(next_value)
            for step in reversed(range(args.steps)):
                delta = rewards[step] + gamma * next_value - values[step]
                gae = delta + gamma * gae_lambda * gae
                advantages[step] = gae
                next_value = values[step]
            returns = (advantages + values).flatten()
            advantages = advantages.flatten()
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
            obs = torch.stack(observations).flatten(0, 1)
            action = torch.stack(actions).flatten(0, 1)
            old_log_prob = torch.stack(log_probs).flatten()
            old_mean = torch.stack(means).flatten(0, 1)
            old_noise_state = torch.stack(noise_states).flatten(0, 1)
        for _ in range(2):
            mean, distribution, value = policy(obs, old_noise_state)
            ratio = (distribution.log_prob(action) - old_log_prob).exp()
            surrogate = torch.minimum(
                ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
            )
            loss = (
                -surrogate.mean()
                + 0.5 * (value - returns).square().mean()
                # Entropy of the per-step marginal, not of the conditional.
                - 0.001
                * policy.noise.distribution(mean, policy.log_std.exp()).entropy().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        with torch.no_grad():
            mean, _, _ = policy(obs, old_noise_state)
            kl = policy.noise.policy_kl(
                mean,
                policy.log_std.exp(),
                policy.noise.log_scale.exp(),
                old_mean,
                old_iid,
                old_eigen,
                state=old_noise_state,
            )
        if not torch.isfinite(loss) or not torch.isfinite(kl):
            raise RuntimeError("nonfinite PPO update")
        print(
            f"update={update + 1} reward={rewards.mean().item():.4f} exact_kl={kl.item():.6f}"
        )
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hand": args.hand,
                "seed": args.seed,
                "rank": policy.noise.basis.shape[0],
                "correlation": policy.noise.correlation,
                "basis_id": load_basis(args.hand).manifest.id,
                "action_coordinates": "normalized_absolute_joints",
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            args.checkpoint,
        )
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro")
    parser.add_argument(
        "--k", type=int, help="Retained PCA rank (default: catalog 90% rank)"
    )
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.0,
        help="AR(1) carry-over of the Eigen noise (default 0: i.i.d.)",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.updates) < 1:
        parser.error("num-envs, steps and updates must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be 0 through 4294967295")
    if not 0 <= args.correlation < 1:
        parser.error("correlation must be in [0, 1)")
    run(args)


if __name__ == "__main__":
    main()
