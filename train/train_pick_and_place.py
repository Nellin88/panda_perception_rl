import argparse
import os
import sys
import time
from pathlib import Path

# Allow running this script directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer


def parse_args() -> argparse.Namespace:
    # Command-line options for a minimal, reproducible training run.
    parser = argparse.ArgumentParser(description="Minimal FrankaPickAndPlace training with SB3 + HER")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument(
        "--env-id",
        type=str,
        default="FrankaPickAndPlaceSparse-v0",
        help="Gymnasium environment id",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/pick_and_place",
        help="Path prefix to save model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes after training")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render MuJoCo viewer during evaluation",
    )
    parser.add_argument(
        "--train-render",
        action="store_true",
        help="Render MuJoCo viewer during training",
    )
    parser.add_argument(
        "--render-sleep",
        type=float,
        default=0.02,
        help="Sleep time (seconds) between rendered evaluation steps",
    )
    return parser.parse_args()


def make_env(env_id: str, render_mode: str | None = None):
    # Importing panda_mujoco_gym above registers all Franka env IDs.
    return gym.make(env_id, render_mode=render_mode)


def build_model(env, seed: int) -> DDPG:
    # DDPG + HER is a standard baseline for sparse-reward goal-conditioned tasks.
    return DDPG(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            # For each transition, relabel 4 additional goals sampled from future states.
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        learning_rate=1e-3,
        batch_size=256,
        gamma=0.95,
        learning_starts=1000,
        verbose=1,
        seed=seed,
    )


def evaluate(
    model: DDPG,
    env_id: str,
    episodes: int,
    render: bool = False,
    render_sleep: float = 0.02,
) -> float:
    # Run deterministic rollouts and compute task success rate.
    env = make_env(env_id, render_mode="human" if render else None)
    success_count = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render and render_sleep > 0:
                time.sleep(render_sleep)
        # The environment reports binary success in info["is_success"].
        success_count += int(info.get("is_success", 0.0) > 0.5)

    env.close()
    return success_count / episodes


def main() -> None:
    args = parse_args()

    # Create parent directory for model checkpoints if needed.
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Build env/model, train, then persist to disk.
    env = make_env(args.env_id, render_mode="human" if args.train_render else None)
    model = build_model(env, seed=args.seed)

    model.learn(total_timesteps=args.timesteps, log_interval=10)
    model.save(str(save_path))

    # Quick post-training sanity check.
    success_rate = evaluate(
        model,
        args.env_id,
        args.eval_episodes,
        render=args.render,
        render_sleep=args.render_sleep,
    )
    print(f"Saved model to: {save_path}.zip")
    print(f"Eval success rate over {args.eval_episodes} episodes: {success_rate:.2f}")

    env.close()


if __name__ == "__main__":
    main()
