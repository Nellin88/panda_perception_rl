import argparse
import os
import sys
import time
from pathlib import Path

# Allow running this file directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import DDPG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Reach policy")
    parser.add_argument("--model-path", type=str, default="outputs/reach.zip", help="Path to saved SB3 model")
    parser.add_argument(
        "--env-id",
        type=str,
        default="FrankaReachSparse-v0",
        help="Gymnasium environment id",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render MuJoCo viewer during rollout")
    parser.add_argument(
        "--render-sleep",
        type=float,
        default=0.02,
        help="Sleep time (seconds) between rendered steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = gym.make(args.env_id, render_mode="human" if args.render else None)
    # HER replay buffer models must be loaded with an environment.
    model = DDPG.load(str(model_path), env=env, device="cpu")

    successes = 0
    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)

            if args.render and args.render_sleep > 0:
                time.sleep(args.render_sleep)

        successes += int(info.get("is_success", 0.0) > 0.5)
        returns.append(ep_return)
        print(f"Episode {ep + 1}/{args.episodes}: return={ep_return:.3f}, success={info.get('is_success', 0.0)}")

    env.close()

    mean_return = sum(returns) / len(returns) if returns else 0.0
    success_rate = successes / args.episodes if args.episodes > 0 else 0.0
    print(f"Model: {model_path}")
    print(f"Mean return: {mean_return:.3f}")
    print(f"Success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
