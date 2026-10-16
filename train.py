"""Train a DQN agent on the Unity Banana environment.

Examples:
    python train.py                         # vanilla DQN, headless
    python train.py --double --dueling      # Double + Dueling DQN
    python train.py --graphics              # watch while training (slower)
"""
import argparse
import json
import os
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from dqn_agent import Agent

SOLVE_SCORE = 13.0
WINDOW = 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="Banana.app", help="path to the Banana build")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-t", type=int, default=1000, help="max steps per episode")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay", type=float, default=0.995)
    p.add_argument("--double", action="store_true", help="use Double DQN targets")
    p.add_argument("--dueling", action="store_true", help="use a dueling network head")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worker-id", type=int, default=1)
    p.add_argument("--graphics", action="store_true", help="show the Unity window")
    p.add_argument("--stop-at-solve", action="store_true",
                   help="stop as soon as the 100-episode average reaches +13")
    return p.parse_args()


def main():
    args = parse_args()
    variant = "+".join([n for n, on in [("Double", args.double),
                                        ("Dueling", args.dueling)] if on]) or "Vanilla"
    out_dir = os.path.join("runs", variant.lower().replace("+", "_"))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pth")

    env = UnityEnvironment(file_name=args.env, worker_id=args.worker_id,
                           no_graphics=not args.graphics, seed=args.seed)
    brain_name = env.brain_names[0]
    action_size = env.brains[brain_name].vector_action_space_size
    state_size = len(env.reset(train_mode=True)[brain_name].vector_observations[0])

    agent = Agent(state_size, action_size, seed=args.seed,
                  double=args.double, dueling=args.dueling)
    print("Training {} DQN | state={} actions={} -> {}".format(
        variant, state_size, action_size, out_dir))

    scores, window = [], deque(maxlen=WINDOW)
    eps = args.eps_start
    solved_episode, best_avg = None, -np.inf
    start = time.time()

    try:
        for i_episode in range(1, args.episodes + 1):
            state = env.reset(train_mode=True)[brain_name].vector_observations[0]
            score = 0.0
            for _ in range(args.max_t):
                action = agent.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores.append(score)
            window.append(score)
            eps = max(args.eps_end, args.eps_decay * eps)
            avg = np.mean(window)

            end = "\n" if i_episode % 100 == 0 else ""
            print("\rEpisode {}\tAverage Score: {:.2f}\teps: {:.3f}".format(
                i_episode, avg, eps), end=end, flush=True)

            if len(window) == WINDOW and avg >= SOLVE_SCORE:
                if solved_episode is None:
                    solved_episode = i_episode - WINDOW
                    print("\nSolved in {} episodes! (avg {:.2f} over episodes {}-{})".format(
                        solved_episode, avg, solved_episode + 1, i_episode))
                if avg > best_avg:
                    best_avg = avg
                    agent.save(ckpt_path)
                if args.stop_at_solve:
                    break
    finally:
        env.close()

    if solved_episode is None:
        agent.save(ckpt_path)
        print("\nNot solved within {} episodes; saved final weights.".format(len(scores)))

    np.save(os.path.join(out_dir, "scores.npy"), np.array(scores))
    results = {
        "variant": variant,
        "episodes_run": len(scores),
        "solved_in_episodes": solved_episode,
        "best_100_episode_avg": None if best_avg == -np.inf else round(float(best_avg), 2),
        "final_100_episode_avg": round(float(np.mean(scores[-WINDOW:])), 2),
        "training_minutes": round((time.time() - start) / 60.0, 1),
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    plot_scores(scores, variant, solved_episode, os.path.join(out_dir, "scores.png"))


def plot_scores(scores, variant, solved_episode, path):
    rolling = [np.mean(scores[max(0, i - WINDOW + 1):i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(scores) + 1), scores, alpha=0.35, label="episode score")
    ax.plot(np.arange(1, len(scores) + 1), rolling, linewidth=2, label="100-episode average")
    ax.axhline(SOLVE_SCORE, color="gray", linestyle="--", label="solve threshold (+13)")
    if solved_episode is not None:
        ax.axvline(solved_episode + WINDOW, color="green", linestyle=":",
                   label="solved ({} episodes)".format(solved_episode))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("{} DQN on Banana".format(variant))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("Saved plot to", path)


if __name__ == "__main__":
    main()
