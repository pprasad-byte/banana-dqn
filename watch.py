"""Watch a trained agent play the Banana environment.

Examples:
    python watch.py                                    # submitted model, 3 episodes
    python watch.py --checkpoint runs/vanilla/checkpoint.pth
    python watch.py --episodes 100 --fast              # quick evaluation, no slow-mo
"""
import argparse

import numpy as np
import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="Banana.app", help="path to the Banana build")
    p.add_argument("--checkpoint", default="checkpoint.pth")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--worker-id", type=int, default=1)
    p.add_argument("--fast", action="store_true",
                   help="run at training speed (train_mode=True) for quick evaluation")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    env = UnityEnvironment(file_name=args.env, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    action_size = env.brains[brain_name].vector_action_space_size
    state_size = len(env.reset(train_mode=True)[brain_name].vector_observations[0])

    agent = Agent(state_size, action_size,
                  double=ckpt.get("double", False), dueling=ckpt.get("dueling", False))
    agent.qnetwork_local.load_state_dict(ckpt["state_dict"])
    variant = "+".join([n for n, on in [("Double", agent.double),
                                        ("Dueling", agent.dueling)] if on]) or "Vanilla"
    print("Loaded {} DQN from {}".format(variant, args.checkpoint))

    scores = []
    try:
        for i in range(1, args.episodes + 1):
            state = env.reset(train_mode=args.fast)[brain_name].vector_observations[0]
            score = 0.0
            while True:
                env_info = env.step(agent.act(state, eps=0.0))[brain_name]
                state = env_info.vector_observations[0]
                score += env_info.rewards[0]
                if env_info.local_done[0]:
                    break
            scores.append(score)
            print("Episode {}: score {:.0f}".format(i, score))
    finally:
        env.close()
    print("Mean score over {} episodes: {:.2f}".format(len(scores), np.mean(scores)))


if __name__ == "__main__":
    main()
