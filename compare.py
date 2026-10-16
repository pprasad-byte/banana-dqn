"""Overlay the 100-episode rolling average of each training run.

Usage: python compare.py   (reads runs/*/scores.npy, writes comparison.png)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WINDOW = 100
SOLVE_SCORE = 13.0


def rolling(x, w=WINDOW):
    return np.array([np.mean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


fig, ax = plt.subplots(figsize=(9, 5))
for run_dir in sorted(glob.glob("runs/*/")):
    scores = np.load(os.path.join(run_dir, "scores.npy"))
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    solved = res["solved_in_episodes"]
    label = "{} (solved in {})".format(res["variant"], solved)
    line, = ax.plot(np.arange(1, len(scores) + 1), rolling(scores), linewidth=2, label=label)
    if solved is not None:
        ax.axvline(solved + WINDOW, color=line.get_color(), linestyle=":", alpha=0.7)

ax.axhline(SOLVE_SCORE, color="gray", linestyle="--", label="solve threshold (+13)")
ax.set_xlabel("Episode")
ax.set_ylabel("100-episode average score")
ax.set_title("DQN variants on Banana")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("comparison.png", dpi=120)
print("Saved comparison.png")
