# Banana Navigation with Deep Q-Networks

A PyTorch implementation of Deep Q-Learning (plus Double DQN and Dueling DQN) that trains an agent to collect yellow bananas and avoid blue ones in a Unity ML-Agents environment.

![Comparison of DQN variants](comparison.png)

| Variant | Solved in | Best 100-ep avg | Greedy eval (100 eps) |
|---|---|---|---|
| Vanilla DQN | 435 episodes | 16.81 | – |
| **Double + Dueling DQN** (submitted) | **408 episodes** | **17.15** | **16.23** |

## Project details

The agent moves through a large, square arena scattered with bananas.

- **Reward:** +1 for collecting a yellow banana, −1 for collecting a blue banana.
- **State space:** 37 continuous values: the agent's velocity plus ray-based perception of objects in front of it. There are 7 rays, and each reports which object type it hits (yellow banana, blue banana, wall, or another agent) and the distance to it.
- **Action space:** 4 discrete actions:
  - `0`: move forward
  - `1`: move backward
  - `2`: turn left
  - `3`: turn right
- **Episodes:** fixed length (300 steps).
- **Solved when:** the agent averages a score of **+13 or more over 100 consecutive episodes**.

## Getting started

### 1. Python environment

The Unity environment uses the legacy `unityagents` 0.4 API, which requires **Python 3.6**.

```bash
conda create -n drlnd python=3.6 -y
conda activate drlnd
```

> **Apple Silicon (M1/M2/M3) Macs:** Python 3.6 and the Banana build are Intel-only, so create an x86_64 environment that runs under Rosetta:
> ```bash
> softwareupdate --install-rosetta --agree-to-license
> CONDA_SUBDIR=osx-64 conda create -n drlnd -c conda-forge --override-channels python=3.6 -y
> conda activate drlnd
> conda config --env --set subdir osx-64
> ```

### 2. Install `unityagents` and dependencies

Install the `unityagents` package from Udacity's course repository:

```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
# torch 0.4.0 is no longer on PyPI, and tensorflow is not needed for this project:
sed -i.bak 's/torch==0.4.0/torch==1.10.2/; /tensorflow/d' requirements.txt
pip install .
cd ../..
```

Then clone this repository and install its requirements:

```bash
git clone https://github.com/pprasad-byte/banana-dqn.git
cd banana-dqn
pip install -r requirements.txt
```

### 3. Download the Unity environment

Download the build for your OS, and unzip it into the root of this repository:

| OS | Download |
|---|---|
| Linux | [Banana_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) |
| Linux (headless) | [Banana_Linux_NoVis.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) |
| macOS | [Banana.app.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) |
| Windows (32-bit) | [Banana_Windows_x86.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) |
| Windows (64-bit) | [Banana_Windows_x86_64.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) |

> **macOS notes:**
> - Clear the Gatekeeper quarantine so the app can launch: `xattr -cr Banana.app`
> - If the Unity window opens full-screen and dark (and the Python side hangs), force windowed mode, then retry:
>   ```bash
>   for D in "com.Company.ProductName" "unity.DefaultCompany.Unity Environment"; do
>     defaults write "$D" "Screenmanager Is Fullscreen mode" -integer 0
>     defaults write "$D" "Screenmanager Resolution Width" -integer 800
>     defaults write "$D" "Screenmanager Resolution Height" -integer 600
>   done
>   ```
> - Training runs headless by default, so this only matters for watching the agent.

On Linux or Windows, pass the path to the executable with `--env`, for example `--env Banana_Linux/Banana.x86_64`.

## Instructions

### Watch the trained agent

```bash
python watch.py                       # submitted Double+Dueling model, 3 episodes in real time
python watch.py --episodes 100 --fast # quick 100-episode greedy evaluation
python watch.py --checkpoint runs/vanilla/checkpoint.pth   # the vanilla baseline
```

### Train an agent

```bash
python train.py                        # vanilla DQN
python train.py --double --dueling     # Double + Dueling DQN
python train.py --stop-at-solve        # stop as soon as the 100-episode average reaches +13
python train.py --graphics             # show the Unity window while training (slower)
python train.py --help                 # all options (episodes, epsilon schedule, seed, ...)
```

Each run writes to `runs/<variant>/`:
- `checkpoint.pth`: the model weights (best 100-episode average after solving)
- `scores.npy`: the score for every episode
- `scores.png`: the learning curve
- `results.json`: the solve episode, best and final averages, and training time

Training takes about 35 minutes for 1,000 episodes on a laptop CPU. To run two trainings at once, give each a different `--worker-id`.

### Compare runs

```bash
python compare.py    # overlays every run in runs/ -> comparison.png
```

## Repository structure

| File | Purpose |
|---|---|
| `model.py` | Q-network: a 37 → 64 → 64 → 4 MLP with an optional dueling head |
| `dqn_agent.py` | DQN agent: experience replay, soft-updated target network, optional Double DQN |
| `train.py` | Training loop with the solve check, checkpointing, plots and results |
| `watch.py` | Loads a checkpoint and plays greedily |
| `compare.py` | Plots learning curves of all runs together |
| `checkpoint.pth` | **Submitted model weights** (Double + Dueling DQN) |
| `runs/` | Per-run checkpoints, scores, plots and results |
| `Report.md` | Algorithm details, hyperparameters, results and future work |

## Acknowledgements

The environment and project come from Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), built on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).
