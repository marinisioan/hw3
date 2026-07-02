# HW3 — Deep Reinforcement Learning

## Environment Choices

|Task|Environment|Justification|
|-|-|-|
|Task 1 (Tabular baseline)|CartPole-v1|4D continuous state → 6⁴=1296 discrete states; fast CPU training; clear curse-of-dimensionality demonstration|
|Tasks 2–4a (DQN, REINFORCE, A2C)|LunarLander-v3|8D continuous state, discrete actions (4), richer dynamics; representative continuous-state RL benchmark|
|Task 4b (Hyperparameter study)|CartPole-v1|Fast feedback loop for sweeping 4×3=12 configurations without GPU|
|Bonus B (Double DQN)|LunarLander-v3|Same as Task 2; comparison integrated into ablation plot|
|Bonus C (Continuous A2C)|LunarLanderContinuous-v3|Box(2,) continuous action space; natural extension of LunarLander|
|Bonus A (GRPO)|GSM8K (arithmetic)|Rule-based verifiable reward; no reward model needed|

## Setup

**Python:** 3.10  
**GPU:** NVIDIA RTX 3070 (8 GB VRAM) — required for Bonus A; all other tasks run on CPU or GPU.

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. LunarLander physics backend
pip install gymnasium\[box2d]

# 3. Bonus A only — authenticate to download Qwen2.5-0.5B-Instruct
pip install transformers==4.44.0 datasets==2.21.0 accelerate==0.33.0 huggingface-hub==0.24.5
huggingface-cli login
```

## How to Run

All experiment scripts are under `hw3/experiments/`. Run from that directory.

```bash
cd hw3/experiments
```
\*\*Directory Structure \& Results:\*\* Executing the scripts automatically creates a `results/` subdirectory inside `experiments/`, where all generated plots and CSV files are saved. Model weights (checkpoints) are saved in `experiments/checkpoints/`.



\*\*Visualization (LunarLander only):\*\* To visualize (render) the trained agents exclusively in the LunarLander environments (discrete/continuous), you can load the `.pt` files from the `checkpoints/` folder by initializing the environment with `render\_mode="human"`.



### Task 1 — Tabular Q-Learning (CartPole-v1)

```bash
python task1\_tabular.py
```

Outputs: `results/task1\_tabular\_qlearning.png`, `results/task1\_tabular\_qlearning.csv`

\---

### Task 2 — DQN Ablation Study (LunarLander-v3)

Runs all four variants (DQN-full, DQN-noReplay, DQN-noTarget, DQN-double) × 3 seeds.

```bash
python task2\_dqn.py
```

Outputs: `results/task2\_dqn\_ablation.png`, `results/task2\_dqn\_epsilon.png`, `results/task2\_dqn\_\*.csv`, `checkpoints/dqn\_full.pt`, `checkpoints/dqn\_double.pt`

\---

### Task 3 — REINFORCE \& A2C (LunarLander-v3)

```bash
python task3\_policy\_gradient.py
```

Outputs: `results/task3\_reinforce.png`, `results/task3\_a2c.png`, `results/task3\_\*.csv`, `checkpoints/reinforce.pt`, `checkpoints/a2c.pt`

\---

### Task 4a — Algorithm Comparison

Loads existing CSVs from Tasks 1–3 if available; re-trains only missing results.

```bash
python task4\_comparison.py --part 4a
```

Output: `results/task4\_comparison.png`

### Task 4b — Hyperparameter Study (CartPole-v1)

Study A: A2C entropy coefficient β ∈ {0.0, 0.001, 0.01, 0.05}  
Study B: REINFORCE discount γ ∈ {0.90, 0.95, 0.99, 1.00}

```bash
python task4\_comparison.py --part 4b
```

Outputs: `results/task4\_hyperparam\_study.png` (A2C β, required filename), `results/task4\_hyperparam\_study\_reinforce\_gamma.png`

\---

### Bonus B — Double DQN

Included automatically in Task 2 as the fourth ablation variant (`mode="double"`). No separate script needed.

\---

### Bonus C — Continuous A2C (LunarLanderContinuous-v3)

```bash
python task\_bonus\_c.py
```

Output: `results/bonus\_c\_continuous\_a2c.png`, `checkpoints/continuous\_a2c.pt`

\---

### Bonus A — GRPO Fine-tuning (requires GPU ≥ 4 GB VRAM)

```bash
cd ../bonus\_a
python task\_bonus\_a.py   # trains Qwen2.5-0.5B-Instruct on GSM8K, \~30–60 min on RTX 3070
python plot\_grpo.py      # generates PNG from saved CSV
```

Outputs: `bonus\_a/results/task\_bonus\_a\_log.csv`, `bonus\_a/results/bonus\_a\_grpo\_training.png`

\---

## Required Plots

|File|Task|
|-|-|
|`results/task1\_tabular\_qlearning.png`|Task 1|
|`results/task2\_dqn\_ablation.png`|Task 2|
|`results/task2\_dqn\_epsilon.png`|Task 2|
|`results/task3\_reinforce.png`|Task 3a|
|`results/task3\_a2c.png`|Task 3b|
|`results/task4\_comparison.png`|Task 4a|
|`results/task4\_hyperparam\_study.png`|Task 4b|

## Results Summary

|Algorithm|Environment|Final Return (mean ± std)|
|-|-|-|
|Tabular Q-Learning|CartPole-v1|168.7 ± 41.0|
|DQN-full|LunarLander-v3|189.4 ± 19.5|
|DQN-double (Bonus B)|LunarLander-v3|153.1 ± 67.9|
|REINFORCE|LunarLander-v3|-175.2 ± 19.8|
|A2C|LunarLander-v3|-507.5 ± 251.7|
|Continuous A2C (Bonus C)|LunarLanderContinuous-v3|-357.7 ± 190.0|



