"""
Task 1 — Tabular Q-Learning Baseline
======================================
Υλοποίηση κλασικού Q-learning με διακριτοποίηση του state space.

Περιβάλλον: CartPole-v1 (4D continuous state → 6^4 = 1,296 discrete states)

Γιατί baseline;
  Δείχνουμε ότι tabular methods μπορούν αρχικά να μάθουν, αλλά:
  - Δεν generalise: κάθε bin είναι ανεξάρτητος — καμία γενίκευση
  - Δεν scale: LunarLander (8D) × 10 bins = 100M states → απαγορευτικό

Τρέξιμο:
  python experiments/task1_tabular.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import gymnasium as gym

from utils.seed_utils import set_seed
from utils.plotting   import plot_learning_curves

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENV_NAME    = "CartPole-v1"
N_EPISODES  = 2_000
GAMMA       = 0.99
LR          = 0.1          # learning rate για Q-table update
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 1_500        # episodes για γραμμική αποσύνθεση
N_BINS      = 6            # bins ανά διάσταση → 6^4 = 1,296 states
SEEDS       = [42, 123, 777]

# Empirical bounds για CartPole (από observation space + εμπειρία)
# Τα ± inf bounds αντικαθίστανται με πεπερασμένες τιμές
STATE_BOUNDS = [
    [-4.8, 4.8],       # cart position
    [-3.5, 3.5],       # cart velocity (inf clipped)
    [-0.42, 0.42],     # pole angle (~24 degrees)
    [-3.5, 3.5],       # pole angular velocity (inf clipped)
]


# ── Discretisation ─────────────────────────────────────────────────────────────
def build_bins(n_bins: int) -> list:
    """
    Δημιουργεί bins για κάθε state dimension.
    Returns: list of 1D arrays (τα edge points, όχι τα centers)
    """
    bins = []
    for low, high in STATE_BOUNDS:
        bins.append(np.linspace(low, high, n_bins + 1)[1:-1])  # n_bins - 1 edges
    return bins


def discretise(state: np.ndarray, bins: list) -> tuple:
    """
    Μετατρέπει continuous state σε discrete tuple (index ανά dimension).
    np.digitize: returns 0..n_bins, clip σε [0, n_bins-1]
    """
    indices = []
    for val, bin_edges in zip(state, bins):
        idx = np.clip(np.digitize(val, bin_edges), 0, N_BINS - 1)
        indices.append(idx)
    return tuple(indices)


# ── Epsilon Schedule ───────────────────────────────────────────────────────────
def get_epsilon(episode: int) -> float:
    """Γραμμική μείωση ε: από EPS_START → EPS_END σε EPS_DECAY episodes."""
    if episode >= EPS_DECAY:
        return EPS_END
    frac = episode / EPS_DECAY
    return (1.0 - frac) * EPS_START + frac * EPS_END


# ── Training Loop ──────────────────────────────────────────────────────────────
def train_tabular(seed: int) -> dict:
    """
    Εκπαιδεύει έναν tabular Q-learning agent.
    Returns: dict με 'steps' και 'returns' lists (για plotting).
    """
    set_seed(seed)
    env  = gym.make(ENV_NAME)
    bins = build_bins(N_BINS)

    # Q-table: shape [6,6,6,6,2] — τελευταία διάσταση = n_actions
    q_table = np.zeros([N_BINS] * env.observation_space.shape[0] + [env.action_space.n])
    print(f"  Q-table size: {q_table.size} entries  "
          f"({' × '.join([str(N_BINS)] * env.observation_space.shape[0])} × {env.action_space.n})")

    returns_log  = []
    steps_log    = []
    total_steps  = 0

    for episode in range(N_EPISODES):
        state, _ = env.reset(seed=seed + episode)
        state_d  = discretise(state, bins)
        eps      = get_epsilon(episode)
        ep_return = 0.0
        ep_steps  = 0

        while True:
            # ε-greedy action selection
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_d])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_d = discretise(next_state, bins)

            # ── Q-Learning Update ──────────────────────────────────────────────
            # Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a') - Q(s,a)]
            # Αν done: το "next state" δεν υπάρχει → target = r
            if terminated:
                td_target = reward
            else:
                td_target = reward + GAMMA * np.max(q_table[next_d])

            td_error         = td_target - q_table[state_d + (action,)]
            q_table[state_d + (action,)] += LR * td_error

            state_d   = next_d
            ep_return += reward
            ep_steps  += 1
            total_steps += 1

            if done:
                break

        returns_log.append(ep_return)
        steps_log.append(total_steps)

        if (episode + 1) % 200 == 0:
            recent_mean = np.mean(returns_log[-100:])
            print(f"  [Seed {seed}] Episode {episode+1}/{N_EPISODES} | "
                  f"Avg Return (last 100): {recent_mean:.1f} | ε={eps:.3f}")

    env.close()
    return {"steps": steps_log, "returns": returns_log}


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Task 1: Tabular Q-Learning Baseline")
    print(f"  Environment : {ENV_NAME}")
    print(f"  N_BINS      : {N_BINS} per dimension → {N_BINS**4} states")
    print(f"  Seeds       : {SEEDS}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Εκπαίδευση για κάθε seed
    all_runs = []
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        run = train_tabular(seed)
        all_runs.append(run)

    # Αποθήκευση CSV log
    csv_path = "results/task1_tabular_qlearning.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "episode", "total_steps", "return"])
        for seed_idx, (seed, run) in enumerate(zip(SEEDS, all_runs)):
            for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                writer.writerow([seed, ep + 1, s, r])
    print(f"\n  CSV saved: {csv_path}")

    # Plot
    plot_learning_curves(
        runs      = {"Tabular Q-Learning": all_runs},
        title     = f"Task 1: Tabular Q-Learning — {ENV_NAME}",
        save_path = "results/task1_tabular_qlearning.png",
        colors    = {"Tabular Q-Learning": "#E07B39"},
        window    = 50,
    )

    # Στατιστικά για report.md
    all_returns = [r for run in all_runs for r in run["returns"]]
    final_returns = [np.mean(run["returns"][-100:]) for run in all_runs]
    print(f"\n── Final Performance (last 100 episodes) ──")
    print(f"  Mean ± Std : {np.mean(final_returns):.1f} ± {np.std(final_returns):.1f}")
    print(f"  (Solved = 475.0)")
