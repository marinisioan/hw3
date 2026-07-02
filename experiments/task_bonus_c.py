"""
Bonus C — Continuous A2C on LunarLanderContinuous-v2
======================================================
Επέκταση του A2C σε συνεχή action space.

LunarLanderContinuous-v2:
  State space  : 8 διαστάσεις (ίδιο με discrete)
  Action space : Box([-1,-1], [1,1]) — 2 συνεχείς τιμές
    action[0]: main engine thrust  ∈ [-1, 1]
    action[1]: lateral engine      ∈ [-1, 1]

Σύγκριση με discrete LunarLander:
  - Η continuous έκδοση έχει άπειρο action space → δεν μπορεί να χρησιμοποιηθεί DQN
  - Ο A2C με Gaussian policy είναι φυσική επέκταση
  - Τυπικά πιο δύσκολη σύγκλιση από discrete λόγω μεγαλύτερου action space

Τρέξιμο:
  python experiments/task_bonus_c.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import gymnasium as gym

from agents  import ContinuousA2CAgent
from utils   import set_seed, get_device, plot_learning_curves

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENV_NAME  = "LunarLanderContinuous-v3"
MAX_STEPS = 1_000_000   # continuous control χρειάζεται περισσότερα steps
SEEDS     = [42, 123, 777]

A2C_CONTINUOUS_CONFIG = dict(
    hidden_dim    = 256,    # μεγαλύτερο network για continuous control
    lr            = 3e-4,
    gamma         = 0.99,
    entropy_coef  = 0.001,  # μικρό β: η Gaussian έχει ήδη built-in exploration (std)
    critic_coef   = 0.5,
)


def train_continuous_a2c(seed: int, max_steps: int) -> dict:
    """
    Εκπαιδεύει ContinuousA2CAgent στο LunarLanderContinuous-v2.
    Η δομή είναι ίδια με τον discrete A2C — μόνο ο agent αλλάζει.
    """
    set_seed(seed)
    device = get_device()
    env    = gym.make(ENV_NAME)

    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.shape[0]         # 2

    agent = ContinuousA2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **A2C_CONTINUOUS_CONFIG,
    )

    returns_log = []
    steps_log   = []
    episode     = 0

    print(f"  State dim: {state_dim} | Action dim: {action_dim} (continuous)")
    print(f"  Action bounds: {env.action_space.low} → {env.action_space.high}")

    while agent.total_steps < max_steps:
        state, _ = env.reset(seed=seed + episode)
        ep_return = 0.0

        while True:
            # select_action επιστρέφει numpy array αντί για integer
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward, done)
            state     = next_state
            ep_return += reward

            if done:
                break

        losses = agent.update(last_state=None, last_done=True)

        episode += 1
        returns_log.append(ep_return)
        steps_log.append(agent.total_steps)

        if episode % 50 == 0:
            recent = np.mean(returns_log[-20:]) if len(returns_log) >= 20 else np.mean(returns_log)
            print(f"  [Continuous A2C | Seed {seed}] Ep {episode} | "
                  f"Steps {agent.total_steps:>7} | Avg-20: {recent:>8.1f} | "
                  f"Entropy: {losses['entropy']:.4f}")

    env.close()
    return {"steps": steps_log, "returns": returns_log, "agent": agent}


if __name__ == "__main__":
    print("=" * 60)
    print("Bonus C: Continuous A2C")
    print(f"  Environment : {ENV_NAME}")
    print(f"  Max Steps   : {MAX_STEPS:,}")
    print(f"  Seeds       : {SEEDS}")
    print("=" * 60)

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    all_runs = []
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        run = train_continuous_a2c(seed=seed, max_steps=MAX_STEPS)
        all_runs.append(run)

        # CSV log
        csv_path = f"results/bonus_c_continuous_a2c_seed{seed}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "episode", "total_steps", "return"])
            for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                writer.writerow([seed, ep + 1, s, r])

    # Checkpoint πρώτου seed
    all_runs[0]["agent"].save("checkpoints/continuous_a2c.pt")

    # Plot — σύγκριση των 3 seeds
    plot_learning_curves(
        runs      = {"Continuous A2C": all_runs},
        title     = f"Bonus C: Continuous A2C — {ENV_NAME}",
        save_path = "results/bonus_c_continuous_a2c.png",
        colors    = {"Continuous A2C": "#00BCD4"},
        window    = 20,
    )

    # Σύνοψη
    print("\n── Final Results ──")
    finals = [np.mean(r["returns"][-20:]) for r in all_runs if len(r["returns"]) >= 20]
    if finals:
        print(f"  Mean ± Std: {np.mean(finals):.1f} ± {np.std(finals):.1f}")
    print(f"  (Solved threshold: ~200)")
