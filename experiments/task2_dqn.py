"""
Task 2 — DQN + Ablation Study
================================
Κύριο περιβάλλον: LunarLander-v2

Ablation variants:
  DQN-full      : replay buffer ✓ + target network ✓  (το κανονικό DQN)
  DQN-noReplay  : χωρίς replay buffer, online updates
  DQN-noTarget  : χωρίς frozen target (target = online network)

Τι αναμένουμε να δούμε:
  - DQN-full     : σταθερή, ομαλή σύγκλιση
  - DQN-noReplay : πολύ υψηλό noise, πιθανό divergence
                   (correlated updates, δεν σπάει χρονική συσχέτιση)
  - DQN-noTarget : oscillation ή divergence
                   (moving target → feedback loop → "deadly triad")

Τρέξιμο:
  python experiments/task2_dqn.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import gymnasium as gym

from agents     import DQNAgent
from utils      import set_seed, get_device, plot_learning_curves, plot_epsilon

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENV_NAME    = "LunarLander-v3"
MAX_STEPS   = 500_000     # συνολικά env steps ανά seed
SEEDS       = [42, 123, 777]

DQN_CONFIG = dict(
    buffer_capacity  = 50_000,
    batch_size       = 64,
    min_buffer_size  = 1_000,
    hidden_dim       = 128,
    lr               = 5e-4,
    gamma            = 0.99,
    target_update_freq = 200,
    eps_start        = 1.0,
    eps_end          = 0.05,
    eps_decay_steps  = 200_000,
)

# Variants για το ablation
VARIANTS = {
    "DQN-full"      : "full",
    "DQN-noReplay"  : "no_replay",
    "DQN-noTarget"  : "no_target",
    "DQN-double"    : "double",    # Bonus B
}

VARIANT_COLORS = {
    "DQN-full"     : "#2196F3",
    "DQN-noReplay" : "#F44336",
    "DQN-noTarget" : "#9C27B0",
    "DQN-double"   : "#009688",   # teal
}


# ── Training Loop ──────────────────────────────────────────────────────────────
def train_dqn(seed: int, mode: str, max_steps: int) -> dict:
    """
    Εκπαιδεύει DQN agent για max_steps steps.

    Args:
        seed     : random seed για reproducibility
        mode     : "full" | "no_replay" | "no_target"
        max_steps: συνολικά environment steps

    Returns: dict με 'steps' και 'returns' lists
    """
    set_seed(seed)
    device = get_device()
    env    = gym.make(ENV_NAME)

    state_dim = env.observation_space.shape[0]   # 8 για LunarLander
    n_actions = env.action_space.n               # 4 για LunarLander

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        mode=mode,
        device=device,
        **DQN_CONFIG,
    )

    returns_log = []
    steps_log   = []

    state, _ = env.reset(seed=seed)
    ep_return  = 0.0
    episode    = 0

    print(f"  Pre-filling buffer ({DQN_CONFIG['min_buffer_size']} transitions)...")

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        loss = agent.step(state, action, reward, next_state, done)
        state     = next_state
        ep_return += reward

        if done:
            episode += 1
            returns_log.append(ep_return)
            steps_log.append(agent.total_steps)

            if episode % 50 == 0:
                recent = np.mean(returns_log[-20:]) if len(returns_log) >= 20 else np.mean(returns_log)
                print(f"  [Seed {seed} | {mode}] Ep {episode} | Steps {step:>6} | "
                      f"Avg-20 Return: {recent:>8.1f} | ε={agent.eps:.3f}")

            ep_return = 0.0
            state, _ = env.reset(seed=seed + episode)

    env.close()
    return {
        "steps"       : steps_log,
        "returns"     : returns_log,
        "eps_history" : agent.eps_history,
        "agent"       : agent,   # ο εκπαιδευμένος agent, για αποθήκευση checkpoint
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Task 2: DQN Ablation Study")
    print(f"  Environment : {ENV_NAME}")
    print(f"  Max Steps   : {MAX_STEPS:,}")
    print(f"  Seeds       : {SEEDS}")
    print("=" * 60)

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    all_variant_runs = {}   # label → list of seed dicts
    eps_history_sample = None  # κρατάμε ε history από ένα run για το plot

    for variant_label, mode in VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"Variant: {variant_label}  (mode={mode})")
        print(f"{'='*50}")

        seed_runs = []
        for seed in SEEDS:
            print(f"\n── Seed {seed} ──")
            run = train_dqn(seed=seed, mode=mode, max_steps=MAX_STEPS)
            seed_runs.append(run)

            # Αποθήκευση CSV
            csv_path = f"results/task2_{variant_label.lower().replace('-', '_')}_seed{seed}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["seed", "episode", "total_steps", "return"])
                for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                    writer.writerow([seed, ep + 1, s, r])

        all_variant_runs[variant_label] = seed_runs

        # Αποθήκευση checkpoint για DQN-full και DQN-double
        # Χρησιμοποιούμε τον ήδη εκπαιδευμένο agent του seed[0] — όχι νέο τυχαίο
        if mode == "full":
            seed_runs[0]["agent"].save("checkpoints/dqn_full.pt")
            eps_history_sample = seed_runs[0]["eps_history"]
        elif mode == "double":
            seed_runs[0]["agent"].save("checkpoints/dqn_double.pt")

    # ── Plot 1: Ablation curves ────────────────────────────────────────────────
    plot_learning_curves(
        runs      = all_variant_runs,
        title     = f"Task 2: DQN Ablation Study — {ENV_NAME}",
        save_path = "results/task2_dqn_ablation.png",
        colors    = VARIANT_COLORS,
        window    = 20,
    )

    # ── Plot 2: Epsilon decay ─────────────────────────────────────────────────
    # Παίρνουμε δείγμα κάθε 500 steps για να χωράει το plot
    if eps_history_sample:
        sample_every = 500
        eps_sampled  = eps_history_sample[::sample_every]
        steps_sampled = list(range(0, len(eps_history_sample), sample_every))

        from utils.plotting import plot_epsilon
        plot_epsilon(
            epsilon_history = eps_sampled,
            steps           = steps_sampled,
            save_path       = "results/task2_dqn_epsilon.png",
        )

    print("\n── Final Results ──")
    for label, seed_runs in all_variant_runs.items():
        final_returns = [np.mean(run["returns"][-20:]) for run in seed_runs
                         if len(run["returns"]) >= 20]
        if final_returns:
            print(f"  {label:20s}: {np.mean(final_returns):>8.1f} ± {np.std(final_returns):.1f}")
