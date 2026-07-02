"""
Task 3 — REINFORCE & A2C
==========================
Κύριο περιβάλλον: LunarLander-v2

Σύγκριση δύο policy gradient αλγορίθμων:

REINFORCE (3a):
  Monte Carlo: μαθαίνει από complete episodes.
  Πρόβλημα: πολύ υψηλό variance → αργή ή καθόλου σύγκλιση στο LunarLander.
  Αν η καμπύλη είναι flat μετά από 300k steps, αυτό είναι EXPECTED.

A2C (3b):
  One-step TD advantage: πολύ λιγότερο variance από REINFORCE.
  Entropy bonus αποτρέπει premature determinism.
  Γενικά πιο γρήγορη σύγκλιση από REINFORCE.

Τρέξιμο:
  python experiments/task3_policy_gradient.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import gymnasium as gym

from agents import REINFORCEAgent, A2CAgent
from utils  import set_seed, get_device, plot_learning_curves

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENV_NAME  = "LunarLander-v3"
MAX_STEPS = 500_000
SEEDS     = [42, 123, 777]

REINFORCE_CONFIG = dict(
    hidden_dim        = 128,
    lr                = 1e-4,  # Μείωση από 3e-4 σε 1e-4 για απόσβεση των ταλαντώσεων (variance reduction)
    gamma             = 0.99,
    normalize_returns = True,
)

A2C_CONFIG = dict(
    hidden_dim    = 128,
    lr            = 3e-4,  
    gamma         = 0.99,
    entropy_coef  = 0.02,  # Αύξηση από 0.01 σε 0.02 για βίαιη επιβολή στοχαστικότητας (exploration)
    critic_coef   = 0.5,
)


# ── REINFORCE Training ─────────────────────────────────────────────────────────
def train_reinforce(seed: int, max_steps: int) -> dict:
    """
    Εκπαιδεύει REINFORCE agent.
    Ο agent εκπαιδεύεται ΜΙΑ φορά ανά episode (Monte Carlo).
    """
    set_seed(seed)
    device = get_device()
    env    = gym.make(ENV_NAME)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
        **REINFORCE_CONFIG,
    )

    returns_log = []
    steps_log   = []
    episode     = 0

    while agent.total_steps < max_steps:
        state, _ = env.reset(seed=seed + episode)
        ep_return = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            state     = next_state
            ep_return += reward

            if done:
                break

        # Update μία φορά στο τέλος του episode (Monte Carlo)
        loss = agent.update()

        episode += 1
        returns_log.append(ep_return)
        steps_log.append(agent.total_steps)

        if episode % 50 == 0:
            recent = np.mean(returns_log[-20:]) if len(returns_log) >= 20 else np.mean(returns_log)
            print(f"  [REINFORCE | Seed {seed}] Ep {episode} | "
                  f"Steps {agent.total_steps:>6} | Avg-20: {recent:>8.1f}")

    env.close()
    return {"steps": steps_log, "returns": returns_log, "agent": agent}


# ── A2C Training ───────────────────────────────────────────────────────────────
def train_a2c(seed: int, max_steps: int, entropy_coef: float = None) -> dict:
    """
    Εκπαιδεύει A2C agent.
    Update γίνεται στο τέλος κάθε episode.
    Η παράμετρος entropy_coef override-άρει το default (για το hyperparameter study).
    """
    set_seed(seed)
    device = get_device()
    env    = gym.make(ENV_NAME)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    config = A2C_CONFIG.copy()
    if entropy_coef is not None:
        config["entropy_coef"] = entropy_coef

    agent = A2CAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
        **config,
    )

    returns_log = []
    steps_log   = []
    episode     = 0

    while agent.total_steps < max_steps:
        state, _ = env.reset(seed=seed + episode)
        ep_return = 0.0
        last_done = False

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward, done)
            state     = next_state
            ep_return += reward
            last_done = done

            if done:
                break

        # Update στο τέλος του episode
        # last_state=None γιατί done=True → bootstrap value = 0
        losses = agent.update(last_state=None, last_done=last_done)

        episode += 1
        returns_log.append(ep_return)
        steps_log.append(agent.total_steps)

        if episode % 50 == 0:
            recent = np.mean(returns_log[-20:]) if len(returns_log) >= 20 else np.mean(returns_log)
            print(f"  [A2C | Seed {seed}] Ep {episode} | "
                  f"Steps {agent.total_steps:>6} | Avg-20: {recent:>8.1f} | "
                  f"Entropy: {losses['entropy']:.3f}")

    env.close()
    return {"steps": steps_log, "returns": returns_log, "agent": agent}


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Task 3: Policy Gradient Methods")
    print(f"  Environment : {ENV_NAME}")
    print(f"  Max Steps   : {MAX_STEPS:,}")
    print(f"  Seeds       : {SEEDS}")
    print("=" * 60)

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ── 3a: REINFORCE ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("3a: REINFORCE")
    print("="*50)

    reinforce_runs = []
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        run = train_reinforce(seed=seed, max_steps=MAX_STEPS)
        reinforce_runs.append(run)

        csv_path = f"results/task3_reinforce_seed{seed}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "episode", "total_steps", "return"])
            for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                writer.writerow([seed, ep + 1, s, r])

    plot_learning_curves(
        runs      = {"REINFORCE": reinforce_runs},
        title     = f"Task 3a: REINFORCE — {ENV_NAME}",
        save_path = "results/task3_reinforce.png",
        colors    = {"REINFORCE": "#4CAF50"},
        window    = 20,
    )

    # Checkpoint: αποθηκεύουμε τον ήδη εκπαιδευμένο agent του seed[0]
    reinforce_runs[0]["agent"].save("checkpoints/reinforce.pt")

    # ── 3b: A2C ────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("3b: A2C")
    print("="*50)

    a2c_runs = []
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        run = train_a2c(seed=seed, max_steps=MAX_STEPS)
        a2c_runs.append(run)

        csv_path = f"results/task3_a2c_seed{seed}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "episode", "total_steps", "return"])
            for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                writer.writerow([seed, ep + 1, s, r])

    plot_learning_curves(
        runs      = {"A2C": a2c_runs},
        title     = f"Task 3b: A2C — {ENV_NAME}",
        save_path = "results/task3_a2c.png",
        colors    = {"A2C": "#FF9800"},
        window    = 20,
    )

    # Checkpoint: αποθηκεύουμε τον ήδη εκπαιδευμένο agent του seed[0]
    a2c_runs[0]["agent"].save("checkpoints/a2c.pt")

    # ── Σύνοψη ─────────────────────────────────────────────────────────────────
    print("\n── Final Results ──")
    for label, runs in [("REINFORCE", reinforce_runs), ("A2C", a2c_runs)]:
        final = [np.mean(r["returns"][-20:]) for r in runs if len(r["returns"]) >= 20]
        if final:
            print(f"  {label:12s}: {np.mean(final):>8.1f} ± {np.std(final):.1f}")