"""
Task 4 — Algorithm Comparison & Hyperparameter Study
======================================================

4a: Σύγκριση όλων των αλγορίθμων (Tabular, DQN, REINFORCE, A2C) σε LunarLander.
    Τρέχει εκ νέου όλους τους αλγορίθμους και τους βάζει σε ένα κοινό plot.
    ΣΗΜΕΙΩΣΗ: Αν έχεις ήδη τρέξει τα task1-3 scripts, μπορείς να φορτώσεις
    τα αποθηκευμένα CSV αντί να ξανατρέξεις (εξοικονόμηση χρόνου).

4b: Hyperparameter study σε CartPole-v1.
    Δύο studies:
      A) A2C: entropy coefficient β ∈ {0.0, 0.001, 0.01, 0.05}
      B) REINFORCE: discount factor γ ∈ {0.90, 0.95, 0.99, 1.00}
    3 seeds ανά configuration.

Τρέξιμο:
  # Μόνο task 4a (χρειάζεται τα task1-3 αποτελέσματα):
  python experiments/task4_comparison.py --part 4a

  # Μόνο task 4b:
  python experiments/task4_comparison.py --part 4b

  # Και τα δύο:
  python experiments/task4_comparison.py
"""

import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import time

from agents import DQNAgent, REINFORCEAgent, A2CAgent
from utils  import set_seed, get_device, plot_learning_curves
from utils.plotting import plot_hyperparam_study

# Imports από τα άλλα task scripts για να μην γράφουμε τον ίδιο κώδικα δύο φορές
from task1_tabular          import train_tabular,   N_BINS, ENV_NAME as TABULAR_ENV
from task3_policy_gradient  import (
    train_reinforce, train_a2c,
    REINFORCE_CONFIG, A2C_CONFIG,
)

# ── Shared Hyperparameters ────────────────────────────────────────────────────
LUNAR_ENV  = "LunarLander-v3"
CARTPOLE   = "CartPole-v1"
MAX_STEPS  = 500_000
SEEDS      = [42, 123, 777]

DQN_CONFIG_LUNAR = dict(
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

# ── Hyperparameter Study Config ───────────────────────────────────────────────
BETA_VALUES  = [0.0, 0.001, 0.01, 0.05]   # entropy coefficient για A2C
GAMMA_VALUES = [0.90, 0.95, 0.99, 1.00]   # discount factor για REINFORCE

# CartPole: λιγότερα steps (σύγκλίνει πολύ πιο γρήγορα)
CARTPOLE_MAX_STEPS = 150_000


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_runs_from_csv(csv_pattern: str, seeds: list) -> list:
    """
    Φορτώνει αποτελέσματα από CSV αρχεία (αν υπάρχουν από προηγούμενα runs).
    Επιστρέφει list of dicts {"steps": [...], "returns": [...]}
    """
    runs = []
    for seed in seeds:
        path = csv_pattern.format(seed=seed)
        if not os.path.exists(path):
            return None   # δεν υπάρχουν αρχεία → πρέπει να ξανατρέξουμε
        steps, returns = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["total_steps"]))
                returns.append(float(row["return"]))
        runs.append({"steps": steps, "returns": returns})
    return runs


def run_dqn_lunar(seeds: list) -> list:
    """Τρέχει DQN full στο LunarLander για όλα τα seeds."""
    runs = []
    for seed in seeds:
        print(f"  [DQN | Seed {seed}]")
        set_seed(seed)
        device = get_device()
        env    = gym.make(LUNAR_ENV)
        agent  = DQNAgent(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            mode="full", device=device, **DQN_CONFIG_LUNAR
        )
        returns_log, steps_log = [], []
        state, _ = env.reset(seed=seed)
        ep_return, episode = 0.0, 0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_s, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.step(state, action, reward, next_s, done)
            state = next_s; ep_return += reward
            if done:
                episode += 1
                returns_log.append(ep_return)
                steps_log.append(agent.total_steps)
                ep_return = 0.0
                state, _ = env.reset(seed=seed + episode)
        env.close()
        runs.append({"steps": steps_log, "returns": returns_log})
    return runs


# ═════════════════════════════════════════════════════════════════════════════
# Task 4a: Algorithm Comparison
# ═════════════════════════════════════════════════════════════════════════════
def task4a():
    """
    Συγκρίνει Tabular, DQN, REINFORCE, A2C σε LunarLander (ή CartPole για Tabular).
    Δοκιμάζει να φορτώσει από CSV πρώτα για να αποφύγει re-training.
    """
    print("\n" + "="*60)
    print("Task 4a: Algorithm Comparison")
    print(f"  Main env: {LUNAR_ENV} | Tabular env: {TABULAR_ENV}")
    print("="*60)

    all_runs = {}

    # ── Tabular (CartPole) ─────────────────────────────────────────────────────
    print("\n[Tabular Q-Learning]")
    tabular_csv = "results/task1_tabular_qlearning.csv"
    tabular_runs = load_runs_from_csv(
        "results/task1_tabular_qlearning.csv", SEEDS
    )
    if tabular_runs:
        print("  Loaded from CSV (skipping re-training)")
    else:
        tabular_runs = [train_tabular(s) for s in SEEDS]
    all_runs["Tabular Q-Learning"] = tabular_runs

    # ── DQN (LunarLander) ─────────────────────────────────────────────────────
    print("\n[DQN-full]")
    dqn_pattern = "results/task2_dqn_full_seed{seed}.csv"
    dqn_runs = load_runs_from_csv(dqn_pattern, SEEDS)
    if dqn_runs:
        print("  Loaded from CSV (skipping re-training)")
    else:
        print("  Training DQN from scratch...")
        dqn_runs = run_dqn_lunar(SEEDS)
    all_runs["DQN"] = dqn_runs

    # ── REINFORCE (LunarLander) ────────────────────────────────────────────────
    print("\n[REINFORCE]")
    reinforce_runs = []
    loaded = True
    for seed in SEEDS:
        path = f"results/task3_reinforce_seed{seed}.csv"
        if not os.path.exists(path):
            loaded = False; break
    if loaded:
        reinforce_runs = load_runs_from_csv("results/task3_reinforce_seed{seed}.csv", SEEDS)
        print("  Loaded from CSV (skipping re-training)")
    else:
        print("  Training REINFORCE from scratch...")
        for seed in SEEDS:
            reinforce_runs.append(train_reinforce(seed, MAX_STEPS))
    all_runs["REINFORCE"] = reinforce_runs

    # ── A2C (LunarLander) ─────────────────────────────────────────────────────
    print("\n[A2C]")
    a2c_runs = []
    loaded = True
    for seed in SEEDS:
        path = f"results/task3_a2c_seed{seed}.csv"
        if not os.path.exists(path):
            loaded = False; break
    if loaded:
        a2c_runs = load_runs_from_csv("results/task3_a2c_seed{seed}.csv", SEEDS)
        print("  Loaded from CSV (skipping re-training)")
    else:
        print("  Training A2C from scratch...")
        for seed in SEEDS:
            a2c_runs.append(train_a2c(seed, MAX_STEPS))
    all_runs["A2C"] = a2c_runs

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_learning_curves(
        runs  = all_runs,
        title = f"Task 4a: Algorithm Comparison\n({LUNAR_ENV}, Tabular on {TABULAR_ENV})",
        save_path = "results/task4_comparison.png",
        colors = {
            "Tabular Q-Learning": "#E07B39",
            "DQN"               : "#2196F3",
            "REINFORCE"         : "#4CAF50",
            "A2C"               : "#FF9800",
        },
        window = 20,
    )

    # ── Report Table ───────────────────────────────────────────────────────────
    print("\n── Report Table (fill in report.md) ──")
    print(f"{'Algorithm':<22} {'Final Return':>14} {'Std':>8}")
    print("-" * 46)
    for label, runs in all_runs.items():
        finals = [np.mean(r["returns"][-20:]) for r in runs if len(r["returns"]) >= 20]
        if finals:
            print(f"  {label:<20} {np.mean(finals):>12.1f} {np.std(finals):>8.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# Task 4b: Hyperparameter Study on CartPole
# ═════════════════════════════════════════════════════════════════════════════

def run_a2c_cartpole(seed: int, beta: float) -> dict:
    """Τρέχει A2C στο CartPole με συγκεκριμένο β (entropy coefficient)."""
    set_seed(seed)
    device = get_device()
    env    = gym.make(CARTPOLE)

    agent = A2CAgent(
        state_dim    = env.observation_space.shape[0],  # 4
        n_actions    = env.action_space.n,              # 2
        device       = device,
        hidden_dim   = 64,    # μικρότερο για CartPole
        lr           = 3e-4,
        gamma        = 0.99,
        entropy_coef = beta,
        critic_coef  = 0.5,
    )

    returns_log, steps_log = [], []
    episode = 0

    while agent.total_steps < CARTPOLE_MAX_STEPS:
        state, _ = env.reset(seed=seed + episode)
        ep_return = 0.0

        while True:
            action = agent.select_action(state)
            next_s, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store_reward(reward, done)
            state = next_s; ep_return += reward
            if done: break

        agent.update(last_state=None, last_done=True)
        episode += 1
        returns_log.append(ep_return)
        steps_log.append(agent.total_steps)

    env.close()
    return {"steps": steps_log, "returns": returns_log}


def run_reinforce_cartpole(seed: int, gamma: float) -> dict:
    """Τρέχει REINFORCE στο CartPole με συγκεκριμένο γ (discount factor)."""
    set_seed(seed)
    device = get_device()
    env    = gym.make(CARTPOLE)

    agent = REINFORCEAgent(
        state_dim        = env.observation_space.shape[0],
        n_actions        = env.action_space.n,
        device           = device,
        hidden_dim       = 64,
        lr               = 1e-3,
        gamma            = gamma,
        normalize_returns= True,
    )

    returns_log, steps_log = [], []
    episode = 0

    while agent.total_steps < CARTPOLE_MAX_STEPS:
        state, _ = env.reset(seed=seed + episode)
        ep_return = 0.0

        while True:
            action = agent.select_action(state)
            next_s, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store_reward(reward)
            state = next_s; ep_return += reward
            if done: break

        agent.update()
        episode += 1
        returns_log.append(ep_return)
        steps_log.append(agent.total_steps)

    env.close()
    return {"steps": steps_log, "returns": returns_log}


def task4b():
    """
    Hyperparameter study σε CartPole.
    Study A: A2C με διαφορετικά β (entropy coefficient)
    Study B: REINFORCE με διαφορετικά γ (discount factor)
    """
    print("\n" + "="*60)
    print("Task 4b: Hyperparameter Study")
    print(f"  Environment : {CARTPOLE}")
    print(f"  Max Steps   : {CARTPOLE_MAX_STEPS:,}")
    print(f"  Seeds       : {SEEDS}")
    print("="*60)

    os.makedirs("results", exist_ok=True)

    # ── Study A: A2C entropy coefficient β ────────────────────────────────────
    print("\n── Study A: A2C Entropy Coefficient (β) ──")
    a2c_hyperparam_results = {}

    for beta in BETA_VALUES:
        label = str(beta)
        print(f"\n  β = {beta}")
        seed_runs = []
        for seed in SEEDS:
            t0  = time.time()
            run = run_a2c_cartpole(seed, beta)
            seed_runs.append(run)
            final = np.mean(run["returns"][-50:]) if len(run["returns"]) >= 50 else np.mean(run["returns"])
            print(f"    Seed {seed}: final return {final:.1f}  ({time.time()-t0:.0f}s)")

            # CSV log
            with open(f"results/task4b_a2c_beta{beta}_seed{seed}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["seed", "episode", "total_steps", "return"])
                for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                    writer.writerow([seed, ep + 1, s, r])

        a2c_hyperparam_results[label] = seed_runs

    plot_hyperparam_study(
        results    = a2c_hyperparam_results,
        param_name = "β (entropy coef)",
        algorithm  = "A2C",
        env_name   = CARTPOLE,
        save_path  = "results/task4_hyperparam_study_a2c_beta.png",
        window     = 30,
    )

    # ── Study B: REINFORCE discount factor γ ──────────────────────────────────
    print("\n── Study B: REINFORCE Discount Factor (γ) ──")
    reinforce_hyperparam_results = {}

    for gamma in GAMMA_VALUES:
        label = str(gamma)
        print(f"\n  γ = {gamma}")
        seed_runs = []
        for seed in SEEDS:
            t0  = time.time()
            run = run_reinforce_cartpole(seed, gamma)
            seed_runs.append(run)
            final = np.mean(run["returns"][-50:]) if len(run["returns"]) >= 50 else np.mean(run["returns"])
            print(f"    Seed {seed}: final return {final:.1f}  ({time.time()-t0:.0f}s)")

            with open(f"results/task4b_reinforce_gamma{gamma}_seed{seed}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["seed", "episode", "total_steps", "return"])
                for ep, (s, r) in enumerate(zip(run["steps"], run["returns"])):
                    writer.writerow([seed, ep + 1, s, r])

        reinforce_hyperparam_results[label] = seed_runs

    plot_hyperparam_study(
        results    = reinforce_hyperparam_results,
        param_name = "γ (discount)",
        algorithm  = "REINFORCE",
        env_name   = CARTPOLE,
        save_path  = "results/task4_hyperparam_study_reinforce_gamma.png",
        window     = 30,
    )

    # Το assignment ζητά ΕΝΑ plot (task4_hyperparam_study.png)
    # Επιλέγουμε το A2C β study ως το κύριο (πιο ενδιαφέρον)
    import shutil
    shutil.copy(
        "results/task4_hyperparam_study_a2c_beta.png",
        "results/task4_hyperparam_study.png"
    )
    print("\n  task4_hyperparam_study.png → copy of A2C β study (required filename)")

    # ── Σύνοψη ─────────────────────────────────────────────────────────────────
    print("\n── Best β for A2C ──")
    best_beta = max(
        a2c_hyperparam_results.keys(),
        key=lambda b: np.mean([np.mean(r["returns"][-50:]) for r in a2c_hyperparam_results[b]
                                if len(r["returns"]) >= 50])
    )
    print(f"  Best β = {best_beta}")

    print("\n── Best γ for REINFORCE ──")
    best_gamma = max(
        reinforce_hyperparam_results.keys(),
        key=lambda g: np.mean([np.mean(r["returns"][-50:]) for r in reinforce_hyperparam_results[g]
                                if len(r["returns"]) >= 50])
    )
    print(f"  Best γ = {best_gamma}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["4a", "4b", "all"], default="all",
                        help="Ποιο μέρος να τρέξει (default: all)")
    args = parser.parse_args()

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.part in ("4a", "all"):
        task4a()

    if args.part in ("4b", "all"):
        task4b()
