"""
Plotting Utilities
==================
Βοηθητικές συναρτήσεις για τα plots της εργασίας.

Όλα τα learning-curve plots ακολουθούν το ίδιο pattern:
  - X-axis: total environment steps (όχι episodes)
  - Y-axis: episodic return
  - Solid line: mean across seeds
  - Shaded region: ± 1 standard deviation

Χρησιμοποιούμε moving average για να εξομαλύνουμε τις καμπύλες,
γιατί τα raw episodic returns έχουν πολύ υψηλό noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # χωρίς display (server / headless)
import os


# ── Σταθερά styling ────────────────────────────────────────────────────────────
COLORS = {
    "tabular"    : "#E07B39",   # πορτοκαλί
    "dqn_full"   : "#2196F3",   # μπλε
    "dqn_noreplay": "#F44336",  # κόκκινο
    "dqn_notarget": "#9C27B0",  # μωβ
    "reinforce"  : "#4CAF50",   # πράσινο
    "a2c"        : "#FF9800",   # amber
}


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Moving average για εξομάλυνση καμπύλης.
    Χρησιμοποιεί 'same' mode ώστε το output να έχει το ίδιο μήκος με το input.
    """
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def align_to_steps(returns_per_ep: list, steps_per_ep: list) -> tuple:
    """
    Μετατρέπει τα per-episode returns σε per-step άξονα.
    Επιστρέφει (cumulative_steps, returns) arrays.
    """
    cum_steps = np.cumsum(steps_per_ep)
    return cum_steps, np.array(returns_per_ep)


def plot_learning_curves(
    runs: dict,           # {"label": {"steps": [...], "returns": [...]}} για κάθε seed
    title: str,
    xlabel: str = "Environment Steps",
    ylabel: str = "Episode Return",
    save_path: str = None,
    colors: dict = None,
    window: int = 20,
):
    """
    Κύρια συνάρτηση για learning curve plots.

    Args:
        runs      : dict όπου κάθε key είναι label, value είναι list of dicts
                    (ένα dict ανά seed), κάθε dict έχει 'steps' και 'returns'
        title     : τίτλος plot
        save_path : αν δοθεί, αποθηκεύει το plot ως PNG
        window    : παράθυρο moving average
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, seed_runs in runs.items():
        color = (colors or COLORS).get(label, None)

        # Interpolate σε κοινό step grid για να μπορούμε να υπολογίσουμε mean/std
        # Βρίσκουμε το μέγιστο κοινό step range
        max_step = min(run["steps"][-1] for run in seed_runs if len(run["steps"]) > 0)
        grid     = np.linspace(0, max_step, 500)

        interpolated = []
        for run in seed_runs:
            steps   = np.array(run["steps"])
            returns = np.array(run["returns"])
            # Linear interpolation στο κοινό grid
            interp = np.interp(grid, steps, returns)
            interpolated.append(smooth(interp, window))

        interpolated = np.array(interpolated)   # [n_seeds, 500]
        mean = interpolated.mean(axis=0)
        std  = interpolated.std(axis=0)

        ax.plot(grid, mean, label=label, color=color, linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_epsilon(epsilon_history: list, steps: list, save_path: str = None):
    """Plot της τιμής ε κατά τη διάρκεια του training (Task 2)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, epsilon_history, color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Epsilon (ε)", fontsize=12)
    ax.set_title("DQN ε-greedy Decay", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_hyperparam_study(
    results: dict,   # {"value_label": list_of_seed_runs}
    param_name: str,
    algorithm: str,
    env_name: str,
    save_path: str = None,
    window: int = 20,
):
    """
    Plot για το hyperparameter study (Task 4b).
    Κάθε καμπύλη αντιστοιχεί σε μία τιμή του hyperparameter.
    """
    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(results) - 1, 1)) for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for (label, seed_runs), color in zip(results.items(), colors):
        max_step = min(run["steps"][-1] for run in seed_runs if run["steps"])
        grid     = np.linspace(0, max_step, 500)

        interpolated = []
        for run in seed_runs:
            steps   = np.array(run["steps"])
            returns = np.array(run["returns"])
            interp  = np.interp(grid, steps, returns)
            interpolated.append(smooth(interp, window))

        interpolated = np.array(interpolated)
        mean = interpolated.mean(axis=0)
        std  = interpolated.std(axis=0)

        ax.plot(grid, mean, label=f"{param_name}={label}", color=color, linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title(
        f"{algorithm} Hyperparameter Study: {param_name}\n({env_name})",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()
