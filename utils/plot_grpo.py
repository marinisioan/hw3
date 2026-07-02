"""
Plot αποτελεσμάτων GRPO training (Bonus A).
Τρέξε μετά το task_bonus_a.py για να παραχθεί το PNG.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_grpo_results(csv_path: str = r"C:\Users\ioannis marinis\Documents\Μαθηματική Προτυποποίηση σε Σύγχρονες Τεχνολογίες και τη Χρηματοοικονομική\2nd semester\A.I. Hands On\Assignment 3\hw3\experiments\results\task_bonus_a_log.csv",
                      save_path: str = r"C:\Users\ioannis marinis\Documents\Μαθηματική Προτυποποίηση σε Σύγχρονες Τεχνολογίες και τη Χρηματοοικονομική\2nd semester\A.I. Hands On\Assignment 3\hw3\experiments\results\bonus_a_grpo_training.png"):
    steps, losses, kl_divs, rewards = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            kl_divs.append(float(row["kl_div"]))
            rewards.append(float(row["mean_reward"]))

    def smooth(vals, w=10):
        return np.convolve(vals, np.ones(w)/w, mode="same")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Bonus A: GRPO Fine-tuning — Qwen2.5-0.5B on GSM8K",
                 fontsize=13, fontweight="bold")

    axes[0].plot(steps, rewards, alpha=0.3, color="#4CAF50")
    axes[0].plot(steps, smooth(rewards), color="#4CAF50", linewidth=2)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Mean Reward (group)")
    axes[0].set_title("Group Mean Reward"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, kl_divs, color="#F44336", linewidth=1.5)
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="KL=1.0 warning")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL(π_ref || π_θ)"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, losses, alpha=0.3, color="#2196F3")
    axes[2].plot(steps, smooth(losses), color="#2196F3", linewidth=2)
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("GRPO Loss")
    axes[2].set_title("Total Loss (PG + β·KL)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_grpo_results()
