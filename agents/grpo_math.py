"""
GRPO Loss (Group Relative Policy Optimization)
================================================
Υλοποίηση του GRPO loss όπως περιγράφεται στο DeepSeekMath (Shao et al., 2024).

Βασική ιδέα:
  Για κάθε prompt παράγουμε G completions (μια "ομάδα").
  Αντί για critic model (όπως στο PPO), χρησιμοποιούμε τα rewards ΜΕΣΑ
  στην ομάδα για να υπολογίσουμε baseline: standardize τα rewards → advantages.

Loss:
  L = -E[A_i · Σ_t log π_θ(a_t|a_{<t})] + β · KL(π_θ || π_ref)

KL Divergence (unbiased estimator, Schulman 2020):
  KL(π_ref || π_θ) ≈ exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
"""

import torch
import torch.nn.functional as F

def compute_grpo_loss(
    policy_logits : torch.Tensor,   # [G, seq_len, vocab] — Ήδη sliced για τα completion tokens
    ref_logits    : torch.Tensor,   # [G, seq_len, vocab] — Ήδη sliced
    completion_ids: torch.Tensor,   # [G, seq_len] — token IDs του completion
    rewards       : list,           # [G] Python list of floats
    beta          : float = 0.01,
    epsilon       : float = 1e-8,
):
    device = policy_logits.device

    # ── Group-Relative Advantages ─────────────────────────────────────────────
    r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r   = r_tensor.mean()
    std_r    = r_tensor.std(unbiased=False)

    if std_r < epsilon:
        advantages = torch.zeros_like(r_tensor)
    else:
        advantages = (r_tensor - mean_r) / (std_r + epsilon)   # [G]

    # ── Per-token Log Probabilities ───────────────────────────────────────────
    # Εφόσον η αποκοπή (slicing) έγινε ορθά στο main script, τα policy_logits 
    # έχουν πλέον απόλυτη αντιστοιχία 1-προς-1 (bijection) με τα completion_ids.
    log_probs_policy = F.log_softmax(policy_logits, dim=-1)   # [G, L, V]
    log_probs_ref    = F.log_softmax(ref_logits, dim=-1)      # [G, L, V]

    targets = completion_ids.unsqueeze(-1)   # [G, L, 1]

    # gather: επιλέγει το log_prob του πραγματικού next token
    token_logp_policy = log_probs_policy.gather(dim=-1, index=targets).squeeze(-1)  # [G, L]
    token_logp_ref    = log_probs_ref.gather(dim=-1, index=targets).squeeze(-1)     # [G, L]

    # ── Sequence Log Probabilities ────────────────────────────────────────────
    seq_logp_policy = token_logp_policy.sum(dim=-1)   # [G]
    seq_logp_ref    = token_logp_ref.sum(dim=-1)       # [G]

    # ── KL Divergence (unbiased estimator) ───────────────────────────────────
    log_ratio = seq_logp_ref - seq_logp_policy          # [G]
    kl_div    = torch.exp(log_ratio) - log_ratio - 1.0  # [G], always ≥ 0

    # ── GRPO Policy Gradient Loss ─────────────────────────────────────────────
    surrogate_loss = -(seq_logp_policy * advantages).mean()

    # ── Combined Loss ─────────────────────────────────────────────────────────
    kl_loss    = beta * kl_div.mean()
    total_loss = surrogate_loss + kl_loss

    return total_loss, kl_div.mean().item(), mean_r.item()