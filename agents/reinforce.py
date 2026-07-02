"""
REINFORCE Agent (Monte Carlo Policy Gradient)
==============================================
Ο απλούστερος policy gradient αλγόριθμος (Williams, 1992).

Βασική ιδέα:
  Θέλουμε να μάθουμε ένα policy π_θ(a|s) που μεγιστοποιεί την
  αναμενόμενη συνολική αποζημίωση J(θ) = E[Σ γ^t · r_t].

  Ο Policy Gradient Theorem δίνει:
    ∇J(θ) = E[ Gₜ · ∇ log π_θ(aₜ|sₜ) ]
  όπου Gₜ = Σ_{k=0}^{T-t-1} γ^k · r_{t+k+1}  (discounted return)

  Το gradient update: θ ← θ + α · Gₜ · ∇ log π_θ(aₜ|sₜ)
  (γράφεται ως minimization: L = -E[Gₜ · log π_θ(aₜ|sₜ)])

Διαφορά από DQN:
  - On-policy: χρησιμοποιεί transitions από το ΤΡΕΧΟΝ policy
  - Monte Carlo: περιμένει το τέλος του episode για να υπολογίσει Gₜ
  - Άρα: υψηλό variance (βλέπουμε ολόκληρο το trajectory) αλλά unbiased
  - Χωρίς replay buffer (on-policy δεν μπορεί να χρησιμοποιήσει παλιά data)

Return Normalization:
  Αφαιρούμε τον μέσο και διαιρούμε με std → μειώνει το variance σημαντικά.
  Δεν αλλάζει την κατεύθυνση του gradient, απλώς σταθεροποιεί το scale.
"""

import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.networks import PolicyNetwork


class REINFORCEAgent:
    def __init__(
        self,
        state_dim  : int,
        n_actions  : int,
        hidden_dim : int   = 128,
        lr         : float = 1e-3,
        gamma      : float = 0.99,
        normalize_returns: bool = True,
        device     : torch.device = None,
    ):
        self.gamma            = gamma
        self.normalize_returns = normalize_returns
        self.device           = device or torch.device("cpu")

        self.policy    = PolicyNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Αποθηκεύουμε το trajectory του τρέχοντος episode
        # (rewards και log_probs για κάθε step)
        self._log_probs = []
        self._rewards   = []
        self.total_steps = 0

    def select_action(self, state: np.ndarray):
        """
        Sample action από το policy.
        Returns: action (int)
        Αποθηκεύει το log_prob για χρήση στο update.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Ένα μόνο forward pass μέσω του net — χωρίς να παρακάμπτουμε το forward()
        dist    = torch.distributions.Categorical(logits=self.policy.net(state_t))
        action  = dist.sample()

        self._log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        """Αποθηκεύει την αποζημίωση του τρέχοντος step."""
        self._rewards.append(reward)
        self.total_steps += 1

    def update(self) -> float:
        """
        Υπολογίζει τα discounted returns και κάνει ένα gradient update.
        Καλείται μία φορά στο τέλος κάθε episode.

        Returns: loss value (scalar)
        """
        T = len(self._rewards)

        # ── Υπολογισμός Gₜ (backward pass) ───────────────────────────────────
        # Ξεκινάμε από το τέλος και πάμε προς τα πίσω:
        #   G_T = 0
        #   G_t = r_{t+1} + γ · G_{t+1}
        # Αυτό είναι πολύ πιο αποδοτικό από O(T²) loop.
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)

        # ── Return Normalization ──────────────────────────────────────────────
        # Αφαιρούμε mean και διαιρούμε με std για σταθερό scale των gradients.
        # Το 1e-8 αποτρέπει division by zero σε short episodes.
        if self.normalize_returns and T > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # ── Policy Gradient Loss ──────────────────────────────────────────────
        # L = -Σₜ Gₜ · log π_θ(aₜ|sₜ)
        # Αρνητικό γιατί κάνουμε gradient descent (θέλουμε gradient ascent)
        log_probs = torch.stack(self._log_probs)   # [T]
        loss      = -(returns_t * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Reset trajectory για το επόμενο episode
        self._log_probs = []
        self._rewards   = []

        return loss.item()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
