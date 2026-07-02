"""
Continuous A2C Agent (Bonus C)
================================
Επέκταση του A2C για συνεχή action spaces.
Περιβάλλον στόχος: LunarLanderContinuous-v2

Η κύρια διαφορά από τον discrete A2CAgent:

  Discrete:   actor βγάζει logits → Categorical(logits) → sample integer action
  Continuous: actor βγάζει (mean, log_std) → Normal(mean, std) → sample float action

Γιατί Gaussian policy;
  Σε continuous spaces δεν μπορούμε να απαριθμήσουμε όλες τις actions.
  Αντίθετα, παραμετροποιούμε μια κατανομή πάνω στον action space.
  Η Gaussian είναι η πιο κοινή επιλογή: απλή, differentiable, και
  η entropy της έχει closed-form (δεν χρειάζεται Monte Carlo εκτίμηση).

Log probability για Gaussian:
  log π(a|s) = Σ_i -0.5 · [(aᵢ - μᵢ)²/σᵢ² + log(2π) + 2·log(σᵢ)]
  Το PyTorch Normal distribution το υπολογίζει αυτόματα.

Action clipping:
  Το LunarLanderContinuous περιμένει actions στο [-1, 1].
  Το tanh στο mean το εξασφαλίζει κατά προσέγγιση, αλλά κάνουμε
  και hard clip μετά το sampling για ασφάλεια.
"""

import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.networks import ContinuousActorCritic


class ContinuousA2CAgent:
    def __init__(
        self,
        state_dim    : int,
        action_dim   : int,
        hidden_dim   : int   = 256,
        lr           : float = 3e-4,
        gamma        : float = 0.99,
        entropy_coef : float = 0.001,  # μικρότερο β για continuous (variance αρκεί για exploration)
        critic_coef  : float = 0.5,
        device       : torch.device = None,
    ):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.critic_coef  = critic_coef
        self.action_dim   = action_dim
        self.device       = device or torch.device("cpu")

        self.ac_net    = ContinuousActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

        # Trajectory buffers
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._entropies = []
        self._dones     = []

        # Pending transition (set by select_action, committed by store_reward)
        self._pending_log_prob = None
        self._pending_value    = None
        self._pending_entropy  = None

        self.total_steps = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Sample continuous action από τη Gaussian policy.
        Returns: action array (clipped στο [-1, 1])

        Σημείωση: log_prob, value και entropy αποθηκεύονται προσωρινά
        και επιβεβαιώνονται (commit) μόνο όταν ακολουθήσει store_reward.
        Έτσι αποφεύγουμε size mismatch αν κληθεί select_action εκτός episode.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, value = self.ac_net(state_t)

        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Αποθήκευση pending — commit γίνεται στο store_reward
        self._pending_log_prob = log_prob.squeeze(0)
        self._pending_value    = value.squeeze(0)
        self._pending_entropy  = dist.entropy().sum(dim=-1).squeeze(0)

        action_np = action.squeeze(0).detach().cpu().numpy()
        return np.clip(action_np, -1.0, 1.0)

    def store_reward(self, reward: float, done: bool):
        """Επιβεβαιώνει το pending transition και αποθηκεύει reward/done."""
        # Commit: μόνο αν υπάρχει pending (δηλ. select_action κλήθηκε πριν)
        if self._pending_log_prob is not None:
            self._log_probs.append(self._pending_log_prob)
            self._values.append(self._pending_value)
            self._entropies.append(self._pending_entropy)
            self._pending_log_prob = None
            self._pending_value    = None
            self._pending_entropy  = None

        self._rewards.append(reward)
        self._dones.append(done)
        self.total_steps += 1

    def update(self, last_state: np.ndarray = None, last_done: bool = True) -> dict:
        """
        Ίδια λογική με τον discrete A2CAgent:
        One-step TD advantage + actor/critic/entropy losses.
        """
        T = len(self._rewards)

        # Bootstrap value
        if last_done or last_state is None:
            next_value = 0.0
        else:
            with torch.no_grad():
                s_t  = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                _, v = self.ac_net(s_t)
                next_value = v.item()

        # One-step TD advantages (backward)
        advantages  = []
        values_list = [v.item() for v in self._values]

        for t in reversed(range(T)):
            v_next      = next_value if t == T - 1 else values_list[t + 1]
            advantage_t = (self._rewards[t]
                           + self.gamma * v_next * (1 - float(self._dones[t]))
                           - values_list[t])
            advantages.insert(0, advantage_t)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        log_probs    = torch.stack(self._log_probs)
        values       = torch.stack(self._values)
        entropies    = torch.stack(self._entropies)

        targets      = advantages_t + values.detach()
        adv_detached = advantages_t.detach()

        actor_loss   = -(adv_detached * log_probs).mean()
        critic_loss  = ((targets - values) ** 2).mean()
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Reset
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._entropies = []
        self._dones     = []
        self._pending_log_prob = None
        self._pending_value    = None
        self._pending_entropy  = None

        return {
            "total_loss" : total_loss.item(),
            "actor_loss" : actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy"    : -entropy_loss.item(),
        }

    def save(self, path: str):
        torch.save(self.ac_net.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def load(self, path: str):
        self.ac_net.load_state_dict(torch.load(path, map_location=self.device))
