"""
Advantage Actor-Critic (A2C) Agent
====================================
Συνδυάζει τα πλεονεκτήματα policy gradient και value-based methods.

Βασική ιδέα:
  - Actor: μαθαίνει policy π_θ(a|s) (τι να κάνω)
  - Critic: μαθαίνει value function V_w(s) (πόσο καλό είναι το state)
  - Advantage: Aₜ = rₜ + γ·V(s'ₜ) - V(sₜ)
    → πόσο ΚΑΛΥΤΕΡΗ ήταν η action από ό,τι αναμενόταν κατά μέσο όρο

  Γιατί advantage αντί για return;
    Το Gₜ έχει υψηλό variance γιατί εξαρτάται από ολόκληρο το μελλοντικό trajectory.
    Το Aₜ χρησιμοποιεί μόνο ΕΝΑ step lookahead → πολύ λιγότερο variance.
    Είναι biased (γιατί V_w είναι approximation), αλλά το tradeoff αξίζει.

Combined Loss:
  L = L_actor + c_v · L_critic - β · H(π)

  L_actor  = -E[Aₜ · log π_θ(aₜ|sₜ)]   ← policy gradient με advantage
  L_critic = E[(rₜ + γ·V(s') - V(s))²]  ← TD error για critic
  H(π)     = -Σ π log π                  ← entropy bonus για exploration

Entropy Bonus:
  Αποτρέπει το policy από το να γίνει too deterministic νωρίς.
  Μεγαλύτερο β → περισσότερη exploration → αργότερη αλλά πιο robust σύγκλιση.

ΚΡΙΣΙΜΟ: advantage.detach()
  Το advantage χρησιμοποιείται και στον actor loss και υπολογίζεται
  από τον critic. Χωρίς detach(), τα gradients του actor loss θα ρέουν
  ΠΙΣΩ μέσα από τον critic → λανθασμένα gradients παντού.
  Το detach() "κόβει" το computational graph.
"""

import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.networks import ActorCritic


class A2CAgent:
    def __init__(
        self,
        state_dim    : int,
        n_actions    : int,
        hidden_dim   : int   = 128,
        lr           : float = 3e-4,
        gamma        : float = 0.99,
        entropy_coef : float = 0.01,    # β: βάρος entropy bonus
        critic_coef  : float = 0.5,     # c_v: βάρος critic loss
        device       : torch.device = None,
    ):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.critic_coef  = critic_coef
        self.device       = device or torch.device("cpu")

        # Shared Actor-Critic network
        self.ac_net    = ActorCritic(state_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

        # Trajectory buffer — reset μετά από κάθε episode
        self._log_probs  = []
        self._values     = []
        self._rewards    = []
        self._entropies  = []
        self._dones      = []

        self.total_steps = 0

    def select_action(self, state: np.ndarray):
        """
        Sample action από τον actor.
        Αποθηκεύει log_prob, value, entropy για το update.
        Returns: action (int)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        dist, value = self.ac_net(state_t)
        action      = dist.sample()

        # Αποθηκεύουμε για το update
        self._log_probs.append(dist.log_prob(action))
        self._values.append(value.squeeze(0))       # scalar
        self._entropies.append(dist.entropy())       # H(π(·|s))

        return action.item()

    def store_reward(self, reward: float, done: bool):
        """Αποθηκεύει reward και done flag."""
        self._rewards.append(reward)
        self._dones.append(done)
        self.total_steps += 1

    def update(self, last_state: np.ndarray = None, last_done: bool = True) -> dict:
        """
        Υπολογίζει τα one-step TD advantages και κάνει gradient update.
        Καλείται στο τέλος κάθε episode (ή ανά N steps για n-step A2C).

        Args:
            last_state: το τελευταίο state (χρήσιμο αν το episode δεν τελείωσε)
            last_done : True αν το episode τελείωσε (→ bootstrap value = 0)

        Returns: dict με τα επιμέρους losses για logging
        """
        T = len(self._rewards)

        # ── Bootstrap Value ───────────────────────────────────────────────────
        # Αν το episode τελείωσε (done=True), V(s_T) = 0
        # Αν ΔΕΝ τελείωσε, χρησιμοποιούμε V(s_T) ως bootstrap estimate
        if last_done or last_state is None:
            next_value = 0.0
        else:
            with torch.no_grad():
                s_t  = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                _, v = self.ac_net(s_t)
                next_value = v.item()

        # ── One-step TD Advantages ────────────────────────────────────────────
        # Aₜ = rₜ + γ·V(s_{t+1}) - V(sₜ)
        # Υπολογίζουμε backwards για αποδοτικότητα
        advantages = []
        G          = next_value

        values_list   = [v.item() for v in self._values]

        for t in reversed(range(T)):
            # V(s_{t+1}): αν t είναι το τελευταίο step, χρησιμοποιούμε bootstrap
            if t == T - 1:
                v_next = next_value
            else:
                v_next = values_list[t + 1]

            # One-step TD advantage
            advantage_t = self._rewards[t] + self.gamma * v_next * (1 - float(self._dones[t])) - values_list[t]
            advantages.insert(0, advantage_t)

        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # ── Stack tensors ─────────────────────────────────────────────────────
        log_probs  = torch.stack(self._log_probs)    # [T]
        values     = torch.stack(self._values)        # [T]
        entropies  = torch.stack(self._entropies)     # [T]
        rewards_t  = torch.FloatTensor(self._rewards).to(self.device)

        # ── TD Targets για Critic ─────────────────────────────────────────────
        # y_t = r_t + γ·V(s_{t+1}) = advantage_t + V(s_t)
        targets = advantages_t + values.detach()

        # ── ΚΡΙΣΙΜΟ: detach() το advantage ───────────────────────────────────
        # Το advantage χρησιμοποιείται ΜΟΝ στον actor loss ΩΣ ΣΤΑΘΕΡΑ βάρος.
        # ΔΕΝ θέλουμε gradients να ρέουν μέσα από αυτό.
        adv_detached = advantages_t.detach()

        # ── Actor Loss ────────────────────────────────────────────────────────
        # L_actor = -E[Aₜ · log π(aₜ|sₜ)]
        actor_loss = -(adv_detached * log_probs).mean()

        # ── Critic Loss ───────────────────────────────────────────────────────
        # L_critic = E[(target - V(s))²]  (MSE)
        critic_loss = ((targets - values) ** 2).mean()

        # ── Entropy Bonus ─────────────────────────────────────────────────────
        # -β·H(π): αρνητικό γιατί θέλουμε να ΜΕΓΙΣΤΟΠΟΙΟΥΜΕ την entropy
        # (minimize αρνητικής entropy = maximize entropy)
        entropy_loss = -entropies.mean()

        # ── Combined Loss ─────────────────────────────────────────────────────
        # L = L_actor + c_v · L_critic - β · H
        total_loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Reset trajectory
        self._log_probs  = []
        self._values     = []
        self._rewards    = []
        self._entropies  = []
        self._dones      = []

        return {
            "total_loss"  : total_loss.item(),
            "actor_loss"  : actor_loss.item(),
            "critic_loss" : critic_loss.item(),
            "entropy"     : -entropy_loss.item(),  # positive entropy για logging
        }

    def save(self, path: str):
        torch.save(self.ac_net.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def load(self, path: str):
        self.ac_net.load_state_dict(torch.load(path, map_location=self.device))
