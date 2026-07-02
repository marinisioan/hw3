"""
Deep Q-Network (DQN) Agent
===========================
Υλοποίηση του DQN αλγορίθμου (Mnih et al., 2015) με:
  1. Experience Replay Buffer
  2. Target Network (hard update κάθε C steps)
  3. ε-greedy exploration με εκθετική αποσύνθεση

Βασική ιδέα:
  Θέλουμε να μάθουμε Q*(s,a) = αναμενόμενη συνολική αποζημίωση αν
  επιλέξουμε action a στο state s και μετά ακολουθήσουμε optimal policy.
  Χρησιμοποιούμε neural network Q_θ(s,a) που εκπαιδεύεται να κάνει:
    Q_θ(s,a) ≈ r + γ · max_a' Q_θ-(s',a')
  όπου θ- είναι τα frozen weights του target network.

Ο agent υποστηρίζει τέσσερα modes:
  - "full"      : replay buffer + target network (κανονικό DQN)
  - "no_replay" : χωρίς replay buffer (online updates)
  - "no_target" : target network = online network (δεν freezάρουμε)
  - "double"    : Double DQN (Bonus B) — διαχωρισμός επιλογής/αξιολόγησης action

Double DQN (van Hasselt et al., 2016):
  Κανονικό DQN target: y = r + γ · max_a' Q_θ-(s',a')
  → το target net επιλέγει ΚΑΙ αξιολογεί → overestimation bias.
  Double DQN: a* = argmax Q_θ(s',a')  [online επιλέγει]
              y  = r + γ · Q_θ-(s', a*) [target αξιολογεί]
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.networks      import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim    : int,
        n_actions    : int,
        # --- Replay Buffer ---
        buffer_capacity : int   = 50_000,
        batch_size      : int   = 64,
        min_buffer_size : int   = 1_000,   # pre-fill πριν αρχίσει training
        # --- Q-Network ---
        hidden_dim   : int   = 128,
        lr           : float = 1e-3,
        gamma        : float = 0.99,
        # --- Target Network ---
        target_update_freq: int = 200,     # hard update κάθε C steps
        # --- Exploration ---
        eps_start    : float = 1.0,
        eps_end      : float = 0.05,
        eps_decay_steps: int = 100_000,    # γραμμική αποσύνθεση
        # --- Ablation Mode ---
        mode         : str   = "full",     # "full" | "no_replay" | "no_target" | "double"
        device       : torch.device = None,
    ):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.batch_size = batch_size
        self.min_buffer_size    = min_buffer_size
        self.target_update_freq = target_update_freq
        self.mode       = mode
        self.device     = device or torch.device("cpu")

        # ── Online Network ────────────────────────────────────────────────────
        # Αυτό εκπαιδεύεται σε κάθε step
        self.online_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer  = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # ── Target Network ────────────────────────────────────────────────────
        # Frozen copy — αντιγράφεται από online_net κάθε C steps
        # Σε "no_target" mode ΔΕΝ freezάρουμε — target = online (κινητός στόχος)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # δεν εκπαιδεύεται ποτέ απευθείας

        # ── Replay Buffer ─────────────────────────────────────────────────────
        # Σε "no_replay" mode κρατάμε buffer size=1 (essentially online)
        buf_cap = 1 if mode == "no_replay" else buffer_capacity
        self.buffer = ReplayBuffer(buf_cap, state_dim)

        # ── Epsilon ───────────────────────────────────────────────────────────
        self.eps            = eps_start
        self.eps_end        = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_history    = []   # για το task2_dqn_epsilon.png

        # ── Counters ──────────────────────────────────────────────────────────
        self.total_steps = 0

    # ── Action Selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy: με πιθανότητα ε επιλέγουμε τυχαία action (exploration),
        αλλιώς την action με το μέγιστο Q-value (exploitation).
        """
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax(dim=1).item()

    def _decay_epsilon(self):
        """Γραμμική αποσύνθεση ε από eps_start → eps_end."""
        frac     = min(self.total_steps / self.eps_decay_steps, 1.0)
        self.eps = (1.0 - frac) * 1.0 + frac * self.eps_end

    # ── Training Step ─────────────────────────────────────────────────────────
    def step(self, state, action, reward, next_state, done) -> float | None:
        """
        Αποθηκεύει transition και εκτελεί ένα gradient update αν είναι έτοιμο.
        Returns: loss value (ή None αν δεν έγινε update ακόμα)
        """
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._decay_epsilon()
        self.eps_history.append(self.eps)

        # Σε "no_replay" mode: εκπαιδεύουμε από το μοναδικό transition στον buffer
        if self.mode == "no_replay":
            min_size = 1
        else:
            min_size = self.min_buffer_size

        # Δεν αρχίζουμε training πριν γεμίσει αρκετά ο buffer
        if len(self.buffer) < min_size:
            return None

        return self._update()

    def _update(self) -> float:
        """
        Ένα gradient descent step πάνω στο Bellman error.

        TD target: y_i = r_i + γ · max_a' Q_θ-(s'_i, a') · (1 - done_i)
        Loss: Huber(Q_θ(s_i, a_i) - y_i)  ← MSE για μικρά errors, L1 για μεγάλα
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Μετατροπή σε PyTorch tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── Q(s, a) από online network ────────────────────────────────────────
        # gather: επιλέγει το Q-value της action που πράγματι επιλέχθηκε
        q_values = self.online_net(states_t)
        q_sa     = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── TD Target ─────────────────────────────────────────────────────────
        with torch.no_grad():
            if self.mode == "no_target":
                # Ablation: moving target → αστάθεια
                next_q     = self.online_net(next_states_t)
                max_next_q = next_q.max(dim=1).values

            elif self.mode == "double":
                # Double DQN: διαχωρισμός επιλογής και αξιολόγησης action.
                # Πρόβλημα κανονικού DQN: το target net επιλέγει ΚΑΙ αξιολογεί
                # την ίδια action → συστηματικό overestimation των Q-values.
                #
                # Βήμα 1: online net επιλέγει την καλύτερη action για s'
                best_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
                # Βήμα 2: target net αξιολογεί ΑΥΤΗ ΑΚΡΙΒΩΣ την action
                # (δεν αφήνουμε το target net να επιλέξει και να αξιολογήσει)
                max_next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)

            else:
                # Standard DQN (full / no_replay)
                next_q     = self.target_net(next_states_t)
                max_next_q = next_q.max(dim=1).values

            # Αν done=True, δεν υπάρχει επόμενο state → target = r
            target = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        # ── Huber Loss ────────────────────────────────────────────────────────
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping για να αποφύγουμε exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Hard Target Update ────────────────────────────────────────────────
        # Κάθε C steps: αντιγράφουμε τα weights του online_net στο target_net
        if self.mode != "no_target" and self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def load(self, path: str):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
