"""
Neural Network Αρχιτεκτονικές
==============================
Τρία δίκτυα για τους τρεις αλγορίθμους μας:

  QNetwork     : για DQN — εξάγει Q-value για κάθε action
  PolicyNetwork: για REINFORCE — εξάγει softmax πιθανότητες
  ActorCritic  : για A2C — κοινό backbone, δύο heads (actor + critic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Deep Q-Network.
    Input  : state vector (διάσταση state_dim)
    Output : Q(s,a) για κάθε δυνατή action (n_actions τιμές)

    Η αρχιτεκτονική είναι απλό MLP με ReLU activations.
    Το output layer δεν έχει activation — Q-values είναι unbounded.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """
    Policy Network για REINFORCE.
    Input  : state vector
    Output : softmax πιθανότητες πάνω από τις actions

    Το softmax εξασφαλίζει ότι το output είναι valid κατανομή πιθανοτήτων,
    αλλά στο training χρησιμοποιούμε log_softmax για αριθμητική σταθερότητα.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Επιστρέφουμε log-probabilities (πιο stable από softmax + log)
        return F.log_softmax(self.net(x), dim=-1)

    def get_action(self, state: torch.Tensor):
        """
        Sample μία action από την κατανομή του policy.
        Returns: (action index, log_prob της action αυτής)
        """
        logits = self.net(state)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class ActorCritic(nn.Module):
    """
    Actor-Critic δίκτυο για A2C με shared backbone.

    Αρχιτεκτονική:
        state → [shared layers] → actor head → logits (→ softmax)
                               └→ critic head → scalar V(s)

    Γιατί shared backbone;
      Τα feature representations που βοηθούν στο να κατανοήσεις το περιβάλλον
      είναι χρήσιμα και για το policy και για τη value function. Το sharing
      μειώνει και τον αριθμό των parameters.

    Εναλλακτικά, δύο ξεχωριστά δίκτυα (PolicyNetwork + ValueNetwork) δουλεύουν
    επίσης — απλώς χάνεις το sharing.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        # Shared trunk
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: logits → Categorical distribution
        self.actor_head  = nn.Linear(hidden_dim, n_actions)

        # Critic head: scalar state value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            dist   : Categorical distribution (για sampling & entropy)
            value  : scalar V(s), shape [batch]
        """
        features = self.backbone(x)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)  # [batch] not [batch,1]
        dist     = torch.distributions.Categorical(logits=logits)
        return dist, value


class ContinuousActorCritic(nn.Module):
    """
    Actor-Critic για ΣΥΝΕΧΗ action spaces (Bonus C).

    Χρησιμοποιείται για LunarLanderContinuous-v2 (action_dim=2).

    Διαφορά από discrete ActorCritic:
      Ο actor δεν βγάζει softmax πάνω από N actions.
      Αντίθετα, βγάζει μία Gaussian κατανομή N(μ, σ²) για κάθε action dimension:
        - mean   : τιμή [-1, 1] μέσω tanh (bounded actions)
        - log_std: clamp σε [-2, 2] για αριθμητική σταθερότητα

    Sampling: a ~ N(μ, σ²)  →  log_prob από το log-likelihood της Gaussian
    Entropy:  H = Σ_i 0.5 · log(2πe · σ_i²)  (ανά action dimension)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared trunk — μεγαλύτερο hidden για continuous control
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),   # Tanh αντί ReLU: πιο smooth για continuous control
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor: βγάζει mean της Gaussian για κάθε action dimension
        self.actor_mean    = nn.Linear(hidden_dim, action_dim)

        # log_std: learnable parameter (ένα scalar ανά action dim)
        # Αρχικοποίηση σε 0 → std=1 στην αρχή (καλό για exploration)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: scalar V(s)
        self.critic_head   = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            dist  : Normal distribution (για sampling, log_prob, entropy)
            value : scalar V(s)
        """
        features = self.backbone(x)

        # Mean: tanh για να μείνει στο [-1, 1] (LunarLanderContinuous action range)
        mean    = torch.tanh(self.actor_mean(features))

        # Std: exp(log_std) — clamp για αποφυγή πολύ μικρού ή μεγάλου std
        log_std = self.actor_log_std.clamp(-2.0, 2.0)
        std     = log_std.exp().expand_as(mean)

        dist    = torch.distributions.Normal(mean, std)
        value   = self.critic_head(features).squeeze(-1)
        return dist, value
