"""
Replay Buffer για DQN
=====================
Ένα circular (ring) buffer σταθερής χωρητικότητας.

Γιατί χρειαζόμαστε replay buffer;
  Χωρίς αυτό, το δίκτυο εκπαιδεύεται σε διαδοχικά (s,a,r,s') tuples
  που είναι έντονα συσχετισμένα χρονικά. Αυτό σπάει την παραδοχή IID
  των stochastic gradient methods και οδηγεί σε αστάθεια.
  Με τυχαία δειγματοληψία από τον buffer, "σπάμε" αυτή τη συσχέτιση.

Υλοποίηση:
  - Χρησιμοποιούμε pre-allocated numpy arrays (όχι Python lists) για ταχύτητα.
  - Ο pointer `_ptr` δείχνει πού θα γραφεί το επόμενο transition.
  - Όταν γεμίσει, ο pointer κυκλώνει (% capacity) και αντικαθιστά
    τα παλιότερα transitions (FIFO).
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        """
        Args:
            capacity  : μέγιστος αριθμός transitions που αποθηκεύουμε
            state_dim : διάσταση του state vector (π.χ. 8 για LunarLander)
        """
        self.capacity = capacity
        self._ptr  = 0        # δείκτης στη θέση εγγραφής
        self._size = 0        # πόσα έχουμε αποθηκεύσει μέχρι τώρα

        # Pre-allocate arrays — πολύ πιο γρήγορο από append σε list
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity,),           dtype=np.int64)
        self.rewards     = np.zeros((capacity,),           dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity,),           dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Αποθηκεύει ένα transition στον buffer."""
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)

        # Κυκλική αντικατάσταση: μόλις γεμίσει, αντικαθιστούμε τα παλιότερα
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Τυχαία δειγματοληψία batch από τον buffer.
        Returns numpy arrays — θα τα μετατρέψουμε σε tensors στον agent.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self._size
