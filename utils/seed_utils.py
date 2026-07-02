"""
Reproducibility Utilities
==========================
Κεντρική συνάρτηση για global seed — πρέπει να καλείται στην αρχή
κάθε script για reproducible αποτελέσματα.

Γιατί χρειαζόμαστε seed σε τόσα μέρη;
  - numpy: για random sampling (replay buffer, bin allocation)
  - random: Python's built-in random module
  - torch: για weight initialization και dropout
  - torch.cuda: για GPU operations
  - env.reset(seed=...): για την αρχική κατάσταση του περιβάλλοντος
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Ορίζει global seed σε όλες τις βιβλιοθήκες."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops (ελαφρώς πιο αργό αλλά fully reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Επιλέγει GPU αν υπάρχει, αλλιώς CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    return device
