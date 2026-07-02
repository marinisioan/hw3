"""
LLM Utilities για Bonus A (GRPO)
==================================
Φόρτωση GSM8K, parsing απαντήσεων, reward computation.

── ΠΑΡΑΤΗΡΗΣΗ για compute_reward ──────────────────────────────────────────────
Ο κώδικας του Gemini κάνει string comparison μεταξύ extracted numbers.
Αυτό έχει ένα subtle bug: "3.0" != "3" ενώ αριθμητικά είναι ίσα.
Επίσης "1,000" vs "1000" θα αποτύχει (το GSM8K χρησιμοποιεί commas).
Διορθώνουμε με numeric comparison μετά από normalization.
"""

import re
from datasets import load_dataset


def load_gsm8k_subset(num_samples: int = 200):
    """
    Φορτώνει τα πρώτα num_samples προβλήματα από το GSM8K training set.
    Το GSM8K (Grade School Math 8K) περιέχει 8,500 προβλήματα αριθμητικής
    με step-by-step λύσεις. Κάθε item έχει 'question' και 'answer'.
    Η 'answer' τελειώνει με '#### <number>' που είναι η τελική απάντηση.
    """
    return load_dataset("openai/gsm8k", "main", split=f"train[:{num_samples}]")


def parse_gsm8k_answer(answer_str: str) -> str:
    """
    Εξάγει την τελική αριθμητική απάντηση από το GSM8K format.
    Format: "...step by step solution... #### 42"
    Returns: "42" (string — την κάνουμε numeric στο compute_reward)
    """
    parts = answer_str.split("####")
    if len(parts) != 2:
        return ""
    # Αφαιρούμε commas (π.χ. "1,234" → "1234") και whitespace
    return parts[1].strip().replace(",", "")


def extract_predicted_number(completion: str) -> float | None:
    """
    Εξάγει τον αριθμό που προβλέπει το μοντέλο από το completion.

    Προσπαθούμε δύο στρατηγικές με σειρά προτεραιότητας:
    1. Ψάχνουμε για "The answer is: <number>" (το format που ζητάμε στο prompt)
    2. Fallback: τελευταίος αριθμός στο κείμενο

    Returns: float ή None αν δεν βρεθεί αριθμός
    """
    # Στρατηγική 1: explicit format
    pattern_explicit = r'[Tt]he answer is[:\s]+(-?[\d,]+(?:\.\d+)?)'
    match = re.search(pattern_explicit, completion)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass

    # Στρατηγική 2: τελευταίος αριθμός στο completion
    all_numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', completion)
    if all_numbers:
        try:
            return float(all_numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Rule-based verifiable reward: 1.0 αν σωστό, 0.0 αλλιώς.

    Το "verifiable" εδώ σημαίνει ότι ο έλεγχος είναι ντετερμινιστικός
    και δεν απαιτεί reward model (δεν υπάρχει subjectivity).
    Αυτό είναι ακριβώς το κίνητρο πίσω από το GRPO για math tasks:
    σε αντίθεση με open-ended text generation, τα math problems έχουν
    μοναδική σωστή απάντηση που μπορεί να ελεγχθεί αυτόματα.

    ── Διόρθωση από Gemini ──
    String comparison "42" == "42.0" → False (λάθος).
    Numeric comparison 42.0 == 42.0 → True (σωστό).
    """
    pred = extract_predicted_number(completion)
    if pred is None:
        return 0.0

    try:
        gt = float(ground_truth.replace(",", ""))
    except ValueError:
        return 0.0

    # Numeric comparison με μικρή ανοχή για floating point
    return 1.0 if abs(pred - gt) < 1e-4 else 0.0
