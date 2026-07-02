"""
Bonus A — Option A: GRPO Fine-tuning με Rule-Based Reward
===========================================================
Μοντέλο   : Qwen/Qwen2.5-0.5B-Instruct
Dataset   : GSM8K (Grade School Math, subset 200 προβλημάτων)
Reward    : rule-based — σύγκριση τελικού αριθμού με ground truth
Steps     : 150
G         : 8 completions ανά prompt

── ΤΙ ΚΑΝΟΥΜΕ ΚΑΙ ΓΙΑΤΙ ────────────────────────────────────────────────────
Το Qwen2.5-0.5B-Instruct έχει ήδη εκπαιδευτεί με SFT (supervised fine-tuning).
Εμείς κάνουμε ένα μικρό RLHF-style fine-tuning βήμα με GRPO:
  1. Για κάθε ερώτηση, παράγουμε G=8 απαντήσεις (completions)
  2. Ελέγχουμε ποιες είναι σωστές → reward ∈ {0,1}
  3. Standardize rewards → group-relative advantages
  4. Policy gradient update + KL penalty προς το reference model

── ΚΡΙΣΙΜΗ ΜΑΘΗΜΑΤΙΚΗ ΔΙΟΡΘΩΣΗ ─────────────────────────────────────────────
FATAL BUG: Ο προηγούμενος κώδικας εκτελούσε forward pass ΜΟΝΟ πάνω στο completion. 
  Αυτό καταστρέφει τη δεσμευμένη πιθανότητα P(Completion | Prompt), καθώς 
  το Causal LM αδυνατεί να αξιολογήσει λογικά τη συνέχεια χωρίς το context.
Διόρθωση: Εκτελούμε forward pass σε ολόκληρη την ακολουθία (full_outputs) 
  ώστε να διατηρηθεί ο μηχανισμός Attention, και απομονώνουμε (slice) τα logits 
  που αντιστοιχούν αυστηρά στην παραγωγή των completion tokens.
"""

import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.llm_utils  import load_gsm8k_subset, parse_gsm8k_answer, compute_reward
from agents.grpo_math import compute_grpo_loss

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS          = 150
G              = 8
MAX_NEW_TOKENS = 128
TEMPERATURE    = 0.8
BETA           = 0.04
LR             = 1e-5
LOG_EVERY      = 10

def get_prompt(question: str) -> str:
    return (
        f"<|im_start|>user\n"
        f"{question}\n"
        f"Solve step by step. End with 'The answer is: <number>'.\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Steps      : {STEPS}  |  G={G}  |  β={BETA}")

    print("\nLoading models (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

    dataset = load_gsm8k_subset(STEPS)
    print(f"Loaded {len(dataset)} GSM8K problems\n")

    os.makedirs("results", exist_ok=True)
    log_data   = []
    t_start    = time.time()

    # Pre-training baseline
    print("── Pre-training baseline (first 20 problems) ──")
    baseline_rewards = []
    policy_model.eval()
    with torch.no_grad():
        for item in list(dataset)[:20]:
            prompt   = get_prompt(item["question"])
            gt       = parse_gsm8k_answer(item["answer"])
            enc      = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
            out      = policy_model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            comp = tokenizer.decode(out[0], skip_special_tokens=True)
            baseline_rewards.append(compute_reward(comp, gt))
    baseline_acc = sum(baseline_rewards) / len(baseline_rewards)
    print(f"  Baseline accuracy (greedy): {baseline_acc:.1%}\n")

    # ── Training Loop ──────────────────────────────────────────────────────────
    print("── Training ──")
    for step, item in enumerate(dataset):
        question = item["question"]
        gt_answer = parse_gsm8k_answer(item["answer"])
        prompt = get_prompt(question)

        # ── Phase 1: Generation ────────────────────────
        policy_model.eval()
        enc = tokenizer([prompt] * G, return_tensors="pt", padding=True).to(device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            full_outputs = policy_model.generate(
                **enc,
                max_new_tokens  = MAX_NEW_TOKENS,
                do_sample       = True,
                temperature     = TEMPERATURE,
                pad_token_id    = tokenizer.eos_token_id,
            )

        completion_ids = full_outputs[:, prompt_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards     = [compute_reward(comp, gt_answer) for comp in completions]

        # ── Phase 2: Forward pass για gradients ────────────────
        policy_model.train()
        
        # Περνάμε όλο το output (μαζί με το prompt context)
        policy_logits_full = policy_model(full_outputs).logits   
        
        with torch.no_grad():
            ref_logits_full = ref_model(full_outputs).logits

        # Απομονώνουμε τα logits που *προβλέπουν* τα completion tokens.
        # Το token στο prompt_len-1 προβλέπει το πρώτο token του completion.
        policy_logits = policy_logits_full[:, prompt_len - 1 : -1, :] 
        ref_logits = ref_logits_full[:, prompt_len - 1 : -1, :]

        # ── Phase 3: GRPO Loss ───────────────────────────────────────────────
        loss, kl_div, mean_reward = compute_grpo_loss(
            policy_logits  = policy_logits,
            ref_logits     = ref_logits,
            completion_ids = completion_ids,
            rewards        = rewards,
            beta           = BETA,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.empty_cache() if device.type == "cuda" else None

        elapsed = (time.time() - t_start) / 60
        log_data.append([step + 1, loss.item(), kl_div, mean_reward, elapsed])

        if (step + 1) % LOG_EVERY == 0 or step == 0:
            n_correct = sum(rewards)
            print(
                f"  Step {step+1:>3}/{STEPS} | "
                f"Loss: {loss.item():>7.4f} | "
                f"KL: {kl_div:>6.4f} | "
                f"Reward: {mean_reward:.3f} ({n_correct}/{G} correct) | "
                f"t={elapsed:.1f}min"
            )

    # ── Post-training evaluation ───────────────────────────────────────────────
    print("\n── Post-training evaluation (first 20 problems) ──")
    post_rewards = []
    policy_model.eval()
    with torch.no_grad():
        for item in list(dataset)[:20]:
            prompt = get_prompt(item["question"])
            gt     = parse_gsm8k_answer(item["answer"])
            enc    = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
            out    = policy_model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            comp = tokenizer.decode(out[0], skip_special_tokens=True)
            post_rewards.append(compute_reward(comp, gt))
    post_acc = sum(post_rewards) / len(post_rewards)

    total_time = (time.time() - t_start) / 60
    print(f"  Post-training accuracy (greedy): {post_acc:.1%}")
    print(f"  Improvement: {post_acc - baseline_acc:+.1%}")
    print(f"  Total wall-clock: {total_time:.1f} min\n")

    csv_path = "results/task_bonus_a_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "kl_div", "mean_reward", "elapsed_min"])
        writer.writerows(log_data)
    print(f"  Log saved: {csv_path}")

    policy_model.save_pretrained("results/qwen_grpo_finetuned")
    tokenizer.save_pretrained("results/qwen_grpo_finetuned")
    print("  Model saved: results/qwen_grpo_finetuned/")

if __name__ == "__main__":
    main()