"""
File 6: 06_llm_finetuning_evaluation.py
======================================
GenAI & Deep Learning Assessment - Module 6
Topic: LLM Fine-Tuning Pipeline & Quantitative Evaluation (ROUGE / BLEU / Perplexity)

Description:
This module covers the fine-tuning workflow for generative language models and
implements quantitative benchmark evaluation metrics including ROUGE-1, ROUGE-2,
ROUGE-L, BLEU score, and Perplexity calculation to evaluate output quality.
"""

import math
import numpy as np
from collections import Counter

# ============================================================
# EVALUATION METRICS IMPLEMENTATION
# ============================================================

def compute_ngram_precision(reference_tokens, candidate_tokens, n=1):
    """Computes modified n-gram precision for BLEU calculation."""
    if len(candidate_tokens) < n:
        return 0.0

    candidate_ngrams = [tuple(candidate_tokens[i:i+n]) for i in range(len(candidate_tokens)-n+1)]
    reference_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens)-n+1)]

    cand_counts = Counter(candidate_ngrams)
    ref_counts = Counter(reference_ngrams)

    clipped_counts = {ngram: min(count, ref_counts[ngram]) for ngram, count in cand_counts.items()}
    
    total_clipped = sum(clipped_counts.values())
    total_candidate = max(1, len(candidate_ngrams))
    
    return total_clipped / total_candidate

def compute_bleu_score(reference: str, candidate: str, max_n=4) -> float:
    """Computes simplified BLEU metric for text evaluation."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(cand_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p_n = compute_ngram_precision(ref_tokens, cand_tokens, n=n)
        if p_n > 0:
            precisions.append(p_n)
        else:
            precisions.append(1e-5) # Smoothing

    # Brevity Penalty (BP)
    r = len(ref_tokens)
    c = len(cand_tokens)
    bp = 1.0 if c > r else math.exp(1 - (r / c))

    log_sum = sum(math.log(p) for p in precisions) / max_n
    bleu = bp * math.exp(log_sum)
    return bleu

def compute_lcs_length(seq1, seq2):
    """Computes Longest Common Subsequence (LCS) length for ROUGE-L."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_scores(reference: str, candidate: str) -> dict:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L recall/precision/F1 scores."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # ROUGE-1
    r1_prec = compute_ngram_precision(ref_tokens, cand_tokens, n=1)
    r1_rec = compute_ngram_precision(cand_tokens, ref_tokens, n=1)
    r1_f1 = (2 * r1_prec * r1_rec) / (r1_prec + r1_rec + 1e-9)

    # ROUGE-2
    r2_prec = compute_ngram_precision(ref_tokens, cand_tokens, n=2)
    r2_rec = compute_ngram_precision(cand_tokens, ref_tokens, n=2)
    r2_f1 = (2 * r2_prec * r2_rec) / (r2_prec + r2_rec + 1e-9)

    # ROUGE-L
    lcs_len = compute_lcs_length(ref_tokens, cand_tokens)
    rl_rec = lcs_len / max(1, len(ref_tokens))
    rl_prec = lcs_len / max(1, len(cand_tokens))
    rl_f1 = (2 * rl_prec * rl_rec) / (rl_prec + rl_rec + 1e-9)

    return {
        "ROUGE-1": {"precision": r1_prec, "recall": r1_rec, "f1": r1_f1},
        "ROUGE-2": {"precision": r2_prec, "recall": r2_rec, "f1": r2_f1},
        "ROUGE-L": {"precision": rl_prec, "recall": rl_rec, "f1": rl_f1}
    }

def compute_perplexity(logits_or_losses):
    """Computes model perplexity from cross-entropy loss values."""
    mean_loss = np.mean(logits_or_losses)
    return math.exp(mean_loss)

# ============================================================
# FINE-TUNING PIPELINE WORKFLOW (MOCK & HF DEMO)
# ============================================================

def run_finetuning_evaluation_demo():
    print("==================================================")
    print("LLM Fine-Tuning & Quantitative Evaluation Suite")
    print("==================================================")

    reference_text = "Retrieval Augmented Generation improves accuracy and reduces hallucinations in large language models."
    candidate_text = "RAG improves model accuracy and reduces hallucinations in large language model responses."

    print(f"\nReference Summary:\n  '{reference_text}'")
    print(f"\nCandidate Generated Summary:\n  '{candidate_text}'\n")

    # 1. Compute BLEU Score
    bleu = compute_bleu_score(reference_text, candidate_text)
    print(f"BLEU-4 Score: {bleu:.4f}")

    # 2. Compute ROUGE Scores
    rouge_metrics = compute_rouge_scores(reference_text, candidate_text)
    print("\n--- ROUGE Evaluation Breakdown ---")
    for metric, scores in rouge_metrics.items():
        print(f"{metric:8s} | Precision: {scores['precision']:.4f} | Recall: {scores['recall']:.4f} | F1: {scores['f1']:.4f}")

    # 3. Compute Perplexity
    sample_losses = [1.85, 1.42, 1.21, 1.15, 1.08]
    ppl = compute_perplexity(sample_losses)
    print(f"\nModel Perplexity (PPL): {ppl:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_finetuning_evaluation_demo()
