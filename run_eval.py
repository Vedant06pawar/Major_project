"""
run_eval.py
───────────
Standalone evaluation script for the trained ChartCaptioner.

Metrics computed
────────────────
  • BLEU-1, BLEU-2, BLEU-3, BLEU-4   (n-gram precision)
  • METEOR                            (harmonic mean of precision + recall
                                       with stemming + synonym matching)
  • ROUGE-L                           (longest common subsequence F1)
  • CIDEr                             (TF-IDF weighted n-gram consensus)
  • Perplexity                        (teacher-forced, measures model confidence)
  • Avg / Min / Max caption length    (tokens and words)

Outputs
───────
  • Live progress printed to console
  • eval_results/summary.txt       — human-readable report
  • eval_results/per_sample.csv    — per-row hypothesis / reference / scores
  • eval_results/score_dist.png    — histogram of per-sample BLEU-4 scores

Usage
─────
  python run_eval.py --checkpoint checkpoints/best.pt

  # All options:
  python run_eval.py \
      --checkpoint checkpoints/best.pt \
      --split      val \
      --max_samples 1000 \
      --batch_size 8 \
      --beam_size  5 \
      --length_penalty 1.2 \
      --max_tokens 200 \
      --out_dir    eval_results
"""

import argparse
import collections
import csv
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# ── import from the same folder ──────────────────────────────────────────────
from model   import ChartCaptioner
from dataset import build_dataloaders
from evaluate import bleu_score, generate_captions_batch


# ═════════════════════════════════════════════════════════════════════════════
# Metric implementations  (all pure-Python, no extra pip installs needed)
# ═════════════════════════════════════════════════════════════════════════════

# ── BLEU per sample ───────────────────────────────────────────────────────────

def bleu_n(hyp_tok: List[str], ref_tok: List[str], n: int) -> float:
    """Sentence-level BLEU-n with add-1 smoothing."""
    if len(hyp_tok) < n:
        return 0.0
    def ngrams(toks, k):
        return collections.Counter(tuple(toks[i:i+k]) for i in range(len(toks)-k+1))
    hyp_ng = ngrams(hyp_tok, n)
    ref_ng = ngrams(ref_tok, n)
    clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    total   = sum(hyp_ng.values())
    if total == 0:
        return 0.0
    prec = (clipped + 1) / (total + 1)          # smoothed
    bp   = min(1.0, math.exp(1 - len(ref_tok) / max(1, len(hyp_tok))))
    return bp * prec


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def rouge_l(hyp_tok: List[str], ref_tok: List[str]) -> float:
    """ROUGE-L F1 via LCS length."""
    h, r = len(hyp_tok), len(ref_tok)
    if h == 0 or r == 0:
        return 0.0
    # LCS via DP
    dp = [[0] * (r + 1) for _ in range(h + 1)]
    for i in range(1, h + 1):
        for j in range(1, r + 1):
            if hyp_tok[i-1] == ref_tok[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[h][r]
    prec = lcs / h
    rec  = lcs / r
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ── METEOR ────────────────────────────────────────────────────────────────────

def _simple_stem(word: str) -> str:
    """Very lightweight suffix stripping (no NLTK needed)."""
    for suffix in ("ing", "tion", "tions", "ed", "er", "est", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def meteor(hyp_tok: List[str], ref_tok: List[str], alpha=0.9, beta=3, gamma=0.5) -> float:
    """
    METEOR (simplified — unigram F-mean with stemming, no WordNet synonyms).
    Alpha controls F-mean weighting; beta/gamma control the fragmentation penalty.
    """
    if not hyp_tok or not ref_tok:
        return 0.0

    # Match exact tokens first, then stems
    ref_counts = collections.Counter(ref_tok)
    matches    = 0
    for tok in hyp_tok:
        if ref_counts.get(tok, 0) > 0:
            matches += 1
            ref_counts[tok] -= 1

    # Stem pass for unmatched tokens
    ref_stems   = collections.Counter(_simple_stem(t) for t in ref_tok)
    hyp_stems   = [_simple_stem(t) for t in hyp_tok]
    stem_matches = 0
    for s in hyp_stems:
        if ref_stems.get(s, 0) > 0:
            stem_matches += 1
            ref_stems[s] -= 1

    total_matches = matches + stem_matches * 0.8   # partial credit for stems
    if total_matches == 0:
        return 0.0

    prec = total_matches / len(hyp_tok)
    rec  = total_matches / len(ref_tok)
    f    = prec * rec / (alpha * prec + (1 - alpha) * rec)

    # Fragmentation penalty (chunks = contiguous matched runs, simplified)
    frag_penalty = gamma * (1 / max(1, total_matches)) ** beta
    return max(0.0, f * (1 - frag_penalty))


# ── CIDEr ─────────────────────────────────────────────────────────────────────

def _cider_ngrams(tokens: List[str], n: int):
    return collections.Counter(
        tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
    )


def cider_score(hypotheses: List[str], references: List[str], n_max: int = 4) -> float:
    """
    Corpus CIDEr-D: TF-IDF weighted n-gram similarity, averaged over n=1..n_max.
    """
    N = len(hypotheses)
    if N == 0:
        return 0.0

    scores_per_n = []
    for n in range(1, n_max + 1):
        # Build document frequency over references
        df: Dict[tuple, int] = collections.defaultdict(int)
        ref_ngrams_list = []
        hyp_ngrams_list = []
        for hyp, ref in zip(hypotheses, references):
            h_ng = _cider_ngrams(hyp.lower().split(), n)
            r_ng = _cider_ngrams(ref.lower().split(), n)
            hyp_ngrams_list.append(h_ng)
            ref_ngrams_list.append(r_ng)
            for g in set(r_ng.keys()):
                df[g] += 1

        # IDF
        idf = {g: math.log((N + 1) / (cnt + 1)) for g, cnt in df.items()}

        # Per-sample cosine similarity between TF-IDF vectors
        sample_scores = []
        for h_ng, r_ng in zip(hyp_ngrams_list, ref_ngrams_list):
            # TF = count / total_count
            h_total = max(1, sum(h_ng.values()))
            r_total = max(1, sum(r_ng.values()))

            dot, h_norm, r_norm = 0.0, 0.0, 0.0
            all_grams = set(h_ng.keys()) | set(r_ng.keys())
            for g in all_grams:
                idf_g = idf.get(g, 0.0)
                h_tfidf = (h_ng.get(g, 0) / h_total) * idf_g
                r_tfidf = (r_ng.get(g, 0) / r_total) * idf_g
                dot    += h_tfidf * r_tfidf
                h_norm += h_tfidf ** 2
                r_norm += r_tfidf ** 2

            denom = math.sqrt(h_norm) * math.sqrt(r_norm)
            sample_scores.append(dot / denom if denom > 0 else 0.0)

        scores_per_n.append(sum(sample_scores) / N)

    return 10.0 * sum(scores_per_n) / n_max   # CIDEr conventionally scaled ×10


# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, val_loader, device, max_batches: int = 50) -> float:
    """Teacher-forced cross-entropy perplexity on the val set."""
    model.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0.0
    total_tok  = 0

    for i, (images, input_tok, target_tok) in enumerate(val_loader):
        if i >= max_batches:
            break
        images     = images.to(device)
        input_tok  = input_tok.to(device)
        target_tok = target_tok.to(device)

        logits = model(images, input_tok)        # (B, T, V)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), target_tok.reshape(B * T))

        non_pad = (target_tok != 0).sum().item()
        total_loss += loss.item()
        total_tok  += non_pad

    return math.exp(total_loss / max(1, total_tok))


# ═════════════════════════════════════════════════════════════════════════════
# Caption length statistics
# ═════════════════════════════════════════════════════════════════════════════

def length_stats(captions: List[str]) -> dict:
    lengths = [len(c.split()) for c in captions]
    return {
        "avg_words": sum(lengths) / max(1, len(lengths)),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "med_words": sorted(lengths)[len(lengths) // 2],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main generation + scoring loop
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_generation(model, val_loader, tokenizer, device,
                   max_samples, beam_size, length_penalty, max_tokens):
    """
    Generate captions for up to max_samples val images using beam search.
    Returns (hypotheses, references, per_sample_scores).
    """
    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []

    bos_id = tokenizer.bos_token_id or 1
    eos_id = tokenizer.eos_token_id or 2

    print(f"\nGenerating captions (beam_size={beam_size}, "
          f"length_penalty={length_penalty}, max_tokens={max_tokens})...")

    n_done  = 0
    t_start = time.time()

    for images, _, target_tok in val_loader:
        if n_done >= max_samples:
            break

        images = images.to(device)

        # Generate captions and decode references together per-image
        # so counts always stay aligned even when max_samples cuts mid-batch
        for i in range(images.size(0)):
            if n_done >= max_samples:
                break

            # ── Reference ────────────────────────────────────────────
            # GPT-2 uses token 50256 as BOS, EOS, AND PAD simultaneously.
            # skip_special_tokens=True would wipe ALL of them and return "".
            # Strip only leading BOS + trailing PAD/EOS manually.
            pad_id     = tokenizer.pad_token_id
            bos_id_ref = tokenizer.bos_token_id
            eos_id_ref = tokenizer.eos_token_id
            ids = target_tok[i].tolist()
            if ids and ids[0] == bos_id_ref:
                ids = ids[1:]
            while ids and ids[-1] in (pad_id, eos_id_ref):
                ids.pop()
            references.append(tokenizer.decode(ids).strip())

            # ── Hypothesis ───────────────────────────────────────────
            img = images[i].unsqueeze(0)
            if beam_size > 1:
                cap = model.beam_search(
                    img, tokenizer,
                    max_new_tokens=max_tokens,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                cap = model.generate(
                    img, tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=0.8, top_k=50,
                )
            hypotheses.append(cap.strip())
            n_done += 1

        elapsed = time.time() - t_start
        rate    = n_done / elapsed
        eta     = (min(max_samples, len(val_loader.dataset)) - n_done) / max(rate, 0.001)
        print(f"  {n_done}/{min(max_samples, len(val_loader.dataset))} captions "
              f"- {rate:.1f} img/s - ETA {eta:.0f}s", end="\r")

    print()   # newline after \r
    return hypotheses, references[:len(hypotheses)]


# ═════════════════════════════════════════════════════════════════════════════
# Per-sample scoring
# ═════════════════════════════════════════════════════════════════════════════

def score_all_samples(hypotheses: List[str],
                      references: List[str]) -> List[dict]:
    rows = []
    for hyp, ref in zip(hypotheses, references):
        h = hyp.lower().split()
        r = ref.lower().split()
        rows.append({
            "hypothesis": hyp,
            "reference":  ref,
            "bleu1":   round(bleu_n(h, r, 1), 4),
            "bleu2":   round(bleu_n(h, r, 2), 4),
            "bleu3":   round(bleu_n(h, r, 3), 4),
            "bleu4":   round(bleu_n(h, r, 4), 4),
            "rouge_l": round(rouge_l(h, r),   4),
            "meteor":  round(meteor(h, r),     4),
            "hyp_len": len(h),
            "ref_len": len(r),
        })
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Report helpers
# ═════════════════════════════════════════════════════════════════════════════

def avg(values): return sum(values) / max(1, len(values))


def print_and_save_report(rows, corpus_bleu, corpus_cider, perplexity,
                          args, out_dir: Path, elapsed_gen: float):
    bleu1  = avg([r["bleu1"]   for r in rows])
    bleu2  = avg([r["bleu2"]   for r in rows])
    bleu3  = avg([r["bleu3"]   for r in rows])
    bleu4  = avg([r["bleu4"]   for r in rows])
    rougeL = avg([r["rouge_l"] for r in rows])
    met    = avg([r["meteor"]  for r in rows])
    h_lens = [r["hyp_len"] for r in rows]
    r_lens = [r["ref_len"] for r in rows]

    lines = [
        "=" * 60,
        "  ChartCaptioner - Evaluation Report",
        "=" * 60,
        f"  Checkpoint   : {args.checkpoint}",
        f"  Samples      : {len(rows)}",
        f"  Beam size    : {args.beam_size}",
        f"  Length pen.  : {args.length_penalty}",
        f"  Max tokens   : {args.max_tokens}",
        f"  Gen time     : {elapsed_gen:.1f}s  "
               f"({elapsed_gen/max(1,len(rows)):.2f}s / sample)",
        "",
        "  -- N-gram Metrics " + "-" * 42,
        f"  BLEU-1       : {bleu1:.4f}",
        f"  BLEU-2       : {bleu2:.4f}",
        f"  BLEU-3       : {bleu3:.4f}",
        f"  BLEU-4       : {bleu4:.4f}  <-- main metric",
        f"  Corpus BLEU-4: {corpus_bleu:.4f}  (corpus-level)",
        "",
        "  -- Semantic Metrics " + "-" * 40,
        f"  METEOR       : {met:.4f}",
        f"  ROUGE-L      : {rougeL:.4f}",
        f"  CIDEr        : {corpus_cider:.4f}",
        "",
        "  -- Model Confidence " + "-" * 40,
        f"  Perplexity   : {perplexity:.2f}  (lower = better)",
        "",
        "  -- Caption Length " + "-" * 42,
        f"  Hyp avg/min/max words : "
               f"{avg(h_lens):.1f} / {min(h_lens)} / {max(h_lens)}",
        f"  Ref avg/min/max words : "
               f"{avg(r_lens):.1f} / {min(r_lens)} / {max(r_lens)}",
        "",
        "  -- Score Interpretation " + "-" * 36,
        "  BLEU-4  > 0.30 = good  |  0.15-0.30 = fair  |  < 0.15 = needs work",
        "  METEOR  > 0.25 = good  |  0.15-0.25 = fair",
        "  ROUGE-L > 0.40 = good  |  0.25-0.40 = fair",
        "  CIDEr   > 1.00 = good  |  0.50-1.00 = fair  (scaled x10)",
        "  Perplexity : depends on vocab size; lower is always better",
        "=" * 60,
    ]

    report = "\n".join(lines)

    # Print safely on Windows terminals (cp1252 can't handle some unicode)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

    report_path = out_dir / "summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report saved -> {report_path}")

    return {
        "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
        "corpus_bleu4": corpus_bleu,
        "meteor": met, "rouge_l": rougeL, "cider": corpus_cider,
        "perplexity": perplexity,
    }


def save_csv(rows: List[dict], out_dir: Path):
    csv_path = out_dir / "per_sample.csv"
    fieldnames = ["hypothesis", "reference",
                  "bleu1", "bleu2", "bleu3", "bleu4",
                  "rouge_l", "meteor", "hyp_len", "ref_len"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Per-sample CSV -> {csv_path}")


def save_histogram(rows: List[dict], out_dir: Path):
    """Save a BLEU-4 distribution histogram (skipped gracefully if matplotlib missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = [r["bleu4"] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.hist(scores, bins=20, edgecolor="black", color="#4C72B0")
        plt.xlabel("Sentence BLEU-4 score")
        plt.ylabel("Number of samples")
        plt.title("Per-sample BLEU-4 Distribution")
        plt.tight_layout()
        path = out_dir / "score_dist.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Histogram     -> {path}")
    except ImportError:
        print("  (matplotlib not installed - skipping histogram)")


def show_qualitative_examples(rows: List[dict], n: int = 5):
    """Print n random examples sorted by BLEU-4 (best, median, worst)."""
    sorted_rows = sorted(rows, key=lambda r: r["bleu4"])
    n_each = max(1, n // 3)
    examples = (
        sorted_rows[:n_each]                               # worst
        + sorted_rows[len(sorted_rows)//2 - n_each//2 :
                      len(sorted_rows)//2 + n_each//2]     # median
        + sorted_rows[-n_each:]                            # best
    )
    random.shuffle(examples)

    print("\n" + "="*70)
    print("  QUALITATIVE EXAMPLES  (worst / median / best BLEU-4)")
    print("="*70)
    for i, row in enumerate(examples[:n], 1):
        print(f"\n  [{i}] BLEU-4={row['bleu4']:.4f}  "
              f"ROUGE-L={row['rouge_l']:.4f}  METEOR={row['meteor']:.4f}")
        # Encode to ASCII for safe printing on Windows cp1252 terminals
        ref = row['reference'].encode('ascii', errors='replace').decode('ascii')
        hyp = row['hypothesis'].encode('ascii', errors='replace').decode('ascii')
        print(f"  Reference : {ref}")
        print(f"  Generated : {hyp}")
    print("="*70)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("ChartCaptioner - Full Evaluation")
    p.add_argument("--checkpoint",      required=True,
                   help="Path to best.pt or last.pt checkpoint")
    p.add_argument("--dataset",         default="oroikon/chart_captioning")
    p.add_argument("--max_samples",     type=int,   default=None,
                   help="Cap number of val samples (default: full val set)")
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--beam_size",       type=int,   default=5)
    p.add_argument("--length_penalty",  type=float, default=1.2)
    p.add_argument("--max_tokens",      type=int,   default=200)
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--out_dir",         default="eval_results")
    p.add_argument("--examples",        type=int,   default=6,
                   help="Number of qualitative examples to print")
    p.add_argument("--perplexity_batches", type=int, default=50,
                   help="Max batches for perplexity (faster eval)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  ChartCaptioner Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output dir : {out_dir}")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    a    = ckpt.get("args", {})

    tokenizer = AutoTokenizer.from_pretrained(a.get("tokenizer", "gpt2"))
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = ckpt.get("vocab_size", len(tokenizer))

    model = ChartCaptioner(
        vocab_size  = vocab_size,
        embed_dim   = a.get("embed_dim",  256),
        enc_heads   = a.get("enc_heads",  8),
        enc_depth   = a.get("enc_depth",  6),
        dec_heads   = a.get("dec_heads",  8),
        dec_depth   = a.get("dec_depth",  6),
        image_size  = a.get("image_size", 224),
        max_seq_len = a.get("max_length", 128),
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")

    # ── 2. Build val loader ───────────────────────────────────────────────────
    # The dataset has train/test/validation splits. build_dataloaders picks
    # train+test by default. Force it to use the 'validation' split for eval.
    from datasets import load_dataset as _load_ds
    from dataset  import ChartCaptionDataset, get_val_transform

    raw = _load_ds(args.dataset)
    val_split_name = "validation" if "validation" in raw else "test"
    val_hf = raw[val_split_name]
    print(f"  Using split    : '{val_split_name}' ({len(val_hf)} samples)")

    val_ds = ChartCaptionDataset(
        val_hf, tokenizer,
        transform   = get_val_transform(a.get("image_size", 224)),
        max_length  = a.get("max_length", 128),
        image_col   = "image",
        caption_col = "text",
    )
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    n_val = len(val_ds)
    max_s = args.max_samples or n_val
    print(f"  Val samples : {n_val}  (evaluating {min(max_s, n_val)})")

    # ── 3. Perplexity (teacher-forced, fast) ─────────────────────────────────
    print(f"\nComputing perplexity ({args.perplexity_batches} batches)...")
    ppl = compute_perplexity(model, val_loader, device, args.perplexity_batches)
    print(f"  Perplexity: {ppl:.2f}")

    # ── 4. Generate captions ──────────────────────────────────────────────────
    t0 = time.time()
    hypotheses, references = run_generation(
        model, val_loader, tokenizer, device,
        max_samples     = max_s,
        beam_size       = args.beam_size,
        length_penalty  = args.length_penalty,
        max_tokens      = args.max_tokens,
    )
    elapsed_gen = time.time() - t0
    print(f"  Generated {len(hypotheses)} captions in {elapsed_gen:.1f}s")

    # ── 5. Score per sample ───────────────────────────────────────────────────
    print("\nScoring...")
    rows         = score_all_samples(hypotheses, references)
    corpus_bleu  = bleu_score(hypotheses, references, max_n=4)
    corpus_cider = cider_score(hypotheses, references)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    save_csv(rows, out_dir)
    save_histogram(rows, out_dir)

    # ── 7. Print report ───────────────────────────────────────────────────────
    print_and_save_report(rows, corpus_bleu, corpus_cider, ppl,
                          args, out_dir, elapsed_gen)

    # ── 8. Qualitative examples ───────────────────────────────────────────────
    show_qualitative_examples(rows, n=args.examples)

    print(f"\nAll outputs saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
