"""
Evaluation utilities for ChartCaptioner.

- BLEU-1 through BLEU-4 (via nltk)
- Per-sample caption generation + display
- Batch evaluation on the val loader
"""

import collections
import math
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# n-gram helpers (no external dependency except nltk for BLEU)
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _count_clip(cand_ngrams, ref_ngrams):
    cand_counts = collections.Counter(cand_ngrams)
    ref_counts  = collections.Counter(ref_ngrams)
    clipped     = {ng: min(cnt, ref_counts[ng]) for ng, cnt in cand_counts.items()}
    return sum(clipped.values()), sum(cand_counts.values())


def bleu_score(
    hypotheses: List[str],
    references: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Corpus-level BLEU-max_n.

    Parameters
    ----------
    hypotheses : list of generated captions
    references : list of ground-truth captions (one per hypothesis)
    """
    assert len(hypotheses) == len(references)

    clipped_counts = [0] * max_n
    total_counts   = [0] * max_n
    total_hyp_len  = 0
    total_ref_len  = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tok = hyp.lower().split()
        ref_tok = ref.lower().split()

        total_hyp_len += len(hyp_tok)
        total_ref_len += len(ref_tok)

        for n in range(1, max_n + 1):
            c, t = _count_clip(_ngrams(hyp_tok, n), _ngrams(ref_tok, n))
            clipped_counts[n-1] += c
            total_counts[n-1]   += t

    # brevity penalty
    if total_hyp_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / max(1, total_hyp_len))
    else:
        bp = 1.0

    # geometric mean of precisions
    log_avg = 0.0
    for n in range(max_n):
        c = clipped_counts[n]
        t = total_counts[n]
        if smooth:
            c += 1
            t += 1
        if t == 0 or c == 0:
            return 0.0
        log_avg += math.log(c / t) / max_n

    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# Batch generation for evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_captions_batch(
    model,
    images: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 100,
    temperature:    float = 1.0,
    top_k:          int   = 50,
):
    """
    Generate captions for a batch of images.
    Returns list[str].
    """
    model.eval()
    device  = next(model.parameters()).device
    images  = images.to(device)
    B       = images.size(0)

    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 1
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2

    enc_out = model.encode(images)                       # (B, N, D)
    tokens  = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits     = model.decoder(tokens, enc_out)[:, -1, :]
        logits     = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs  = F.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, 1)              # (B, 1)
        tokens = torch.cat([tokens, next_t], dim=1)
        finished |= (next_t.squeeze(-1) == eos_id)
        if finished.all():
            break

    captions = []
    for seq in tokens:
        captions.append(
            tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        )
    return captions


# ---------------------------------------------------------------------------
# Full BLEU evaluation on a DataLoader
# ---------------------------------------------------------------------------

def compute_bleu(
    model,
    val_loader,
    tokenizer,
    device,
    max_samples: int = 500,
    temperature: float = 1.0,
    top_k:       int   = 50,
) -> float:
    """
    Runs generation on up to max_samples val examples and returns BLEU-4.
    """
    model.eval()
    hypotheses = []
    references = []

    n_processed = 0

    for images, input_tok, target_tok in val_loader:
        if n_processed >= max_samples:
            break

        # Decode references (target_tok shifted by 1 from BOS)
        for t in target_tok:
            references.append(
                tokenizer.decode(t.tolist(), skip_special_tokens=True)
            )

        caps = generate_captions_batch(model, images, tokenizer,
                                       temperature=temperature, top_k=top_k)
        hypotheses.extend(caps)
        n_processed += len(images)

    score = bleu_score(hypotheses[:max_samples], references[:max_samples])
    print(f"  BLEU-4 ({min(n_processed, max_samples)} samples): {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Pretty-print a few val examples
# ---------------------------------------------------------------------------

def show_examples(
    model,
    val_loader,
    tokenizer,
    device,
    n: int = 5,
    temperature: float = 0.8,
    top_k:       int   = 50,
):
    model.eval()
    images, _, target_tok = next(iter(val_loader))

    # pick n random samples
    idxs = random.sample(range(len(images)), min(n, len(images)))

    caps = generate_captions_batch(
        model, images[idxs], tokenizer, temperature=temperature, top_k=top_k
    )
    refs = [tokenizer.decode(target_tok[i].tolist(), skip_special_tokens=True)
            for i in idxs]

    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    for i, (cap, ref) in enumerate(zip(caps, refs)):
        print(f"\n[Sample {i+1}]")
        print(f"  Reference : {ref}")
        print(f"  Generated : {cap}")
    print("="*70)
