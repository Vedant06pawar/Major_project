"""
diagnose_eval.py
────────────────
Run this BEFORE run_eval.py to diagnose why scores are all 0.0.
It checks every stage of the pipeline and prints exactly what is
failing: tokenizer mismatch, empty captions, pad-only targets, etc.

Usage:
    python diagnose_eval.py --checkpoint checkpoints/best.pt
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model   import ChartCaptioner
from dataset import build_dataloaders


def diagnose(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  ChartCaptioner - Evaluation Diagnostics")
    print(f"{'='*60}\n")

    # ── 1. Load checkpoint ────────────────────────────────────────────
    print("[1] Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    a    = ckpt.get("args", {})
    print(f"    Trained for epoch : {ckpt.get('epoch', '?')}")
    print(f"    Tokenizer name    : {a.get('tokenizer', 'gpt2')}")
    print(f"    embed_dim         : {a.get('embed_dim', 256)}")
    print(f"    max_length        : {a.get('max_length', 128)}")
    print(f"    vocab_size in ckpt: {ckpt.get('vocab_size', 'NOT STORED')}")

    # ── 2. Build tokenizer ────────────────────────────────────────────
    print("\n[2] Building tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(a.get("tokenizer", "gpt2"))
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = ckpt.get("vocab_size", len(tokenizer))
    print(f"    Current vocab size : {len(tokenizer)}")
    print(f"    Checkpoint vocab   : {vocab_size}")
    print(f"    BOS id             : {tokenizer.bos_token_id}")
    print(f"    EOS id             : {tokenizer.eos_token_id}")
    print(f"    PAD id             : {tokenizer.pad_token_id}")

    if len(tokenizer) != vocab_size:
        print(f"\n    !! WARNING: vocab size MISMATCH.")
        print(f"    The model was trained with vocab_size={vocab_size}")
        print(f"    but the tokenizer now has {len(tokenizer)} tokens.")
        print(f"    This will produce garbage output.")

    # ── 3. Build model ─────────────────────────────────────────────────
    print("\n[3] Loading model...")
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
    print(f"    Model loaded OK")

    # ── 4. Load one batch from val loader ─────────────────────────────
    print("\n[4] Loading one val batch...")
    _, val_loader = build_dataloaders(
        tokenizer   = tokenizer,
        image_size  = a.get("image_size", 224),
        max_length  = a.get("max_length", 128),
        batch_size  = 4,
        num_workers = 0,
    )
    images, input_tok, target_tok = next(iter(val_loader))
    print(f"    images shape    : {images.shape}")
    print(f"    input_tok shape : {input_tok.shape}")
    print(f"    target_tok shape: {target_tok.shape}")

    # ── 5. Inspect raw token ids ──────────────────────────────────────
    print("\n[5] Inspecting raw token sequences (sample 0)...")
    it = input_tok[0].tolist()
    tt = target_tok[0].tolist()
    print(f"    input_tok  ids : {it[:20]}{'...' if len(it)>20 else ''}")
    print(f"    target_tok ids : {tt[:20]}{'...' if len(tt)>20 else ''}")

    # Check how many non-pad tokens exist
    n_nonpad_input  = sum(1 for x in it if x != tokenizer.pad_token_id)
    n_nonpad_target = sum(1 for x in tt if x != tokenizer.pad_token_id)
    print(f"    Non-pad input tokens  : {n_nonpad_input} / {len(it)}")
    print(f"    Non-pad target tokens : {n_nonpad_target} / {len(tt)}")

    if n_nonpad_target == 0:
        print("\n    !! CRITICAL: target tokens are ALL padding (id=0).")
        print("    This means the dataset is loading empty captions.")
        print("    Check your caption column name in the dataset.")

    # ── 6. Decode reference caption ───────────────────────────────────
    print("\n[6] Decoding reference captions...")
    for i in range(min(2, len(target_tok))):
        raw_ids = target_tok[i].tolist()
        decoded = tokenizer.decode(raw_ids, skip_special_tokens=True).strip()
        # Also try without filtering
        decoded_raw = tokenizer.decode(raw_ids, skip_special_tokens=False).strip()
        print(f"\n    Sample {i}:")
        print(f"      Raw ids (first 15) : {raw_ids[:15]}")
        print(f"      Decoded (skip spec): '{decoded[:120]}'")
        print(f"      Decoded (raw)      : '{decoded_raw[:120]}'")
        if not decoded:
            print(f"      !! WARNING: decoded caption is EMPTY after skip_special_tokens")

    # ── 7. Run teacher-forced forward pass ────────────────────────────
    print("\n[7] Teacher-forced forward pass...")
    with torch.no_grad():
        images_d    = images.to(device)
        input_tok_d = input_tok.to(device)
        logits = model(images_d, input_tok_d)
        print(f"    Logits shape : {logits.shape}")
        probs = F.softmax(logits[0, 0], dim=-1)
        top5  = torch.topk(probs, 5)
        print(f"    Top-5 token ids at position 0: {top5.indices.tolist()}")
        print(f"    Top-5 probs at position 0    : {[f'{p:.4f}' for p in top5.values.tolist()]}")

    # ── 8. Generate one caption (greedy) ─────────────────────────────
    print("\n[8] Greedy generation (single sample)...")
    with torch.no_grad():
        img_single = images[[0]].to(device)
        bos_id = tokenizer.bos_token_id or 1
        eos_id = tokenizer.eos_token_id or 2
        enc_out = model.encode(img_single)
        tokens  = torch.tensor([[bos_id]], device=device)

        generated_ids = [bos_id]
        for step in range(50):
            logits_step = model.decoder(tokens, enc_out)[:, -1, :]
            next_id     = logits_step.argmax(dim=-1).item()
            generated_ids.append(next_id)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == eos_id:
                break

    print(f"    Generated ids (first 20) : {generated_ids[:20]}")
    decoded_gen = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    decoded_raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    print(f"    Decoded (skip_special)   : '{decoded_gen[:150]}'")
    print(f"    Decoded (raw)            : '{decoded_raw[:150]}'")

    if not decoded_gen:
        print("\n    !! PROBLEM: Generated caption is empty after skip_special_tokens!")
        print("    Possible causes:")
        print("    a) Model only generates BOS/EOS/PAD tokens (undertrained)")
        print("    b) All generated ids ARE special tokens")
        print("    c) BOS/EOS ids mismatch between training and evaluation")
        print(f"\n    BOS id used now: {bos_id}")
        print(f"    EOS id used now: {eos_id}")
        print(f"    All generated ids: {generated_ids}")

    # ── 9. Quick BLEU sanity check ────────────────────────────────────
    print("\n[9] Quick BLEU sanity check...")
    ref = tokenizer.decode(target_tok[0].tolist(), skip_special_tokens=True).strip()
    hyp = decoded_gen

    print(f"    Reference : '{ref[:100]}'")
    print(f"    Hypothesis: '{hyp[:100]}'")

    if not ref:
        print("    !! Reference is empty -> BLEU will be 0.0")
        print("    Root cause: target tokens decoded to empty string")
        print("    Fix: check that caption_col matches the actual dataset column name")
    elif not hyp:
        print("    !! Hypothesis is empty -> BLEU will be 0.0")
        print("    Root cause: model generates only special tokens")
        print("    Fix: see [8] above for diagnosis")
    else:
        ref_toks = ref.lower().split()
        hyp_toks = hyp.lower().split()
        overlap  = set(hyp_toks) & set(ref_toks)
        print(f"    Reference tokens  : {len(ref_toks)}")
        print(f"    Hypothesis tokens : {len(hyp_toks)}")
        print(f"    Overlapping words : {len(overlap)}")
        if len(overlap) == 0:
            print("    !! No word overlap -> all n-gram metrics will be 0.0")
            print("    The model is generating words that never appear in references.")
        else:
            print(f"    Some overlap found: {list(overlap)[:10]}")
            print("    Scores should be > 0. If still 0, the issue is in run_eval.py's decode step.")

    # ── 10. Check dataset column names ───────────────────────────────
    print("\n[10] Checking dataset structure...")
    try:
        from datasets import load_dataset
        ds = load_dataset("oroikon/chart_captioning")
        split = list(ds.keys())[0]
        print(f"    Splits available : {list(ds.keys())}")
        print(f"    Columns in '{split}': {ds[split].column_names}")
        print(f"    First sample keys: {list(ds[split][0].keys())}")
        cap_col = None
        for col in ds[split].column_names:
            if "caption" in col.lower() or "text" in col.lower() or "label" in col.lower():
                cap_col = col
                break
        if cap_col:
            sample_cap = str(ds[split][0][cap_col])
            print(f"    Caption column   : '{cap_col}'")
            print(f"    Sample caption   : '{sample_cap[:150]}'")
            if not sample_cap.strip():
                print("    !! Caption is empty in the raw dataset!")
        else:
            print("    !! Could not detect caption column automatically")
    except Exception as e:
        print(f"    Could not load dataset for inspection: {e}")

    print(f"\n{'='*60}")
    print("  Diagnostics complete. Read the !! warnings above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser("ChartCaptioner Eval Diagnostics")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoints/best.pt")
    args = p.parse_args()
    diagnose(args)
