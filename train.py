"""
Training script for ChartCaptioner.

Features:
  - Mixed-precision (AMP) training
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Checkpoint saving (best val loss)
  - BLEU-4 evaluation on validation set
  - Detailed logging to console + CSV
"""

import os
import csv
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer

from model   import ChartCaptioner
from dataset import build_dataloaders
from evaluate import compute_bleu


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / optimizer.defaults["lr"], cosine)

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, log_every=50):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore padding

    total_loss = 0.0
    n_batches  = len(loader)
    t0 = time.time()

    for step, (images, input_tok, target_tok) in enumerate(loader):
        images     = images.to(device, non_blocking=True)
        input_tok  = input_tok.to(device, non_blocking=True)
        target_tok = target_tok.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images, input_tok)              # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V),
                             target_tok.reshape(B * T))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg     = total_loss / (step + 1)
            lr      = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} [{step+1}/{n_batches}] "
                  f"loss={avg:.4f}  lr={lr:.2e}  {elapsed:.0f}s")

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0

    for images, input_tok, target_tok in loader:
        images     = images.to(device)
        input_tok  = input_tok.to(device)
        target_tok = target_tok.to(device)

        logits = model(images, input_tok)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V),
                         target_tok.reshape(B * T))
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("ChartCaptioner Training")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--embed_dim",    type=int,   default=256)
    p.add_argument("--enc_depth",    type=int,   default=6)
    p.add_argument("--dec_depth",    type=int,   default=6)
    p.add_argument("--enc_heads",    type=int,   default=8)
    p.add_argument("--dec_heads",    type=int,   default=8)
    p.add_argument("--image_size",   type=int,   default=224)
    p.add_argument("--max_length",   type=int,   default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--save_dir",     type=str,   default="checkpoints")
    p.add_argument("--bleu_every",   type=int,   default=5,
                   help="Compute BLEU score every N epochs (expensive)")
    p.add_argument("--tokenizer",    type=str,   default="gpt2",
                   help="HuggingFace tokenizer name")
    p.add_argument("--dataset",      type=str,   default="oroikon/chart_captioning")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- tokenizer ----
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # ---- data ----
    train_loader, val_loader = build_dataloaders(
        dataset_name = args.dataset,
        tokenizer    = tokenizer,
        image_size   = args.image_size,
        max_length   = args.max_length,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )

    # ---- model ----
    model = ChartCaptioner(
        vocab_size  = vocab_size,
        embed_dim   = args.embed_dim,
        enc_heads   = args.enc_heads,
        enc_depth   = args.enc_depth,
        dec_heads   = args.dec_heads,
        dec_depth   = args.dec_depth,
        image_size  = args.image_size,
        max_seq_len = args.max_length,
        dropout     = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-2, betas=(0.9, 0.95))
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(args.warmup_ratio * total_steps)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = GradScaler()

    # ---- optional resume ----
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ---- logging ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "bleu4", "lr"])

    # ---- training loop ----
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        val_loss = validate(model, val_loader, device)

        # BLEU evaluation (every N epochs)
        bleu4 = 0.0
        if epoch % args.bleu_every == 0:
            bleu4 = compute_bleu(model, val_loader, tokenizer, device,
                                 max_samples=200)

        elapsed = time.time() - epoch_t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"bleu4={bleu4:.4f}  lr={lr_now:.2e}  time={elapsed:.0f}s")

        # log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}",
                             f"{val_loss:.6f}", f"{bleu4:.4f}", f"{lr_now:.2e}"])

        # save checkpoint
        ckpt = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args":          vars(args),
            "vocab_size":    vocab_size,
        }
        torch.save(ckpt, save_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
