"""
Inference script for ChartCaptioner.

Usage:
    python inference.py --image path/to/chart.png --checkpoint checkpoints/best.pt

Or from Python:
    from inference import load_model, caption_image
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from model import ChartCaptioner


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]

    tokenizer = AutoTokenizer.from_pretrained(args.get("tokenizer", "gpt2"))
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = ckpt.get("vocab_size", len(tokenizer))

    model = ChartCaptioner(
        vocab_size  = vocab_size,
        embed_dim   = args.get("embed_dim",  256),
        enc_heads   = args.get("enc_heads",  8),
        enc_depth   = args.get("enc_depth",  6),
        dec_heads   = args.get("dec_heads",  8),
        dec_depth   = args.get("dec_depth",  6),
        image_size  = args.get("image_size", 224),
        max_seq_len = args.get("max_length", 128),
        dropout     = 0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Caption a single image file
# ---------------------------------------------------------------------------

def preprocess(image_path: str, image_size: int = 224) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # (1, 3, H, W)


def caption_image(
    model,
    tokenizer,
    image_path:     str,
    device:         torch.device,
    max_new_tokens: int   = 150,
    temperature:    float = 0.7,
    top_k:          int   = 50,
) -> str:
    image_size = model.image_size
    img = preprocess(image_path, image_size).to(device)
    caption = model.generate(img, tokenizer,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             top_k=top_k)
    return caption


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("ChartCaptioner Inference")
    p.add_argument("--image",      required=True, help="Path to chart/graph image")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--max_tokens", type=int,   default=150)
    p.add_argument("--temperature",type=float, default=0.7)
    p.add_argument("--top_k",      type=int,   default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(args.checkpoint, device)

    caption = caption_image(
        model, tokenizer, args.image, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n" + "="*60)
    print("CHART CAPTION")
    print("="*60)
    print(caption)
    print("="*60)


if __name__ == "__main__":
    main()
