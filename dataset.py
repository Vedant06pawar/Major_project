"""
Dataset utilities for oroikon/chart_captioning (HuggingFace).

Dataset structure (inferred from HF page):
  - image : PIL Image  (chart / graph)
  - text  : str        (caption / description)

We wrap it in a PyTorch Dataset with image augmentation for training
and deterministic preprocessing for validation/test.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# HuggingFace-backed Dataset
# ---------------------------------------------------------------------------

class ChartCaptionDataset(Dataset):
    """
    Wraps the oroikon/chart_captioning HuggingFace dataset.

    Parameters
    ----------
    hf_split    : HuggingFace dataset split object (already loaded)
    tokenizer   : HuggingFace tokenizer
    transform   : torchvision transform for images
    max_length  : maximum caption token length (including BOS/EOS)
    image_col   : column name for images  (default "image")
    caption_col : column name for captions (default "text")
    """

    def __init__(
        self,
        hf_split,
        tokenizer,
        transform,
        max_length:  int = 128,
        image_col:   str = "image",
        caption_col: str = "text",
    ):
        self.data        = hf_split
        self.tokenizer   = tokenizer
        self.transform   = transform
        self.max_length  = max_length
        self.image_col   = image_col
        self.caption_col = caption_col

        # BOS / EOS ids
        self.bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id or 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # --- image ---
        img = row[self.image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        img_tensor = self.transform(img)

        # --- caption ---
        caption = str(row[self.caption_col])
        enc = self.tokenizer(
            caption,
            max_length=self.max_length - 2,   # leave room for BOS/EOS
            truncation=True,
            add_special_tokens=False,
        )
        token_ids = [self.bos_id] + enc["input_ids"] + [self.eos_id]

        # Pad to max_length
        pad_len   = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * pad_len

        tokens      = torch.tensor(token_ids, dtype=torch.long)
        input_tok   = tokens[:-1]    # (T-1,)  — decoder input
        target_tok  = tokens[1:]     # (T-1,)  — decoder target (shifted)

        return img_tensor, input_tok, target_tok


# ---------------------------------------------------------------------------
# Collate & DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_name: str   = "oroikon/chart_captioning",
    tokenizer           = None,
    image_size:  int    = 224,
    max_length:  int    = 128,
    batch_size:  int    = 16,
    num_workers: int    = 2,
    val_split:   float  = 0.1,
    seed:        int    = 42,
    image_col:   str    = "image",
    caption_col: str    = "text",
):
    """
    Downloads oroikon/chart_captioning and returns (train_loader, val_loader).
    If the dataset has a predefined train/test split those are used;
    otherwise we randomly split the 'train' split.
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    # Identify splits
    if "train" in ds and "test" in ds:
        train_hf = ds["train"]
        val_hf   = ds["test"]
    elif "train" in ds and "validation" in ds:
        train_hf = ds["train"]
        val_hf   = ds["validation"]
    else:
        # Fall back: split 'train' ourselves
        split = ds["train"].train_test_split(test_size=val_split, seed=seed)
        train_hf = split["train"]
        val_hf   = split["test"]

    print(f"  Train samples: {len(train_hf)}")
    print(f"  Val   samples: {len(val_hf)}")

    # Peek at columns
    print(f"  Columns: {train_hf.column_names}")

    # Auto-detect column names if different
    cols = train_hf.column_names
    if image_col not in cols:
        # try to find image column
        for c in cols:
            if "image" in c.lower() or "img" in c.lower():
                image_col = c
                break
    if caption_col not in cols:
        for c in cols:
            if "caption" in c.lower() or "text" in c.lower() or "label" in c.lower():
                caption_col = c
                break

    print(f"  Using image_col='{image_col}', caption_col='{caption_col}'")

    train_ds = ChartCaptionDataset(
        train_hf, tokenizer,
        transform=get_train_transform(image_size),
        max_length=max_length,
        image_col=image_col,
        caption_col=caption_col,
    )
    val_ds = ChartCaptionDataset(
        val_hf, tokenizer,
        transform=get_val_transform(image_size),
        max_length=max_length,
        image_col=image_col,
        caption_col=caption_col,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
