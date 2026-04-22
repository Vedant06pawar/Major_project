"""
captioner.py
============
Thin wrapper around the user's ChartCaptioner model (inference.py / model.py).

Usage
-----
from captioner import CaptionerService

svc = CaptionerService("checkpoints/best.pt")
caption = svc.caption("path/to/image.png")
"""

import logging
import os
import sys
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# Ensure the chart_captioner package is importable
_PKG_DIR = Path(__file__).parent / "chart_captioner"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))


class CaptionerService:
    """
    Loads a trained ChartCaptioner checkpoint and exposes a `caption(image_path)`
    method that returns a string description of the image.

    The underlying model (CNN + ViT + Transformer decoder) and tokenizer are
    loaded once at construction time; inference is thread-safe because we hold
    a lock around the forward pass.
    """

    def __init__(
        self,
        checkpoint_path: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
    ):
        import threading
        from inference import load_model  # from chart_captioner/inference.py

        self._lock = threading.Lock()
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_k          = top_k

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        log.info("CaptionerService: using device=%s", self.device)

        self.model, self.tokenizer = load_model(checkpoint_path, self.device)
        self.model.eval()
        log.info("CaptionerService: model loaded from %s", checkpoint_path)

    def caption(self, image_path: str) -> str:
        """
        Generate a textual description for the image at `image_path`.
        Returns a string.
        """
        from inference import caption_image  # from chart_captioner/inference.py

        with self._lock:
            try:
                result = caption_image(
                    self.model,
                    self.tokenizer,
                    image_path=image_path,
                    device=self.device,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                )
                return result.strip() if result else "Unable to generate a caption."
            except Exception as exc:
                log.error("caption() failed for %s: %s", image_path, exc, exc_info=True)
                return "An image is present but could not be described."
