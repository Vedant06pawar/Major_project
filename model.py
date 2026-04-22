"""
Chart Captioning Model: CNN + Vision Transformer (ViT) Encoder + Transformer Decoder
Architecture:
  1. CNN Feature Extractor (ResNet-style backbone) → local patch features
  2. Vision Transformer Encoder → global context via self-attention
  3. Transformer Decoder → autoregressive caption generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  CNN Backbone (lightweight ResNet-style)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.gelu(out)


class CNNBackbone(nn.Module):
    """
    Produces a grid of feature vectors from an input image.
    Input:  (B, 3, H, W)
    Output: (B, embed_dim, H/32, W/32)
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),   # /2
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),                    # /4
        )
        self.layer1 = self._make_layer(64,  128, 2, stride=2)       # /8
        self.layer2 = self._make_layer(128, 256, 2, stride=2)       # /16
        self.layer3 = self._make_layer(256, embed_dim, 2, stride=2) # /32

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x   # (B, embed_dim, H/32, W/32)


# ---------------------------------------------------------------------------
# 2.  Positional Encoding helpers
# ---------------------------------------------------------------------------

class SinusoidalPosEmb1D(nn.Module):
    """1-D sinusoidal positional embedding (for sequence length up to max_len)."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class LearnedPosEmb2D(nn.Module):
    """Learnable 2-D positional embedding for CNN patch grid."""
    def __init__(self, embed_dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.row_emb = nn.Embedding(grid_h, embed_dim // 2)
        self.col_emb = nn.Embedding(grid_w, embed_dim // 2)

    def forward(self, h, w, device):
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        row_e = self.row_emb(rows).unsqueeze(1).expand(-1, w, -1)  # (h, w, D/2)
        col_e = self.col_emb(cols).unsqueeze(0).expand(h, -1, -1)  # (h, w, D/2)
        return torch.cat([row_e, col_e], dim=-1).view(h * w, -1)   # (h*w, D)


# ---------------------------------------------------------------------------
# 3.  Vision Transformer Encoder
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Takes CNN patch features and applies transformer self-attention.
    Input:  (B, N_patches, embed_dim)
    Output: (B, N_patches, embed_dim)
    """
    def __init__(self, embed_dim: int, n_heads: int, depth: int, dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)         # prepend CLS token
        x   = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)   # (B, 1+N, D)


# ---------------------------------------------------------------------------
# 4.  Transformer Decoder
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out     = nn.Linear(embed_dim, embed_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, q, kv, key_padding_mask=None):
        B, Tq, D = q.shape
        _, Tk, _  = kv.shape
        H, Hd    = self.n_heads, self.head_dim

        Q  = self.q_proj(q).reshape(B, Tq, H, Hd).transpose(1, 2)
        KV = self.kv_proj(kv).reshape(B, Tk, 2, H, Hd).permute(2, 0, 3, 1, 4)
        K, V = KV.unbind(0)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ V).transpose(1, 2).reshape(B, Tq, D)
        return self.out(out)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1    = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, n_heads, dropout)
        self.norm3    = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, enc_out, causal_mask=None):
        x = x + self.self_attn(self.norm1(x), mask=causal_mask)
        x = x + self.cross_attn(self.norm2(x), enc_out)
        x = x + self.mlp(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int,
                 depth: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb  = SinusoidalPosEmb1D(embed_dim, max_len)
        self.blocks   = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(embed_dim)
        self.head     = nn.Linear(embed_dim, vocab_size, bias=False)
        self.drop     = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.tok_emb.weight, std=0.02)

    def _causal_mask(self, T, device):
        mask = torch.tril(torch.ones(T, T, device=device))
        return mask  # (T, T)

    def forward(self, tokens, enc_out):
        B, T = tokens.shape
        x = self.drop(self.pos_emb(self.tok_emb(tokens)))
        causal = self._causal_mask(T, tokens.device)
        for blk in self.blocks:
            x = blk(x, enc_out, causal_mask=causal)
        x = self.norm(x)
        return self.head(x)   # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# 5.  Full Model
# ---------------------------------------------------------------------------

class ChartCaptioner(nn.Module):
    """
    End-to-end chart captioning model.
    CNN → flatten patches → learnable 2D pos emb → ViT Encoder → Transformer Decoder
    """
    def __init__(
        self,
        vocab_size:  int   = 8000,
        embed_dim:   int   = 256,
        enc_heads:   int   = 8,
        enc_depth:   int   = 6,
        dec_heads:   int   = 8,
        dec_depth:   int   = 6,
        image_size:  int   = 224,
        max_seq_len: int   = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        grid = image_size // 32           # spatial grid after CNN

        self.cnn        = CNNBackbone(embed_dim)
        self.pos_emb_2d = LearnedPosEmb2D(embed_dim, grid, grid)
        self.encoder    = ViTEncoder(embed_dim, enc_heads, enc_depth, dropout)
        self.decoder    = TransformerDecoder(vocab_size, embed_dim, dec_heads,
                                             dec_depth, max_seq_len, dropout)

        self.grid = grid
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, images):
        """images: (B, 3, H, W) → encoder memory (B, 1+N, D)"""
        feats = self.cnn(images)                                # (B, D, h, w)
        B, D, h, w = feats.shape
        patches = feats.flatten(2).transpose(1, 2)             # (B, h*w, D)
        pos = self.pos_emb_2d(h, w, images.device)             # (h*w, D)
        patches = patches + pos.unsqueeze(0)
        return self.encoder(patches)                           # (B, 1+N, D)

    def forward(self, images, captions):
        """
        images:   (B, 3, H, W)
        captions: (B, T)  — input tokens (teacher-forced, shifted right)
        returns logits (B, T, vocab_size)
        """
        enc_out = self.encode(images)
        logits  = self.decoder(captions, enc_out)
        return logits

    @torch.no_grad()
    def generate(self, image, tokenizer, max_new_tokens=128, temperature=1.0, top_k=50):
        """
        Greedy / top-k sampling for a single image.
        image: (1, 3, H, W)

        GPT-2 fix: BOS == EOS == PAD == 50256. Starting generation with
        just [BOS] means the first predicted token can equal EOS and stop
        immediately. We use a short forced prefix ("This") so the model
        generates real tokens before any EOS check begins.

        Sequence-length fix: truncate context to max_seq_len-1 before each
        decoder call so positional embeddings never overflow.
        """
        self.eval()
        device   = next(self.parameters()).device
        image    = image.to(device)
        max_pos  = self.decoder.pos_emb.pe.size(1)  # max positional emb length

        enc_out = self.encode(image)

        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2

        prefix_ids = tokenizer.encode("This", add_special_tokens=False)
        tokens     = torch.tensor([prefix_ids], device=device)
        min_new_tokens = 8

        for step in range(max_new_tokens):
            # Truncate to the last (max_pos-1) tokens so pos emb never overflows
            context = tokens[:, -(max_pos - 1):]
            logits  = self.decoder(context, enc_out)[:, -1, :]
            logits  = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_t], dim=1)

            if step >= min_new_tokens and next_t.item() == eos_id:
                break

            # Hard stop at sequence length limit
            if tokens.size(1) >= max_pos:
                break

        ids = tokens[0].tolist()
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]
        return tokenizer.decode(ids).strip()

    @torch.no_grad()
    def beam_search(self, image, tokenizer, max_new_tokens=300,
                    beam_size=5, length_penalty=1.2):
        """
        Beam search decoding for a single image.

        GPT-2 fix: BOS == EOS == PAD == 50256 — seed beams with "This"
        so no beam terminates immediately on the EOS token.

        Sequence-length fix: each decoder call uses only the last
        (max_seq_len - 1) tokens so positional embeddings never overflow.
        """
        self.eval()
        device  = next(self.parameters()).device
        image   = image.to(device)
        max_pos = self.decoder.pos_emb.pe.size(1)  # positional emb limit

        enc_out = self.encode(image)

        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2

        prefix_ids = tokenizer.encode("This", add_special_tokens=False)
        min_new    = 8

        beams     = [(0.0, 0.0, prefix_ids[:])]
        completed = []

        for step in range(max_new_tokens):
            if not beams:
                break

            candidates = []
            for norm_score, raw_score, tokens in beams:
                # Truncate to last (max_pos-1) tokens before feeding decoder
                context    = tokens[-(max_pos - 1):]
                tok_tensor = torch.tensor([context], device=device)
                logits     = self.decoder(tok_tensor, enc_out)[:, -1, :]
                log_probs  = F.log_softmax(logits, dim=-1).squeeze(0)

                top_lp, top_ids = torch.topk(log_probs, beam_size)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    new_raw    = raw_score + lp
                    new_tokens = tokens + [tid]
                    new_norm   = new_raw / (len(new_tokens) ** length_penalty)
                    candidates.append((new_norm, new_raw, new_tokens))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            # Allow EOS only after min_new steps
            if step >= min_new:
                done  = [b for b in beams if b[2][-1] == eos_id]
                beams = [b for b in beams if b[2][-1] != eos_id]
                completed.extend(done)

            if not beams:
                break

            # Hard stop if every beam has hit the length limit
            if all(len(b[2]) >= max_pos for b in beams):
                completed.extend(beams)
                break

        completed.extend(beams)

        if not completed:
            return ""

        best_tokens = sorted(completed, key=lambda x: x[0], reverse=True)[0][2]
        if best_tokens and best_tokens[-1] == eos_id:
            best_tokens = best_tokens[:-1]
        return tokenizer.decode(best_tokens).strip()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ChartCaptioner(vocab_size=8000, embed_dim=256, image_size=224)
    imgs  = torch.randn(2, 3, 224, 224)
    caps  = torch.randint(0, 8000, (2, 32))
    out   = model(imgs, caps)
    print(f"Output logits: {out.shape}")   # (2, 32, 8000)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.1f}M")
