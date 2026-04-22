"""
plot_metrics.py
───────────────
Reads the outputs from run_eval.py and train.py and produces
8 publication-quality metric plots, saved individually as PNGs
and combined into a single PDF report.

Plots generated
───────────────
  1. Corpus metric bar chart       — BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
  2. Training loss curves          — train_loss + val_loss vs epoch
  3. BLEU-4 per-sample histogram   — distribution of sentence-level scores
  4. METEOR per-sample histogram   — distribution of METEOR scores
  5. ROUGE-L per-sample histogram  — distribution of ROUGE-L scores
  6. Score correlation scatter     — BLEU-4 vs ROUGE-L coloured by METEOR
  7. Caption length comparison     — hypothesis vs reference word count (box)
  8. Metric radar / spider chart   — all 6 metrics on one polar axes

Usage
─────
  # After running run_eval.py:
  python plot_metrics.py \
      --csv      eval_results/per_sample.csv \
      --log      checkpoints/training_log.csv \
      --summary  eval_results/summary.txt \
      --out_dir  eval_results/plots

  # Minimal (no training log available):
  python plot_metrics.py --csv eval_results/per_sample.csv
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Dict, Optional

# ── matplotlib setup ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
})

# Colour palette
C = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "teal":   "#64B5CD",
    "gray":   "#8C8C8C",
    "gold":   "#CCB974",
}
METRIC_COLORS = [C["blue"], C["orange"], C["green"], C["red"],
                 C["purple"], C["teal"], C["gold"]]


# ═════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_per_sample(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "bleu1":   float(row["bleu1"]),
                "bleu2":   float(row["bleu2"]),
                "bleu3":   float(row["bleu3"]),
                "bleu4":   float(row["bleu4"]),
                "rouge_l": float(row["rouge_l"]),
                "meteor":  float(row["meteor"]),
                "hyp_len": int(row["hyp_len"]),
                "ref_len": int(row["ref_len"]),
            })
    return rows


def load_training_log(log_path: str) -> Optional[Dict]:
    """Load training_log.csv → dict of lists."""
    try:
        data = {"epoch": [], "train_loss": [], "val_loss": [], "bleu4": []}
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data["epoch"].append(int(row["epoch"]))
                data["train_loss"].append(float(row["train_loss"]))
                data["val_loss"].append(float(row["val_loss"]))
                data["bleu4"].append(float(row.get("bleu4", 0)))
        return data
    except Exception as e:
        print(f"  [Warning] Could not load training log: {e}")
        return None


def load_summary_scores(summary_path: str) -> Optional[Dict]:
    """Parse corpus-level scores from summary.txt."""
    try:
        text = Path(summary_path).read_text()
        def get(pattern):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else None
        return {
            "BLEU-1":   get(r"BLEU-1\s*:\s*([\d.]+)"),
            "BLEU-2":   get(r"BLEU-2\s*:\s*([\d.]+)"),
            "BLEU-3":   get(r"BLEU-3\s*:\s*([\d.]+)"),
            "BLEU-4":   get(r"BLEU-4\s*:\s*([\d.]+)"),
            "METEOR":   get(r"METEOR\s*:\s*([\d.]+)"),
            "ROUGE-L":  get(r"ROUGE-L\s*:\s*([\d.]+)"),
            "CIDEr":    get(r"CIDEr\s*:\s*([\d.]+)"),
            "PPL":      get(r"Perplexity\s*:\s*([\d.]+)"),
        }
    except Exception as e:
        print(f"  [Warning] Could not parse summary.txt: {e}")
        return None


def avg(values): return sum(values) / max(1, len(values))


# ═════════════════════════════════════════════════════════════════════════════
# Individual plot functions
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Corpus-level bar chart ─────────────────────────────────────────────────

def plot_corpus_bar(rows: List[Dict], summary: Optional[Dict], ax: plt.Axes):
    # Compute averages from per-sample CSV (fallback if no summary)
    metrics = {
        "BLEU-1":  avg([r["bleu1"]   for r in rows]),
        "BLEU-2":  avg([r["bleu2"]   for r in rows]),
        "BLEU-3":  avg([r["bleu3"]   for r in rows]),
        "BLEU-4":  avg([r["bleu4"]   for r in rows]),
        "METEOR":  avg([r["meteor"]  for r in rows]),
        "ROUGE-L": avg([r["rouge_l"] for r in rows]),
    }
    # Override with corpus-level values from summary if available
    if summary:
        for k in metrics:
            if summary.get(k) is not None:
                metrics[k] = summary[k]

    names  = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values, color=METRIC_COLORS[:len(names)],
                  edgecolor="white", linewidth=0.8, width=0.6)

    # Value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Corpus-level Evaluation Metrics", fontweight="bold", pad=12)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")

    # Reference lines
    for ref, label, colour in [
        (0.30, "BLEU-4 good", C["green"]),
        (0.15, "BLEU-4 fair", C["orange"]),
    ]:
        ax.axhline(ref, color=colour, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(len(names) - 0.5, ref + 0.003, label,
                color=colour, fontsize=7, ha="right")


# ── 2. Training loss curves ───────────────────────────────────────────────────

def plot_loss_curves(log: Dict, ax: plt.Axes):
    epochs = log["epoch"]
    ax.plot(epochs, log["train_loss"], color=C["blue"],   linewidth=2,
            marker="o", markersize=3, label="Train loss")
    ax.plot(epochs, log["val_loss"],   color=C["orange"], linewidth=2,
            marker="s", markersize=3, label="Val loss")

    # Mark best val loss
    best_idx = int(np.argmin(log["val_loss"]))
    ax.scatter(epochs[best_idx], log["val_loss"][best_idx],
               color=C["green"], zorder=5, s=80,
               label=f"Best val loss ({log['val_loss'][best_idx]:.4f})")

    ax.set_title("Training & Validation Loss", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()


# ── 3–5. Per-metric histograms ────────────────────────────────────────────────

def plot_histogram(values: List[float], metric_name: str,
                   colour: str, ax: plt.Axes, good_thresh=None, fair_thresh=None):
    ax.hist(values, bins=25, color=colour, edgecolor="white",
            linewidth=0.5, alpha=0.85)

    mean_v = avg(values)
    median_v = sorted(values)[len(values) // 2]
    ax.axvline(mean_v,   color="black",   linestyle="--", linewidth=1.4,
               label=f"Mean  {mean_v:.3f}")
    ax.axvline(median_v, color=C["gray"], linestyle=":",  linewidth=1.4,
               label=f"Median {median_v:.3f}")

    if good_thresh:
        ax.axvline(good_thresh, color=C["green"], linestyle="-.",
                   linewidth=1.2, label=f"Good ≥{good_thresh}")
    if fair_thresh:
        ax.axvline(fair_thresh, color=C["orange"], linestyle="-.",
                   linewidth=1.2, label=f"Fair ≥{fair_thresh}")

    ax.set_title(f"{metric_name} Score Distribution  (n={len(values)})",
                 fontweight="bold", pad=12)
    ax.set_xlabel(f"{metric_name} Score")
    ax.set_ylabel("Number of samples")
    ax.legend(fontsize=8)


# ── 6. Scatter: BLEU-4 vs ROUGE-L coloured by METEOR ────────────────────────

def plot_scatter(rows: List[Dict], ax: plt.Axes):
    x      = [r["bleu4"]   for r in rows]
    y      = [r["rouge_l"] for r in rows]
    colour = [r["meteor"]  for r in rows]

    sc = ax.scatter(x, y, c=colour, cmap="viridis", alpha=0.55,
                    s=18, linewidths=0)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("METEOR score", fontsize=9)

    # Correlation coefficient
    if len(x) > 1:
        xm, ym = avg(x), avg(y)
        num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
        den = math.sqrt(sum((xi - xm)**2 for xi in x) *
                        sum((yi - ym)**2 for yi in y))
        r = num / den if den > 0 else 0
        ax.text(0.05, 0.92, f"Pearson r = {r:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title("BLEU-4 vs ROUGE-L  (colour = METEOR)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("BLEU-4")
    ax.set_ylabel("ROUGE-L")


# ── 7. Caption length box plot ────────────────────────────────────────────────

def plot_length_comparison(rows: List[Dict], ax: plt.Axes):
    hyp_lens = [r["hyp_len"] for r in rows]
    ref_lens = [r["ref_len"] for r in rows]

    bp = ax.boxplot(
        [ref_lens, hyp_lens],
        labels=["Reference\n(ground truth)", "Generated\n(hypothesis)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    colors = [C["orange"], C["blue"]]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.set_title("Caption Length: Reference vs Generated",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Word count")

    # Annotation: avg lengths
    for i, (vals, label) in enumerate([(ref_lens, "ref"), (hyp_lens, "hyp")], 1):
        ax.text(i, max(vals) * 1.02,
                f"avg {avg(vals):.1f}",
                ha="center", fontsize=8, color="black")


# ── 8. Radar / spider chart ───────────────────────────────────────────────────

def plot_radar(rows: List[Dict], summary: Optional[Dict], ax: plt.Axes):
    # 6 metrics, normalised to [0,1] using "good" thresholds as 100%
    metrics = {
        "BLEU-1":  (avg([r["bleu1"]   for r in rows]), 0.50),
        "BLEU-4":  (avg([r["bleu4"]   for r in rows]), 0.30),
        "METEOR":  (avg([r["meteor"]  for r in rows]), 0.25),
        "ROUGE-L": (avg([r["rouge_l"] for r in rows]), 0.40),
    }
    if summary:
        if summary.get("CIDEr"):
            metrics["CIDEr"] = (summary["CIDEr"] / 10.0, 0.10)   # scale ÷10

    labels = list(metrics.keys())
    N = len(labels)
    raw    = [metrics[k][0] for k in labels]
    thresh = [metrics[k][1] for k in labels]
    norm   = [min(r / t, 1.3) for r, t in zip(raw, thresh)]   # cap at 130%

    angles = [2 * math.pi * i / N for i in range(N)] + [0]
    norm   = norm + [norm[0]]
    thresh_norm = [1.0] * (N + 1)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, norm,        color=C["blue"],  linewidth=2)
    ax.fill(angles, norm,        color=C["blue"],  alpha=0.25)
    ax.plot(angles, thresh_norm, color=C["green"], linewidth=1.2,
            linestyle="--", label='"Good" threshold')

    ax.set_thetagrids(
        [math.degrees(a) for a in angles[:-1]],
        labels,
        fontsize=9,
    )
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_yticklabels(["25%", "50%", "75%", "100%", "125%"], fontsize=7)
    ax.set_title("Metric Radar\n(% of 'good' threshold)",
                 fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1))

    # Label actual values on the spokes
    for angle, label, val in zip(angles[:-1], labels, raw):
        ax.text(angle, 1.38, f"{val:.3f}",
                ha="center", va="center", fontsize=7.5,
                color=C["blue"], fontweight="bold")


# ── 9. BLEU-4 vs training BLEU-4 over epochs (if available) ──────────────────

def plot_bleu_over_epochs(log: Dict, ax: plt.Axes):
    epochs = log["epoch"]
    bleu4  = log["bleu4"]

    # Only plot non-zero entries (BLEU is computed every N epochs)
    ep_nonzero = [e for e, b in zip(epochs, bleu4) if b > 0]
    bl_nonzero = [b for b in bleu4 if b > 0]

    if not ep_nonzero:
        ax.text(0.5, 0.5, "No BLEU-4 checkpoints found\nin training log",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=C["gray"])
        ax.set_title("BLEU-4 over Training", fontweight="bold", pad=12)
        return

    ax.plot(ep_nonzero, bl_nonzero, color=C["purple"], linewidth=2.2,
            marker="D", markersize=6, label="Val BLEU-4")
    ax.fill_between(ep_nonzero, bl_nonzero, alpha=0.15, color=C["purple"])

    best_idx = int(np.argmax(bl_nonzero))
    ax.scatter(ep_nonzero[best_idx], bl_nonzero[best_idx],
               color=C["green"], zorder=5, s=100,
               label=f"Best ({bl_nonzero[best_idx]:.4f})")

    ax.set_title("Validation BLEU-4 over Training", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU-4")
    ax.legend()


# ═════════════════════════════════════════════════════════════════════════════
# Layout & save
# ═════════════════════════════════════════════════════════════════════════════

def build_all_plots(rows, log, summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # ── Figure 1: Corpus metric bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_corpus_bar(rows, summary, ax)
    fig.tight_layout()
    p = out_dir / "01_corpus_metrics.png"
    fig.savefig(p); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    # ── Figure 2 & 9: Loss curves + BLEU over epochs (side by side) ──────────
    if log:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_loss_curves(log, axes[0])
        plot_bleu_over_epochs(log, axes[1])
        fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        p = out_dir / "02_training_curves.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)
        print(f"  Saved → {p.name}")
    else:
        print("  Skipping training curves (no log file)")

    # ── Figure 3: BLEU-4 histogram ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_histogram([r["bleu4"] for r in rows], "BLEU-4",
                   C["blue"], ax, good_thresh=0.30, fair_thresh=0.15)
    fig.tight_layout()
    p = out_dir / "03_bleu4_histogram.png"
    fig.savefig(p); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    # ── Figure 4: METEOR histogram ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_histogram([r["meteor"] for r in rows], "METEOR",
                   C["purple"], ax, good_thresh=0.25, fair_thresh=0.15)
    fig.tight_layout()
    p = out_dir / "04_meteor_histogram.png"
    fig.savefig(p); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    # ── Figure 5: ROUGE-L histogram ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_histogram([r["rouge_l"] for r in rows], "ROUGE-L",
                   C["teal"], ax, good_thresh=0.40, fair_thresh=0.25)
    fig.tight_layout()
    p = out_dir / "05_rougel_histogram.png"
    fig.savefig(p); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    # ── Figure 6: Scatter BLEU-4 vs ROUGE-L ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_scatter(rows, ax)
    fig.tight_layout()
    p = out_dir / "06_scatter_bleu_rougel.png"
    fig.savefig(p); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    # ── Figure 7: Caption length box plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_length_comparison(rows, ax)
    fig.tight_layout()
    p = out_dir / "07_caption_length.png"
    fig.savefig(p); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    # ── Figure 8: Radar chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    plot_radar(rows, summary, ax)
    fig.tight_layout()
    p = out_dir / "08_radar_chart.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)
    print(f"  Saved → {p.name}")

    return saved


def build_combined_pdf(saved_pngs: list, out_dir: Path):
    """Combine all PNGs into a single multi-page PDF."""
    pdf_path = out_dir / "all_metrics_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for png in saved_pngs:
            img = plt.imread(str(png))
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "ChartCaptioner — Evaluation Metrics Report"
        d["Subject"] = "BLEU, METEOR, ROUGE-L, CIDEr, Perplexity"

    print(f"\n  Combined PDF → {pdf_path}")
    return pdf_path


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("ChartCaptioner — Plot Evaluation Metrics")
    p.add_argument("--csv",     required=True,
                   help="Path to eval_results/per_sample.csv")
    p.add_argument("--log",     default=None,
                   help="Path to checkpoints/training_log.csv  (optional)")
    p.add_argument("--summary", default=None,
                   help="Path to eval_results/summary.txt  (optional, for corpus CIDEr)")
    p.add_argument("--out_dir", default="eval_results/plots",
                   help="Directory to save plots  (default: eval_results/plots)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)

    print(f"\n{'='*55}")
    print("  ChartCaptioner — Metric Plots")
    print(f"{'='*55}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading per-sample CSV: {args.csv}")
    rows = load_per_sample(args.csv)
    print(f"  {len(rows)} samples loaded")

    log     = load_training_log(args.log)     if args.log     else None
    summary = load_summary_scores(args.summary) if args.summary else None

    if log:
        print(f"  Training log: {len(log['epoch'])} epochs")
    if summary:
        print(f"  Summary scores loaded")

    # ── Build plots ────────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {out_dir}/")
    saved = build_all_plots(rows, log, summary, out_dir)

    # ── Combine into PDF ───────────────────────────────────────────────────────
    build_combined_pdf(saved, out_dir)

    print(f"\n{'='*55}")
    print(f"  {len(saved)} plots saved to {out_dir}/")
    print(f"  Open all_metrics_report.pdf for the full report")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
