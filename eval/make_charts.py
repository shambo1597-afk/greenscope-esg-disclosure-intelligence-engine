"""Slide-ready charts of the tuning results (reads eval/tuning_results.json).

Run after eval/tune.py:  python eval/make_charts.py   ->   docs/figures/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "docs" / "figures"

# Colour-blind-checked categorical palette, used in fixed order.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax, title, subtitle, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def chunk_size_chart(results):
    full = results["full_set"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for colour, model in zip(SERIES, results["grid"]["models"]):
        points = sorted((int(c.split(" / ")[1]), s["MRR@5"]) for c, s in full.items()
                        if c.startswith(model + " /") and c.endswith("/ 150"))
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=6, label=model)
        ax.annotate(f"{model}  {ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks(results["grid"]["chunk_sizes"])
    ax.set_xlim(350, 1450)
    ax.set_ylim(0.4, 0.85)
    ax.set_xlabel("Chunk size (characters)", color=INK_2)
    style(ax, "400-character chunks scored highest for all three embedding models",
          "MRR@5 on 40 questions, overlap 150 characters (higher is better)", "MRR@5")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "tuning_chunk_size.png", dpi=200)
    plt.close(fig)


def cv_chart(results):
    folds = results["folds"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.36
    xs = range(1, len(folds) + 1)
    for offset, colour, key, label in ((-width / 2, SERIES[1], "test_baseline", "Original (800/150, MiniLM)"),
                                       (width / 2, SERIES[0], "test_tuned", "Tuned on the other 4 folds")):
        values = [f[key]["MRR@5"] for f in folds]
        bars = ax.bar([x + offset for x in xs], values, width=width - 0.03, color=colour, label=label)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"Fold {x}" for x in xs])
    ax.set_ylim(0, 1.05)
    tuned = sum(f["test_tuned"]["MRR@5"] for f in folds) / len(folds)
    base = sum(f["test_baseline"]["MRR@5"] for f in folds) / len(folds)
    style(ax, "5-fold cross-validation: held-out MRR@5 per fold",
          f"Mean: tuned {tuned:.2f} vs original {base:.2f}. Each fold's 8 test questions were not used to choose the setting.",
          "Held-out MRR@5")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "cross_validation.png", dpi=200)
    plt.close(fig)


def hit_curve_chart(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = list(range(1, 11))
    for colour, key, label in ((SERIES[1], "baseline", f"Original ({results['baseline_config']})"),
                               (SERIES[0], "final", f"Tuned ({results['final_config']})")):
        ys = results["hit_curve"][key]
        ax.plot(ks, ys, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate(f"{ys[4]:.2f}", (5, ys[4]), xytext=(0, 9 if key == "final" else -14),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.axvline(5, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(4.9, 0.42, "top_k = 5 (used by the app)", fontsize=9, color=INK_2, ha="right")
    ax.set_xticks(ks)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("k (passages retrieved per company)", color=INK_2)
    style(ax, "Choosing top_k: the right page is found by k = 5, little gain after",
          "Hit@k on 40 questions: share of questions with a gold page in the top k", "Hit@k")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "hit_at_k.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((PROJECT_ROOT / "eval" / "tuning_results.json").read_text())
    chunk_size_chart(data)
    cv_chart(data)
    hit_curve_chart(data)
    print(f"Charts written to {OUT}")
