"""Figure 1 -- learning curves for three tasks, with the steps-to-match arrow.

Reads ``val_curve`` out of each task's committed ``results.json``. Plots the
**running best** validation metric, which is the quantity the protocol selects
on, so the curves are monotone and no arm is displayed mid-overfit.

Hosts are the three where validation and test agree on the ordering. STL-10 and
WikiText-2 reach AdamW's final validation score at step 1,500, but on STL-10
EvieKF's test score is lower than AdamW's -- validation on 5,000 images is noisy
-- so neither is plotted and the 6.7x figure is never quoted.

    python benchmark12/make_curve_figure.py
"""

import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "paper", "figdata")
COLUMN_IN = 5.5

HOSTS = [
    ("vision_cifar100", "CIFAR-100", False, "Validation top-1 (%)"),
    ("language_wikitext103", "WikiText-103", True, "Validation perplexity"),
    ("language_enwik8", "enwik8", True, "Validation perplexity"),
]
ARMS = [
    ("EvieKF", "#b2182b", "-", 1.5, 5),
    ("AdamW", "#4d4d4d", "-", 1.0, 3),
    ("AdaBelief", "#9aa3ad", "--", 1.0, 2),
]


def running_best(cells, arm, lower):
    """Per-seed running-best curves, aligned on the shared step grid."""
    out = []
    for key, c in cells.items():
        if c["opt"] != arm or c["diverged"]:
            continue
        pts = c.get("val_curve") or []
        cur, run = None, []
        for step, val in pts:
            cur = val if cur is None else (min(cur, val) if lower else max(cur, val))
            run.append((step, cur))
        if run:
            out.append(run)
    if not out:
        return [], [], []
    n = min(len(r) for r in out)
    steps = [out[0][i][0] for i in range(n)]
    med = [st.median(r[i][1] for r in out) for i in range(n)]
    lo = [min(r[i][1] for r in out) for i in range(n)]
    hi = [max(r[i][1] for r in out) for i in range(n)]
    return steps, med, (lo, hi)


def main():
    fig, axes = plt.subplots(1, len(HOSTS), figsize=(COLUMN_IN, 1.75))
    for ax, (task, title, lower, ylab) in zip(axes, HOSTS):
        cells = json.load(open(os.path.join(HERE, "results", task,
                                            "results.json")))["cells"]
        curves = {}
        for arm, colour, style, lw, z in ARMS:
            steps, med, band = running_best(cells, arm, lower)
            if not steps:
                continue
            curves[arm] = (steps, med)
            ax.fill_between(steps, band[0], band[1], color=colour, alpha=0.13,
                            lw=0, zorder=z - 1)
            ax.plot(steps, med, color=colour, ls=style, lw=lw, zorder=z,
                    label=arm, solid_capstyle="round")

        # The steps-to-match annotation: where does EvieKF reach AdamW's final?
        if "EvieKF" in curves and "AdamW" in curves:
            target = curves["AdamW"][1][-1]
            hit = None
            for s, x in zip(*curves["EvieKF"]):
                if (x <= target) if lower else (x >= target):
                    hit = s
                    break
            ax.axhline(target, color="#4d4d4d", lw=0.5, ls=":", zorder=1)
            if hit is not None:
                end = curves["AdamW"][0][-1]
                ax.annotate("", xy=(hit, target), xytext=(end, target),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7,
                                            color="#b2182b",
                                            shrinkA=0, shrinkB=0))
                # Offset in points, so the label clears the arrow on any axis
                # scale rather than depending on the data range.
                ax.annotate(r"%.1f$\times$ fewer steps" % (end / hit),
                            xy=((hit + end) / 2, target),
                            xytext=(0, 3), textcoords="offset points",
                            fontsize=4.9, color="#b2182b",
                            ha="center", va="bottom", zorder=8,
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.85, pad=0.7))

        ax.set_title(title, fontsize=6.5, pad=3)
        ax.set_xlabel("training step", fontsize=5.8)
        ax.set_ylabel(ylab + (r" $\downarrow$" if lower else r" $\uparrow$"),
                      fontsize=5.5)
        ax.tick_params(labelsize=5, length=1.8, pad=1.5)
        ax.grid(lw=0.25, alpha=0.3)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.5)
        if lower:
            # the first eval dominates the range on a perplexity axis
            ys = [v for a in curves for v in curves[a][1][1:]]
            ax.set_ylim(min(ys) * 0.94, max(ys) * 1.05)

    axes[0].legend(fontsize=5.2, frameon=False, handlelength=1.4,
                   labelspacing=0.25, loc="lower right")
    fig.tight_layout(pad=0.35, w_pad=0.9)
    out = os.path.join(DATA, "fig_curves.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
