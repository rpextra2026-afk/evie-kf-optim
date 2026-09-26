"""Build Figures 5 and 6 from the exported CSVs.

Run ``export_figure_data.py`` first. This script reads only
``results/paper/figdata/*.csv`` and writes PDFs beside them, so it can be dropped
into Prism next to those three CSVs and run unchanged.

**No number is hardcoded here.** Re-export after a re-run and the figures follow;
that is the whole point, and it is what the existing ``make_figures.py`` does not
do.

    python benchmark12/export_figure_data.py
    python benchmark12/make_results_figures.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "paper", "figdata")

# Column width of the ICLR template, in inches.
COLUMN_IN = 5.5

# Fig 6 colour scale, in "percent worse than this panel's own best scored
# point". The 14 surfaces span very different ranges -- a median point is 1.1%
# off the best on the finance panels and 171% off on WikiText-103 -- so the
# scale is logarithmic. A linear scale makes every vision and finance panel
# look uniformly optimal and every language panel look uniformly bad.
PCT_LO, PCT_HI = 0.3, 300.0

# The frozen point is outlined rather than starred: an outline does not hide
# the cell it marks, and it stays visible at both ends of the colour ramp.
PICK_EDGE = "#d7191c"

# Fig 6 colour map. Sequential, perceptually uniform, dark = best, so it
# survives greyscale printing. cividis is built specifically for colour-vision
# deficiency and is the most restrained of the perceptually uniform maps, which
# matches the palettes used in the optimizer literature. Reversed, so the bright
# end marks the best-scoring cells.
CMAP = "cividis_r"

HOST_LABEL = {
    "vision_cifar10": "CIFAR-10",
    "vision_cifar100": "CIFAR-100",
    "vision_svhn": "SVHN",
    "vision_stl10": "STL-10",
    "language_wikitext2": "WikiText-2",
    "language_wikitext103": "WikiText-103",
    "language_ptb": "Penn Treebank",
    "language_enwik8": "enwik8",
    "finance_tech": "Tech",
    "finance_finance": "Finance",
    "finance_healthcare": "Healthcare",
    "finance_consumer_industrials": "Consumer/Ind.",
    "vision_tinyimagenet": "Tiny-ImageNet",
    "language_openwebtext": "OpenWebText",
}

# Rows of the Fig 6 grid: the benchmark's own three domains, four tasks each.
FIG6_ROWS = [
    ("Vision", ["vision_cifar10", "vision_cifar100", "vision_svhn", "vision_stl10"]),
    ("Language", ["language_wikitext2", "language_wikitext103", "language_ptb",
                  "language_enwik8"]),
    ("Finance", ["finance_tech", "finance_finance", "finance_healthcare",
                 "finance_consumer_industrials"]),
]
ARM_ORDER = ["EvieKF", "EvieKFu", "EvieDiagAbs", "EvieDiag"]
# Short forms; the caption expands them. Horizontal labels under narrow panels
# need to be short, and rotated labels in a three-panel strip are unreadable.
ARM_LABEL = {
    "EvieKF": "KF",
    "EvieKFu": "KFu",
    "EvieDiagAbs": "DiagAbs",
    "EvieDiag": "Diag",
}
# Optimizer papers keep line plots to one or two saturated hues on white. The
# arms here are x positions, not separate series, so a per-arm palette would
# imply a distinction that is not there: seeds are one neutral, median is one
# accent.
METRIC_LABEL = {
    "test_mse": (r"Test MSE ($\times 10^{-4}$)", 1e4),
    "test_perplexity": ("Test perplexity", 1.0),
    "test_top1_acc": ("Test top-1 (%)", 1.0),
}
SEED_LINE = "#9aa3ad"
MEDIAN_LINE = "#b2182b"


def _sci(v):
    """Compact axis label: 1e-3 rather than 0.001, 27k rather than 27000."""
    if v >= 1000:
        return "%gk" % (v / 1000.0)
    if v >= 1:
        return "%g" % v
    return "%.0e" % v


def _rows(name):
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def figure5():
    """Paired per-seed slopegraph: one line per seed across the ablation arms.

    A slopegraph rather than a scatter, because the contrast is *paired* -- the
    claim is "10 of 10 seeds", and only a line joining each seed's own points
    shows that. A dot cloud shows the marginal spread and hides the pairing,
    which is the one thing this figure exists to display.
    """
    rows = _rows("fig5_ablation_per_seed.csv")
    contrasts = {(r["host"], r["baseline_arm"]): r
                 for r in _rows("fig5_contrasts.csv")}

    by_host = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for r in rows:
        scale = METRIC_LABEL.get(r["metric"], (r["metric"], 1.0))[1]
        by_host[r["host"]][r["arm"]][int(r["seed"])] = float(r["value"]) * scale
        meta[r["host"]] = (r["metric"], r["lower_is_better"] == "1")

    hosts = [h for h in ["finance_tech", "language_wikitext2", "vision_cifar10"]
             if h in by_host]
    fig, axes = plt.subplots(1, len(hosts), figsize=(COLUMN_IN, 1.55))
    if len(hosts) == 1:
        axes = [axes]

    for ax, host in zip(axes, hosts):
        arms = [a for a in ARM_ORDER if a in by_host[host]]
        seeds = sorted(set.intersection(*[set(by_host[host][a]) for a in arms]))
        x = list(range(len(arms)))

        for seed in seeds:
            ax.plot(x, [by_host[host][a][seed] for a in arms],
                    lw=0.45, color=SEED_LINE, alpha=0.8, zorder=2,
                    solid_capstyle="round")
        med = [sorted(by_host[host][a][s] for s in seeds)[len(seeds) // 2]
               for a in arms]
        ax.plot(x, med, lw=1.4, color=MEDIAN_LINE, zorder=4,
                marker="o", ms=2.6, mfc="white", mew=0.9)

        metric, lower = meta[host]
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=5.5)
        ax.set_xlim(-0.3, len(arms) - 0.7)
        ax.set_title(HOST_LABEL.get(host, host), fontsize=6.5, pad=9)
        label = METRIC_LABEL.get(metric, (metric, 1.0))[0]
        ax.set_ylabel(label + (r" $\downarrow$" if lower else r" $\uparrow$"),
                      fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5, length=1.6, pad=1.5)
        ax.grid(axis="y", lw=0.25, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.5)

        key = (host, "EvieDiagAbs" if (host, "EvieDiagAbs") in contrasts
               else "EvieDiag")
        c = contrasts.get(key)
        if c is not None:
            ax.text(0.5, 1.01, "%+.1f%%, %s/%s seeds"
                    % (100 * float(c["task_level_delta"]),
                       c["eviekf_wins"], c["n_seeds"]),
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=5, color="#444444")

    fig.tight_layout(pad=0.35, w_pad=0.9)
    out = os.path.join(DATA, "fig5_ablation_ladder.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (%d hosts)" % (os.path.relpath(out, HERE), len(hosts)))


def figure6():
    """The 12 benchmark tasks' scored (lr, gamma) surfaces, three domains by row."""
    rows = _rows("fig6_gamma_surface.csv")
    panels = defaultdict(list)
    for r in rows:
        panels[r["panel"]].append(r)

    ncol = max(len(ts) for _, ts in FIG6_ROWS)
    fig, axes = plt.subplots(len(FIG6_ROWS), ncol,
                             figsize=(COLUMN_IN, 0.78 * len(FIG6_ROWS) + 0.52))
    norm = LogNorm(vmin=PCT_LO, vmax=PCT_HI)
    cmap = plt.get_cmap(CMAP).copy()
    # Boundary extensions make some grids ragged. An unscored (lr, gamma) pair
    # is drawn white, which is off the cividis ramp entirely -- a mid grey would
    # collide with the map's own mid-tones and read as a measured value.
    cmap.set_bad("white")
    im = None
    drawn = 0

    for ri, (domain, tasks) in enumerate(FIG6_ROWS):
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci >= len(tasks) or tasks[ci] not in panels:
                ax.axis("off")
                continue
            panel = tasks[ci]
            rs = panels[panel]
            lrs = sorted({float(r["lr"]) for r in rs})
            gammas = sorted({float(r["gamma"]) for r in rs})
            lower = rs[0]["lower_is_better"] == "1"

            grid = [[float("nan")] * len(lrs) for _ in gammas]
            pick = None
            for r in rs:
                gi, li = gammas.index(float(r["gamma"])), lrs.index(float(r["lr"]))
                grid[gi][li] = float(r["search_score"])
                if r["is_frozen"] == "1":
                    pick = (li, gi)

            finite = [v for row in grid for v in row if v == v]
            ref = min(finite) if lower else max(finite)
            data = [[min(max(100.0 * ((v - ref) if lower else (ref - v)) / abs(ref),
                            PCT_LO), PCT_HI) if v == v else float("nan")
                     for v in row] for row in grid]

            im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap,
                           norm=norm)
            drawn += 1
            if pick is not None:
                ax.add_patch(Rectangle((pick[0] - 0.5, pick[1] - 0.5), 1, 1,
                                       fill=False, edgecolor=PICK_EDGE,
                                       linewidth=1.1, zorder=5))

            ax.set_title(HOST_LABEL.get(panel, panel), fontsize=5.6, pad=1.5)
            xi = [0, len(lrs) - 1]
            ax.set_xticks(xi)
            ax.set_xticklabels([_sci(lrs[i]) for i in xi], fontsize=5,
                               rotation=30, ha="right", rotation_mode="anchor")
            yi = [0, len(gammas) - 1]
            ax.set_yticks(yi)
            ax.set_yticklabels([_sci(gammas[i]) for i in yi], fontsize=5)
            ax.tick_params(length=1.4, pad=1)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if ci == 0:
                ax.set_ylabel(domain + "\n" + r"$\gamma$", fontsize=5)
            if ri == len(FIG6_ROWS) - 1:
                ax.set_xlabel("learning rate", fontsize=4.6, labelpad=1)

    fig.tight_layout(pad=0.3, h_pad=0.45, w_pad=0.35, rect=(0, 0.10, 1, 1))
    cax = fig.add_axes([0.30, 0.020, 0.40, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("% worse than the panel's own best scored point  "
                 "(log scale; lighter is better)", fontsize=5.2, labelpad=1.5)
    cb.ax.tick_params(labelsize=4.6, length=1.5, pad=1)
    out = os.path.join(DATA, "fig6_gamma_surface.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (%d panels)" % (os.path.relpath(out, HERE), drawn))


if __name__ == "__main__":
    figure5()
    figure6()
