"""Figure 3 -- gamma sensitivity, in Lion's Figure 8 layout.

One figure, three panels: a line plot of how much each host loses as gamma moves
away from its best, then two contrasting (lr, gamma) surfaces under a shared
colourbar. Replaces the twelve-panel block, which is kept in the appendix.

    python benchmark12/export_figure_data.py
    python benchmark12/make_gamma_figure.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "paper", "figdata")
COLUMN_IN = 5.5
PCT_LO, PCT_HI = 0.3, 300.0
CMAP = "cividis_r"
PICK_EDGE = "#d7191c"

# Left panel: one line per host. OpenWebText is the 34M-parameter model and the
# only one whose gamma row is monotone over four orders of magnitude.
LINES = [("vision_cifar100", "CIFAR-100", "#4a7fb5"),
         ("language_wikitext103", "WikiText-103", "#c9772e"),
         ("finance_tech", "Finance-Tech", "#5aa469"),
         ("language_openwebtext", "OpenWebText", "#b2182b")]
# Right panels: a sharp learning-rate ridge against a broad basin.
MAPS = [("language_wikitext103", "WikiText-103"), ("finance_tech", "Finance-Tech")]


def load():
    panels = defaultdict(list)
    with open(os.path.join(DATA, "fig6_gamma_surface.csv"), encoding="utf-8",
              newline="") as fh:
        for r in csv.DictReader(fh):
            panels[r["panel"]].append(r)
    return panels


def grid_of(rs):
    lrs = sorted({float(r["lr"]) for r in rs})
    gammas = sorted({float(r["gamma"]) for r in rs})
    lower = rs[0]["lower_is_better"] == "1"
    g = [[float("nan")] * len(lrs) for _ in gammas]
    pick = None
    for r in rs:
        gi, li = gammas.index(float(r["gamma"])), lrs.index(float(r["lr"]))
        g[gi][li] = float(r["search_score"])
        if r["is_frozen"] == "1":
            pick = (li, gi)
    finite = [v for row in g for v in row if v == v]
    ref = min(finite) if lower else max(finite)
    pct = [[(100.0 * ((v - ref) if lower else (ref - v)) / abs(ref))
            if v == v else float("nan") for v in row] for row in g]
    return lrs, gammas, pct, pick, ref, lower


def _sci(v):
    if v >= 1000:
        return "%gk" % (v / 1000.0)
    return "%g" % v if v >= 1 else "%.0e" % v


def main():
    panels = load()
    fig = plt.figure(figsize=(COLUMN_IN, 1.75))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.35, 1, 1, 0.07],
                  wspace=0.42)
    ax0 = fig.add_subplot(gs[0, 0])

    for key, label, colour in LINES:
        if key not in panels:
            continue
        lrs, gammas, pct, pick, _, _ = grid_of(panels[key])
        li = pick[0] if pick else len(lrs) // 2          # the frozen lr column
        row = [pct[gi][li] for gi in range(len(gammas))]
        xs = [g for g, y in zip(gammas, row) if y == y]
        ys = [y for y in row if y == y]
        ax0.plot(xs, ys, color=colour, lw=1.2, marker="o", ms=2.2,
                 label=label, solid_capstyle="round")
    ax0.set_xscale("log")
    ax0.set_yscale("symlog", linthresh=1.0)
    ax0.set_ylim(bottom=-0.15)
    ax0.set_xlabel(r"$\gamma$", fontsize=6.5)
    ax0.set_ylabel("% above the host's\nbest scored point", fontsize=5.8)
    ax0.tick_params(labelsize=5, length=1.8, pad=1.5)
    ax0.grid(lw=0.25, alpha=0.3)
    ax0.set_axisbelow(True)
    for sp in ("top", "right"):
        ax0.spines[sp].set_visible(False)
    ax0.legend(fontsize=4.8, frameon=False, handlelength=1.2,
               labelspacing=0.2, ncol=2, columnspacing=0.8,
               loc="lower left", bbox_to_anchor=(0.0, 1.0, 1.0, 0.16),
               mode="expand", borderaxespad=0.0)

    norm = LogNorm(vmin=PCT_LO, vmax=PCT_HI)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("white")
    im = None
    for i, (key, title) in enumerate(MAPS):
        ax = fig.add_subplot(gs[0, 1 + i])
        lrs, gammas, pct, pick, _, _ = grid_of(panels[key])
        data = [[min(max(v, PCT_LO), PCT_HI) if v == v else float("nan")
                 for v in row] for row in pct]
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        if pick:
            ax.add_patch(Rectangle((pick[0] - 0.5, pick[1] - 0.5), 1, 1,
                                   fill=False, edgecolor=PICK_EDGE, lw=1.1,
                                   zorder=5))
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xticks([0, len(lrs) - 1])
        ax.set_xticklabels([_sci(lrs[0]), _sci(lrs[-1])], fontsize=5,
                           rotation=25, ha="right", rotation_mode="anchor")
        ax.set_yticks([0, len(gammas) - 1])
        ax.set_yticklabels([_sci(gammas[0]), _sci(gammas[-1])], fontsize=5)
        ax.tick_params(length=1.4, pad=1)
        ax.set_xlabel("learning rate", fontsize=5.8)
        if i == 0:
            ax.set_ylabel(r"$\gamma$", fontsize=6.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("% above best", fontsize=5.2, labelpad=9)
    cb.ax.tick_params(labelsize=4.6, length=1.5, pad=1)

    fig.subplots_adjust(left=0.11, right=0.93, top=0.80, bottom=0.26)
    out = os.path.join(DATA, "fig_gamma.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
