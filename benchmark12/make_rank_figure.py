"""Rank strip: each arm's position on all 12 tasks, one dot per task."""
import csv, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("benchmark12/results/paper/table1_main_results.csv")))
tasks = [k for k in rows[0] if k != "Optimizer"]
LOWER = {"vis": False, "lang": True, "fin": True}
v = {r["Optimizer"]: {t: float(r[t]) for t in tasks} for r in rows}
ARMS = ["EvieKF","AdamW","AdaBelief","SGD","Sophia","Muon","Lion","Shampoo"]
DOMC = {"vis": "#4a7fb5", "lang": "#c9772e", "fin": "#5aa469"}
DOML = {"vis": "Vision", "lang": "Language", "fin": "Finance"}

rank = {a: {} for a in ARMS}
for t in tasks:
    p = t.split("/")[0]
    for i, a in enumerate(sorted(ARMS, key=lambda x: v[x][t], reverse=not LOWER[p])):
        rank[a][t] = i + 1

order = sorted(ARMS, key=lambda a: st.mean(rank[a].values()))
fig, ax = plt.subplots(figsize=(5.5, 1.85))

for yi, a in enumerate(order):
    y = len(order) - 1 - yi
    counts = {}
    for t in tasks:
        r = rank[a][t]
        counts[r] = counts.get(r, 0) + 1
        off = (counts[r] - 1) * 0.135 - 0.0
        ax.scatter(r, y + off - 0.07, s=13, color=DOMC[t.split("/")[0]],
                   zorder=3, edgecolors="white", linewidths=0.35)
    m = st.mean(rank[a].values())
    ax.plot([m, m], [y - 0.34, y + 0.34], color="#b2182b", lw=1.5, zorder=4)
    ax.text(8.55, y, "%.2f" % m, fontsize=5.6, va="center", color="#b2182b")

ax.set_yticks(range(len(order)))
ax.set_yticklabels(list(reversed(order)), fontsize=6)
ax.set_xticks(range(1, 9)); ax.set_xlim(0.5, 8.95)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.tick_params(axis="x", labelsize=6, length=2)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("rank on a task  (1 = best of 8)", fontsize=6.5)
ax.text(8.55, len(order) - 0.28, "mean", fontsize=5.6, color="#b2182b", va="center")
ax.grid(axis="x", lw=0.3, alpha=0.3); ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(0.5)
h = [plt.Line2D([0],[0],marker="o",ls="",ms=4,color=DOMC[k],label=DOML[k]) for k in ["vis","lang","fin"]]
ax.legend(handles=h, fontsize=5.8, ncol=3, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.20), handletextpad=0.2, columnspacing=1.2)
fig.tight_layout(pad=0.3)
out = "benchmark12/results/paper/figdata/fig_rank_strip.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(os.environ["SP"]+"/rank.png", dpi=260, bbox_inches="tight")
print("wrote", out)
