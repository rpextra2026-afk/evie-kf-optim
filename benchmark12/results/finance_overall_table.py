"""Consolidated finance table: one number per optimizer per metric, all panels.

Prints and writes FINANCE_OVERALL.txt from the committed results_raw.csv files. No numbers
are typed in by hand.

Per panel: complete-case tickers (a ticker any arm diverged on is dropped for
all arms), median over tickers per seed, median over seeds. Overall = mean of
the per-panel values. MSE uses all 4 panels. RMSE and MAE use the 3 panels
where every run recorded them: on finance_tech, 1145 cells were recovered from
logs without secondary metrics, so tech is excluded for every arm.
"""
import csv, math, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OPTS = ["EvieKF", "AdamW", "AdaBelief", "SGD", "Sophia", "Muon", "Lion", "Shampoo"]
TASKS = ["finance_tech", "finance_consumer_industrials", "finance_healthcare", "finance_finance"]

val, div, tot = defaultdict(dict), defaultdict(int), defaultdict(int)
for task in TASKS:
    rows = list(csv.DictReader(open(HERE / task / "results_raw.csv")))
    bad = {r["unit"] for r in rows if r["diverged"] != "False"}
    for r in rows:
        tot[r["opt"]] += 1
        div[r["opt"]] += r["diverged"] != "False"
    for m in ("primary", "rmse", "mae"):
        for o in OPTS:
            per_seed = defaultdict(list)
            for r in rows:
                if r["opt"] == o and r["unit"] not in bad and r.get(m) not in (None, "", "nan"):
                    v = float(r[m])
                    if math.isfinite(v):
                        per_seed[r["seed"]].append(v)
            if per_seed:
                val[(m, task)][o] = st.median(st.median(v) for v in per_seed.values())

def overall(m, tasks):
    return {o: st.mean(val[(m, t)][o] for t in tasks) for o in OPTS}

mse, rmse, mae = overall("primary", TASKS), overall("rmse", TASKS[1:]), overall("mae", TASKS[1:])
rows = [
    ("test_mse (x1e-4)", [f"{mse[o]*1e4:.4f}" for o in OPTS]),
    ("test_rmse", [f"{rmse[o]:.5f}" for o in OPTS]),
    ("test_mae", [f"{mae[o]:.5f}" for o in OPTS]),
    ("diverged_runs", [f"{div[o]}/{tot[o]}" for o in OPTS]),
]
w0 = max(len(r[0]) for r in rows)
w = max(max(len(x) for x in OPTS), max(len(v) for _, vals in rows for v in vals)) + 2
out = ["metric".ljust(w0) + "".join(o.rjust(w) for o in OPTS)]
out += [name.ljust(w0) + "".join(v.rjust(w) for v in vals) for name, vals in rows]
out += ["", "(rmse/mae exclude finance_tech)"]
text = "\n".join(out) + "\n"
print(text, end="")
(HERE / "FINANCE_OVERALL.txt").write_text(text, encoding="utf-8")
