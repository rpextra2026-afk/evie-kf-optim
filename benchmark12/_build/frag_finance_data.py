# ==============================================================================
#  DOMAIN: FINANCE  --  3-layer OHLCV MLP, next-day log-return regression, one
#  sector panel per file: {Tech, Finance, Healthcare, Consumer_Industrials}.
#  GENERATED: identical across all 4 finance files. Edit
#  _build/frag_finance_data.py
#
#  Panels (CLAUDE.md sec 3, resolved + extended): the 4 sectors that had >= 30
#  candidate tickers in the original 140-ticker pool (Energy, at 20, stays
#  dropped). Each sector pool is expanded to 50 candidates (original 30 + next
#  ~20 by market cap; see scripts/download_finance_panels.py). Each panel =
#  EVERY one of those meeting the old repo's usability rule (>= 300 usable rows
#  after return alignment; 80/10/10 chronological split; scaler fit on train
#  only) -- typically ~40-48 tickers after newer IPOs/spin-offs are filtered.
#  The panel is a FROZEN list: data is downloaded once by
#  scripts/download_finance_panels.py and committed as
#  benchmark12/finance/data/<panel>.csv -- there are NO live yfinance calls at
#  runtime.
#
#  LR-search subset: 8 tickers per panel, chosen once with random.Random(42)
#  (sample without replacement) and stored in the data manifest; the same 8 are
#  reused for every optimizer's LR search on that panel.
#
#  Unit of analysis for the 12-task test is the PANEL: per (optimizer, seed) we
#  train one MLP per ticker and average test MSE over the panel; the per-seed
#  panel numbers are then median-reduced over seeds (done in the engine).
# ==============================================================================
import numpy as np

_require("pandas")
import pandas as pd

# primary metric = test_mse (the ONLY significance-tested quantity for finance).
# Everything below is descriptive-only telemetry.
AUX_METRIC_KEYS = ["rmse", "dir_acc", "test_loss", "mae", "r2", "sign_sharpe"]


def estimate_fwd_flops(model, batch_size):
    """Rough forward-pass FLOPs (2*MACs) over the MLP. Descriptive only."""
    return float(sum(2.0 * m.in_features * m.out_features * batch_size
                     for m in model.modules() if isinstance(m, nn.Linear)))

FIN_H1, FIN_H2 = 64, 32
MIN_USABLE_ROWS = 300
N_LR_SEARCH_TICKERS = 8
LR_SEARCH_TICKER_SEED = 42


class FinanceMLP(nn.Module):
    def __init__(self, h1=FIN_H1, h2=FIN_H2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self._muon_exclude_names = []      # Muon acts on the 3 weight matrices

    def forward(self, x):
        return self.net(x)


def _data_path():
    if os.environ.get("BENCH12_FINANCE_DATA"):
        return os.environ["BENCH12_FINANCE_DATA"]
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    return os.path.join(here, "data", f"{TASK['panel']}.csv")


def _manifest_path():
    return os.path.join(os.path.dirname(_data_path()), "manifest.json")


def _build_ticker_arrays(df_t):
    """df_t: rows for one ticker, columns open/high/low/close/volume, date-sorted.
    Returns (Xtr,ytr,Xva,yva,Xte,yte) or None if not usable."""
    ohlcv = df_t[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
    close = ohlcv[:, 3]
    if len(close) < MIN_USABLE_ROWS + 2:
        return None
    ratio = close[1:] / np.maximum(close[:-1], 1e-12)
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1e-8)
    y = np.log(np.maximum(ratio, 1e-8)).astype(np.float64)
    X = ohlcv[:-1]
    n = len(X)
    if n < MIN_USABLE_ROWS:
        return None
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    if min(n_tr, n_va, n - n_tr - n_va) < 20:
        return None
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    Xte, yte = X[n_tr + n_va:], y[n_tr + n_va:]
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    return (((Xtr - mu) / sd).astype(np.float32), ytr.astype(np.float32),
            ((Xva - mu) / sd).astype(np.float32), yva.astype(np.float32),
            ((Xte - mu) / sd).astype(np.float32), yte.astype(np.float32))


def _smoke_panel():
    rng = np.random.RandomState(1)
    out = {}
    for i in range(4):
        n = 400
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        o = close * (1 + rng.normal(0, 0.005, n))
        h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
        lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
        vol = rng.uniform(1e6, 5e6, n)
        df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": close,
                           "volume": vol})
        arr = _build_ticker_arrays(df)
        if arr is not None:
            out[f"SMK{i}"] = arr
    return out, list(out)[:2]


def prepare_data():
    if SMOKE:
        cache, lr_tickers = _smoke_panel()
        log(f"[data] SMOKE finance panel: {list(cache)}")
        return {"panel": TASK["panel"], "cache": cache,
                "main_tickers": list(cache), "lr_tickers": lr_tickers}

    path = _data_path()
    if not os.path.exists(path):
        fail(f"frozen panel CSV not found: {path}. Run "
             f"`python scripts/download_finance_panels.py` once and commit "
             f"benchmark12/finance/data/*.csv (CLAUDE.md sec 3 -- no live "
             f"yfinance at runtime).")
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    cache, dropped = {}, []
    for tk, g in df.sort_values(["ticker", "date"]).groupby("ticker"):
        arr = _build_ticker_arrays(g.reset_index(drop=True))
        if arr is None:
            dropped.append(tk)
        else:
            cache[str(tk)] = arr
    main_tickers = sorted(cache)
    if not main_tickers:
        fail(f"no usable tickers in {path}")

    manifest = _read_json(_manifest_path(), {})
    mkey = TASK["panel"]
    lr_tickers = (manifest.get(mkey, {}) or {}).get("lr_search_tickers")
    if not lr_tickers or any(t not in cache for t in lr_tickers):
        rr = random.Random(LR_SEARCH_TICKER_SEED)
        lr_tickers = rr.sample(main_tickers,
                               min(N_LR_SEARCH_TICKERS, len(main_tickers)))
    log(f"[data] panel={mkey}: {len(main_tickers)} usable tickers "
        f"(dropped {len(dropped)}: {dropped}); LR-search tickers={lr_tickers}")
    return {"panel": mkey, "cache": cache, "main_tickers": main_tickers,
            "lr_tickers": list(lr_tickers)}


def lr_search_units(data):
    return data["lr_tickers"]


def main_units(data):
    return data["main_tickers"]


def build_model(data, unit):
    ms = data.get("model_size")          # set only by the model-size sweep driver
    if ms:
        return FinanceMLP(int(ms[0]), int(ms[1]))
    return FinanceMLP(FIN_H1, FIN_H2)


def make_criterion():
    return nn.MSELoss()


class _FinDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(data, unit, batch_size, seed):
    Xtr, ytr, Xva, yva, Xte, yte = data["cache"][unit]
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        _FinDS(Xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=True,
        generator=g)
    val_loader = torch.utils.data.DataLoader(_FinDS(Xva, yva), batch_size=512)
    test_loader = torch.utils.data.DataLoader(_FinDS(Xte, yte), batch_size=512)
    return train_loader, val_loader, test_loader


def batch_to_device(batch):
    x, y = batch
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)


def forward_loss(model, batch, criterion):
    x, y = batch
    return criterion(model(x), y)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    se_sum, ae_sum, n, hit, hit_n = 0.0, 0.0, 0, 0, 0
    preds, tgts, strat = [], [], []
    for batch in loader:
        x, y = batch_to_device(batch)
        p = model(x)
        se_sum += float(((p - y) ** 2).sum().item())
        ae_sum += float((p - y).abs().sum().item())
        n += y.numel()
        preds.append(p.reshape(-1).cpu()); tgts.append(y.reshape(-1).cpu())
        mask = y.abs() > 1e-9
        if mask.any():
            hit += int(((p[mask] * y[mask]) > 0).sum().item())
            hit_n += int(mask.sum().item())
        strat.append((torch.sign(p) * y).reshape(-1).cpu())
    mse = se_sum / max(n, 1)
    yt = torch.cat(tgts) if tgts else torch.zeros(1)
    ss_tot = float(((yt - yt.mean()) ** 2).sum().item())
    r2 = 1.0 - se_sum / ss_tot if ss_tot > 0 else float("nan")
    s = torch.cat(strat) if strat else torch.zeros(1)
    sharpe = (float(s.mean().item()) / (float(s.std().item()) + 1e-12)
              * math.sqrt(252.0)) if s.numel() > 1 else float("nan")
    return mse, {
        "rmse": math.sqrt(max(mse, 0.0)),
        "dir_acc": (hit / hit_n) if hit_n else float("nan"),
        "test_loss": mse,
        "mae": ae_sum / max(n, 1),               # descriptive
        "r2": r2,                                # descriptive
        "sign_sharpe": sharpe,                   # ann. Sharpe of a sign(pred) strategy
    }
