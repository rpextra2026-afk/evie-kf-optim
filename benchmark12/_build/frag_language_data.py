# ==============================================================================
#  DOMAIN: LANGUAGE  --  one small GPT on {WikiText-2, WikiText-103, PTB,
#  enwik8} + OpenWebText (language_openwebtext, the scalegen secondary task,
#  evie-scalegen-spec.md -- larger 8L/d384/ctx512 GPT, TASK-overridable, see
#  build_model()/make_loaders() below).
#  GENERATED: identical across all 5 language files (the 4 main + 1 scalegen).
#  Edit _build/frag_language_data.py
#
#  Model (CLAUDE.md sec 3): 6 layers, d_model=256, 8 heads, ctx=256, GELU MLP
#  (4x), pre-LN, causal self-attention, weight-tied token embedding / output
#  projection. ~10M parameters (body ~4.7M + tied embedding).
#
#  Vocab (CLAUDE.md sec 7 -- keep the RULE consistent across the 4 tasks, not
#  the size, which necessarily differs by corpus): word-level tasks use the
#  top-(VOCAB_CAP-1) train tokens by frequency + <unk> at id 0; enwik8 is
#  byte-level (fixed 256-symbol vocab). Perplexity is a WITHIN-TASK optimizer
#  comparison and is NOT comparable to external LM leaderboards (capped vocab,
#  whitespace tokenization).
#
#  WikiText-103 is capped by a training-token budget, not by taking a subsample
#  of the corpus (CLAUDE.md sec 3): the 10,000-step budget at ctx=256 x batch
#  consumes a fixed, documented number of tokens; the eval/test splits are the
#  full untouched canonical splits.
# ==============================================================================
import numpy as np

# primary metric = test_perplexity (the ONLY significance-tested quantity for
# language). Everything below is descriptive-only telemetry.
AUX_METRIC_KEYS = ["bpc", "test_loss", "token_acc"]


def estimate_fwd_flops(model, batch_size):
    """Rough forward-pass FLOPs (2*MACs): all Linear matmuls over B*T tokens,
    plus the attention score/​context matmuls. Descriptive only."""
    B, T = batch_size, TASK.get("ctx", GPT_CTX)
    flops = 0.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2.0 * m.in_features * m.out_features * B * T
    n_layer = len(model.blocks)
    d_model = model.wte.embedding_dim
    flops += 4.0 * B * n_layer * T * T * d_model        # QK^T + softmax@V
    return flops

GPT_N_LAYER = 6
GPT_D_MODEL = 256
GPT_N_HEAD  = 8
GPT_CTX     = 256
GPT_MLP_MULT = 4
VOCAB_CAP   = 16000        # word-level cap (enwik8 overrides to 256, byte-level)


class _Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, GPT_MLP_MULT * d)
        self.fc2 = nn.Linear(GPT_MLP_MULT * d, d)
        self.n_head = n_head
        self.d = d

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(self.d, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # Manual causal attention (NOT F.scaled_dot_product_attention): the fused
        # SDPA kernels do not support the DOUBLE backward that Sophia's
        # Hutchinson Hessian probe needs (fails on CPU, fragile on CUDA). This
        # plain form is double-backward-safe everywhere and is cheap at this
        # model size.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                            diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        a = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class SmallGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=GPT_N_LAYER, d_model=GPT_D_MODEL,
                 n_head=GPT_N_HEAD, ctx=GPT_CTX):
        super().__init__()
        d = d_model
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([_Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight            # weight tying
        self.apply(self._init)
        # Muon: exclude the (tied) token embedding, positional embedding and the
        # output projection -> those go to Muon's AdamW sub-group. Everything
        # else with ndim>=2 (attn/MLP matrices) gets the Muon update.
        self._muon_exclude_names = ["wte", "wpe", "lm_head"]

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
#  Corpora -> a single int64 token stream per split.
# ---------------------------------------------------------------------------
def _word_vocab(tokens, cap):
    from collections import Counter
    cnt = Counter(tokens)
    top = [w for w, _ in cnt.most_common(cap - 1)]
    stoi = {"<unk>": 0}
    for w in top:
        stoi[w] = len(stoi)
    return stoi


def _encode_words(tokens, stoi):
    unk = 0
    return np.fromiter((stoi.get(t, unk) for t in tokens), dtype=np.int32,
                       count=len(tokens))       # int32: 16k vocab, half the RAM


def _hf_text(name, config):
    _require("datasets")
    import datasets as hfds
    ds = hfds.load_dataset(name, config)
    return ("".join(ds["train"]["text"]),
            "".join(ds["validation"]["text"]),
            "".join(ds["test"]["text"]))


def _load_ptb():
    try:
        _require("datasets")
        import datasets as hfds
        ds = hfds.load_dataset("ptb_text_only", "penn_treebank", trust_remote_code=True)
        j = lambda s: "\n".join(x.strip() for x in ds[s]["sentence"])
        return j("train"), j("validation"), j("test")
    except Exception as e:
        log(f"[data] HF ptb_text_only failed ({e!r}); downloading raw Mikolov PTB")
        import urllib.request
        base = ("https://raw.githubusercontent.com/wojzaremba/lstm/master/data/")
        out = {}
        for split, fn in [("train", "ptb.train.txt"), ("validation", "ptb.valid.txt"),
                          ("test", "ptb.test.txt")]:
            with urllib.request.urlopen(base + fn, timeout=60) as r:
                out[split] = r.read().decode("utf-8")
        return out["train"], out["validation"], out["test"]


def _load_enwik8():
    root = os.environ.get("BENCH12_DATA_ROOT", os.path.join(CKPT_ROOT, "_data"))
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "enwik8")
    if not os.path.exists(path):
        import urllib.request, zipfile, io
        for url in ("https://huggingface.co/datasets/enwik8/resolve/main/enwik8.zip",
                    "http://mattmahoney.net/dc/enwik8.zip"):
            try:
                log(f"[data] downloading enwik8 from {url}")
                with urllib.request.urlopen(url, timeout=120) as r:
                    raw = r.read()
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    with z.open("enwik8") as f, open(path, "wb") as g:
                        g.write(f.read())
                break
            except Exception as e:
                log(f"[data] enwik8 source failed ({url}): {e!r}")
        if not os.path.exists(path):
            fail("could not download enwik8 from any known source; place the "
                 "decompressed 'enwik8' file at " + path)
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)  # bytes, cast in getitem
    n = len(data)
    n_tr = 90_000_000 if n >= 100_000_000 else int(n * 0.9)
    n_va = 5_000_000 if n >= 100_000_000 else int(n * 0.05)
    return data[:n_tr], data[n_tr:n_tr + n_va], data[n_tr + n_va:], 256


def _smoke_streams():
    rng = np.random.RandomState(0)
    V = 97
    return (rng.randint(0, V, size=20000).astype(np.int64),
            rng.randint(0, V, size=4000).astype(np.int64),
            rng.randint(0, V, size=4000).astype(np.int64), V)


# ---------------------------------------------------------------------------
#  OpenWebText (evie-scalegen-spec.md sec 3.2/4.2): GPT-2 BPE via tiktoken,
#  NOT the word-level _word_vocab/_encode_words pipeline above (real 50257-
#  token vocab, no <unk>). Streamed from Skylion007/openwebtext in HF
#  streaming mode -- never downloads the full ~40GB corpus -- and cached as
#  nanoGPT-style uint16 .bin files + meta.json under
#  {BENCH12_DATA_ROOT}/openwebtext_gpt2bpe/. A warm, verified cache loads with
#  no network at all (matches the enwik8/PTB "frozen snapshot" convention).
# ---------------------------------------------------------------------------
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _openwebtext_cache_ok(cache_dir, token_budget):
    """Returns the validated meta dict if a warm cache exists, matches the
    requested token_budget, and passes the token-count + sha256 integrity
    check (spec sec 5: never trust a cache hit without recomputing this).
    Returns None (with a logged reason) on any mismatch -- the caller
    rebuilds from scratch rather than silently continuing on a corrupt or
    truncated cache."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except Exception as e:
        log(f"[data] openwebtext: meta.json unreadable ({e!r}); rebuilding")
        return None
    if meta.get("tokenizer") != "gpt2" or meta.get("vocab") != 50257:
        log(f"[data] openwebtext: meta.json tokenizer/vocab mismatch; rebuilding")
        return None
    for split in ("train", "val", "test"):
        p = os.path.join(cache_dir, f"{split}.bin")
        want_n = meta.get("tokens", {}).get(split)
        if not os.path.exists(p) or want_n is None:
            log(f"[data] openwebtext: {split}.bin/meta entry missing; rebuilding")
            return None
        n_tok = os.path.getsize(p) // 2                # uint16 = 2 bytes/token
        if n_tok != want_n:
            log(f"[data] openwebtext: {split}.bin has {n_tok} tokens, "
                f"meta.json says {want_n}; rebuilding")
            return None
        sha = _sha256_file(p)
        want_sha = meta.get("sha256", {}).get(split)
        if sha != want_sha:
            log(f"[data] openwebtext: {split}.bin sha256 mismatch (cache "
                f"truncated/corrupt from an interrupted prior run); rebuilding")
            return None
    if meta["tokens"]["train"] < token_budget:
        log(f"[data] openwebtext: cached train={meta['tokens']['train']} tokens "
            f"< requested token_budget={token_budget}; rebuilding")
        return None
    return meta


def _build_openwebtext_cache(cache_dir, token_budget, val_tokens=5_000_000,
                             test_tokens=5_000_000):
    _require("tiktoken")
    _require("datasets")
    import tiktoken
    import datasets as hfds

    enc = tiktoken.get_encoding("gpt2")
    EOT = 50256
    need = token_budget + val_tokens + test_tokens
    log(f"[data] openwebtext: no warm/valid cache -- streaming "
        f"Skylion007/openwebtext (HF streaming mode, never downloads the full "
        f"~40GB corpus) until {need} tokens accumulated (train={token_budget} "
        f"val={val_tokens} test={test_tokens})")

    # Retry with backoff (spec sec 5): this custom streaming/tokenize path,
    # unlike torchvision/HF datasets' own loaders for the other 4 language
    # tasks, has no retry handling upstream.
    buf, total, last_err = [], 0, None
    for attempt in range(3):
        try:
            ds = hfds.load_dataset("Skylion007/openwebtext", split="train",
                                   streaming=True, trust_remote_code=True)
            for doc in ds:
                toks = enc.encode_ordinary(doc["text"])
                toks.append(EOT)
                buf.append(np.asarray(toks, dtype=np.uint16))
                total += len(toks)
                if len(buf) % 2000 == 0:
                    log(f"[data] openwebtext: streamed {total}/{need} tokens "
                        f"({len(buf)} docs)")
                if total >= need:
                    break
            break
        except Exception as e:
            last_err = e
            wait = 2.0 ** attempt
            log(f"[data] openwebtext: streaming pull failed ({e!r}); "
                f"retrying in {wait:.0f}s")
            time.sleep(wait)
    else:
        fail(f"openwebtext: HF streaming failed after 3 attempts: {last_err!r}")

    if total < need:
        fail(f"openwebtext stream exhausted at {total} tokens, need {need} "
             f"(train={token_budget} val={val_tokens} test={test_tokens}); "
             f"lower token_budget or check the HF streaming source")

    all_tok = np.concatenate(buf)
    # Fixed, deterministic slices (spec sec 3.2): first val_tokens -> val, next
    # test_tokens -> test, remainder up to token_budget -> train. Reproducible
    # without needing to shuffle a stream.
    splits = {
        "val": all_tok[:val_tokens],
        "test": all_tok[val_tokens:val_tokens + test_tokens],
        "train": all_tok[val_tokens + test_tokens:
                         val_tokens + test_tokens + token_budget],
    }

    # Atomic write: temp file + os.replace per split, same discipline as
    # atomic_torch_save/_write_json in frag_engine.py (spec sec 5) -- a
    # process killed mid-tokenization must leave either the old good cache or
    # nothing, never a half-written file a later run mistakes for complete.
    tmp_paths = {}
    for split, arr in splits.items():
        tmp = os.path.join(cache_dir, f".{split}.bin.tmp.{os.getpid()}")
        with open(tmp, "wb") as fh:
            arr.astype(np.uint16).tofile(fh)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_paths[split] = tmp

    meta = {
        "tokenizer": "gpt2", "vocab": 50257,
        "tokens": {s: int(len(a)) for s, a in splits.items()},
        "sha256": {s: _sha256_file(p) for s, p in tmp_paths.items()},
    }
    meta_tmp = os.path.join(cache_dir, f".meta.json.tmp.{os.getpid()}")
    with open(meta_tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())

    for split, tmp in tmp_paths.items():
        os.replace(tmp, os.path.join(cache_dir, f"{split}.bin"))
    os.replace(meta_tmp, os.path.join(cache_dir, "meta.json"))
    log(f"[data] openwebtext: wrote cache at {cache_dir} "
        f"(train={meta['tokens']['train']} val={meta['tokens']['val']} "
        f"test={meta['tokens']['test']} tokens)")
    return meta


def _load_openwebtext():
    cache_dir = os.path.join(
        os.environ.get("BENCH12_DATA_ROOT", os.path.join(CKPT_ROOT, "_data")),
        "openwebtext_gpt2bpe")
    os.makedirs(cache_dir, exist_ok=True)
    token_budget = int(TASK.get("token_budget", 220_000_000))

    meta = _openwebtext_cache_ok(cache_dir, token_budget)
    if meta is not None:
        log(f"[data] openwebtext: warm cache hit at {cache_dir}, no network")
    else:
        meta = _build_openwebtext_cache(cache_dir, token_budget)

    def _memmap(split):
        return np.memmap(os.path.join(cache_dir, f"{split}.bin"), dtype=np.uint16,
                         mode="r", shape=(meta["tokens"][split],))

    train = _memmap("train")[:token_budget]
    val, test = _memmap("val"), _memmap("test")
    return train, val, test, 50257


def prepare_data():
    ds = TASK["dataset"]
    if SMOKE:
        tr, va, te, V = _smoke_streams()
    elif ds == "enwik8":
        tr, va, te, V = _load_enwik8()
    elif ds == "openwebtext":
        tr, va, te, V = _load_openwebtext()
    else:
        if ds == "wikitext2":
            train_txt, val_txt, test_txt = _hf_text("wikitext", "wikitext-2-raw-v1")
        elif ds == "wikitext103":
            train_txt, val_txt, test_txt = _hf_text("wikitext", "wikitext-103-raw-v1")
        elif ds == "ptb":
            train_txt, val_txt, test_txt = _load_ptb()
        else:
            raise ValueError(f"unknown language dataset {ds!r}")
        tr_tok = train_txt.split()
        stoi = _word_vocab(tr_tok, VOCAB_CAP)
        V = len(stoi)
        tr = _encode_words(tr_tok, stoi)
        va = _encode_words(val_txt.split(), stoi)
        te = _encode_words(test_txt.split(), stoi)

    if ds == "wikitext103" and not SMOKE:
        budget = int(TASK.get("token_budget", 120_000_000))
        if len(tr) > budget:
            log(f"[data] wikitext103: capping train stream {len(tr)} -> {budget} "
                f"tokens (token-budget cap; eval/test splits are full)")
            tr = tr[:budget]

    if ds == "openwebtext":
        # real GPT-2 BPE vocab, no <unk>/OOV concept -- the train_oov line the
        # other 4 tasks print is meaningless here, so it's skipped, not faked
        # (evie-scalegen-spec.md sec 3.2).
        log(f"[data] {ds}: vocab={V}  train_tok={len(tr)}  val_tok={len(va)}  "
            f"test_tok={len(te)}  (GPT-2 BPE vocab -- no <unk>/OOV concept, "
            f"train_oov line intentionally skipped for this dataset)")
    else:
        oov = float(np.mean(tr == 0)) if len(tr) else float("nan")
        log(f"[data] {ds}: vocab={V}  train_tok={len(tr)}  val_tok={len(va)}  "
            f"test_tok={len(te)}  train_oov(<unk>)={oov:.3f}")
    return {"name": ds, "vocab": int(V), "train": tr, "val": va, "test": te}


class _LMWindows(torch.utils.data.Dataset):
    """Overlapping windows for train (stride 1), non-overlapping for eval."""

    def __init__(self, stream, ctx, stride):
        self.s = stream
        self.ctx = ctx
        self.stride = stride
        self.n = max(0, (len(stream) - ctx - 1) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = i * self.stride
        chunk = np.asarray(self.s[j:j + self.ctx + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def lr_search_units(data):
    return ["full"]


def main_units(data):
    return ["full"]


def build_model(data, unit):
    ms = data.get("model_size")          # set only by the model-size sweep driver
    ctx = TASK.get("ctx", GPT_CTX)        # scalegen (language_openwebtext): 512
    if ms:
        return SmallGPT(data["vocab"], n_layer=ms["n_layer"],
                        d_model=ms["d_model"], n_head=ms["n_head"], ctx=ctx)
    n_layer = TASK.get("n_layer", GPT_N_LAYER)
    d_model = TASK.get("d_model", GPT_D_MODEL)
    n_head = TASK.get("n_head", GPT_N_HEAD)
    return SmallGPT(data["vocab"], n_layer=n_layer, d_model=d_model,
                    n_head=n_head, ctx=ctx)


def make_criterion():
    return nn.CrossEntropyLoss()


def make_loaders(data, unit, batch_size, seed):
    nw = 0 if SMOKE else 2
    g = torch.Generator().manual_seed(seed)
    ctx = TASK.get("ctx", GPT_CTX)
    tr = _LMWindows(data["train"], ctx, stride=ctx)      # non-overlap train
    va = _LMWindows(data["val"], ctx, stride=ctx)
    te = _LMWindows(data["test"], ctx, stride=ctx)
    eval_bs = max(batch_size, 64)
    train_loader = torch.utils.data.DataLoader(
        tr, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=nw,
        pin_memory=torch.cuda.is_available(), generator=g,
        persistent_workers=(nw > 0))
    val_loader = torch.utils.data.DataLoader(
        va, batch_size=eval_bs, shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), persistent_workers=(nw > 0))
    test_loader = torch.utils.data.DataLoader(
        te, batch_size=eval_bs, shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), persistent_workers=(nw > 0))
    return train_loader, val_loader, test_loader


def batch_to_device(batch):
    x, y = batch
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)


def forward_loss(model, batch, criterion):
    x, y = batch
    logits = model(x)
    return criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_tok, ce_sum, correct = 0, 0.0, 0
    for batch in loader:
        x, y = batch_to_device(batch)
        logits = model(x)
        flat = logits.reshape(-1, logits.size(-1))
        yt = y.reshape(-1)
        ce_sum += float(F.cross_entropy(flat, yt, reduction="sum").item())
        tot_tok += yt.numel()
        correct += int((flat.argmax(-1) == yt).sum().item())
    mean_ce = ce_sum / max(tot_tok, 1)
    return math.exp(min(mean_ce, 50.0)), {
        "bpc": mean_ce / math.log(2.0),          # bits per symbol (== bpc for enwik8)
        "test_loss": mean_ce,
        "token_acc": correct / max(tot_tok, 1),  # next-token top-1, descriptive
    }
