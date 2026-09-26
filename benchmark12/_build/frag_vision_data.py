# ==============================================================================
#  DOMAIN: VISION  --  CIFAR-stem ResNet-18 on {CIFAR-10, CIFAR-100, SVHN,
#  STL-10} + Tiny-ImageNet-200 (vision_tinyimagenet, the scalegen secondary
#  task, evie-scalegen-spec.md).
#  GENERATED: identical across all 5 vision files (the 4 main + 1 scalegen).
#  Edit _build/frag_vision_data.py
#
#  Architecture parity (CLAUDE.md sec 3): the SAME CIFAR-stem ResNet-18 (3x3
#  stem conv, stride 1, NO initial max-pool -- the standard "ResNet for 32x32"
#  variant) is used for all four tasks. STL-10 (natively 96x96) is resized to
#  32x32 so the architecture and compute profile are identical to the other
#  three; STL-10's role here is its low-data-by-design property (5,000 labelled
#  training images), which survives the resize.
#
#  Init: standard ResNet init (Kaiming-normal / fan_out for the ReLU convs,
#  BN weight=1 bias=0, zero-init the last BN in each residual branch, Linear
#  ~ N(0, 0.01)). This intentionally departs from the old 2-conv CNN's
#  Xavier-uniform default (CLAUDE.md sec 4): Xavier-uniform on a deep 3x3-conv
#  ReLU stack systematically under-scales early-layer signal and would handicap
#  SGD relative to the adaptive optimizers, contaminating the comparison. See
#  benchmark12/README.md "Deviations".
# ==============================================================================
import numpy as np

# primary metric = test_top1_acc (the ONLY significance-tested quantity for
# vision). Everything below is descriptive-only telemetry.
AUX_METRIC_KEYS = ["top5", "test_loss", "ece"]


def estimate_fwd_flops(model, batch_size):
    """Rough forward-pass FLOPs (2*MACs). Descriptive only."""
    flops, h = 0.0, float(TASK.get("image_size", 32))
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            s = m.stride[0] if isinstance(m.stride, tuple) else m.stride
            ho = math.ceil(h / s)
            flops += (2.0 * m.in_channels * m.out_channels
                      * m.kernel_size[0] * m.kernel_size[1] * ho * ho * batch_size)
            if m.kernel_size[0] >= 3:      # only the 3x3 path drives the spatial size
                h = ho
        elif isinstance(m, nn.Linear):
            flops += 2.0 * m.in_features * m.out_features * batch_size
    return flops

_VISION_MEAN_STD = {}   # filled per-dataset from the (post val-carve) train split


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


RESNET18_WIDTHS = (64, 128, 256, 512)   # baseline; the model-size sweep overrides


class CifarStemResNet18(nn.Module):
    """ResNet-18 with a 3x3 stride-1 stem and no initial max-pool (32x32 input).
    `widths` is the 4 stage channel counts (default = the benchmark baseline;
    the model-size sweep passes 0.5x / 2x variants). `stem_pool` (default False,
    so the other 4 vision tasks are byte-for-byte unaffected) inserts one
    nn.MaxPool2d(2) right after the stem conv+BN+ReLU, before layer1 -- the
    minimal "CIFAR stem + one pool" adaptation for 64x64 input used by
    vision_tinyimagenet (evie-scalegen-spec.md sec 3.1), keeping the exact same
    4-stage block structure/width/init as every other vision task."""

    def __init__(self, num_classes, widths=RESNET18_WIDTHS, stem_pool=False):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.in_c = w0
        self.stem = nn.Conv2d(3, w0, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w0)
        self.stem_pool = nn.MaxPool2d(2) if stem_pool else None
        self.layer1 = self._make_layer(w0, 2, stride=1)
        self.layer2 = self._make_layer(w1, 2, stride=2)
        self.layer3 = self._make_layer(w2, 2, stride=2)
        self.layer4 = self._make_layer(w3, 2, stride=2)
        self.fc = nn.Linear(w3, num_classes)
        self._init_weights()
        # Muon operates on all >=2D params (conv weights reshaped to 2D + the
        # classifier matrix); nothing is excluded for vision.
        self._muon_exclude_names = []

    def _make_layer(self, out_c, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_c, out_c, s))
            self.in_c = out_c * _BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        for m in self.modules():                    # zero-init last BN per branch
            if isinstance(m, _BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.stem(x)), inplace=True)
        if self.stem_pool is not None:
            x = self.stem_pool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
#  Data: load once into uint8 CPU tensors, carve a deterministic val split from
#  train, compute normalization stats on the train split only.
# ---------------------------------------------------------------------------
def _load_raw(name):
    _require("torchvision")
    import torchvision
    root = os.environ.get("BENCH12_DATA_ROOT", os.path.join(CKPT_ROOT, "_data"))
    os.makedirs(root, exist_ok=True)
    if name == "cifar10":
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        Xtr = np.asarray(tr.data, dtype=np.uint8)                 # (N,32,32,3)
        ytr = np.asarray(tr.targets, dtype=np.int64)
        Xte = np.asarray(te.data, dtype=np.uint8)
        yte = np.asarray(te.targets, dtype=np.int64)
        ncls = 10
    elif name == "cifar100":
        tr = torchvision.datasets.CIFAR100(root, train=True, download=True)
        te = torchvision.datasets.CIFAR100(root, train=False, download=True)
        Xtr = np.asarray(tr.data, dtype=np.uint8)
        ytr = np.asarray(tr.targets, dtype=np.int64)
        Xte = np.asarray(te.data, dtype=np.uint8)
        yte = np.asarray(te.targets, dtype=np.int64)
        ncls = 100
    elif name == "svhn":
        tr = torchvision.datasets.SVHN(root, split="train", download=True)
        te = torchvision.datasets.SVHN(root, split="test", download=True)
        Xtr = np.transpose(np.asarray(tr.data, dtype=np.uint8), (0, 2, 3, 1))
        ytr = np.asarray(tr.labels, dtype=np.int64)
        Xte = np.transpose(np.asarray(te.data, dtype=np.uint8), (0, 2, 3, 1))
        yte = np.asarray(te.labels, dtype=np.int64)
        ncls = 10
    elif name == "stl10":
        tr = torchvision.datasets.STL10(root, split="train", download=True)
        te = torchvision.datasets.STL10(root, split="test", download=True)
        # (N,3,96,96) uint8 -> resize to 32x32 once, store as (N,32,32,3)
        def _resize(arr):
            t = torch.from_numpy(np.asarray(arr, dtype=np.uint8)).float()
            t = F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
            return t.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        Xtr = _resize(tr.data); ytr = np.asarray(tr.labels, dtype=np.int64)
        Xte = _resize(te.data); yte = np.asarray(te.labels, dtype=np.int64)
        ncls = 10
    elif name == "tinyimagenet":
        Xtr, ytr, Xte, yte, ncls = _load_tinyimagenet(root)
    else:
        raise ValueError(f"unknown vision dataset {name!r}")
    return Xtr, ytr, Xte, yte, ncls


# ---------------------------------------------------------------------------
#  Tiny-ImageNet-200 (evie-scalegen-spec.md sec 3.1/4.1): native 64x64, 200
#  classes, 100k train / 10k val. Decoded once, cached as a single atomic
#  .npz under {BENCH12_DATA_ROOT}/tinyimagenet/ -- warm re-runs skip the
#  zip/decode step entirely (no network at all), same "check local cache,
#  download once" convention _load_enwik8() already uses in
#  frag_language_data.py. Held-out TEST = the official 10k-image val split
#  (labels public); the internal validation split is carved from the 100k
#  train images by prepare_data() exactly as for the other 4 vision datasets.
# ---------------------------------------------------------------------------
def _download_with_retry(url, timeout=120, attempts=3):
    """3-attempt exponential backoff (evie-scalegen-spec.md sec 5) -- this
    custom download/decode path, unlike torchvision/HF datasets' own loaders,
    has no retry handling upstream."""
    import urllib.request
    last_err = None
    for i in range(attempts):
        try:
            log(f"[data] downloading {url} (attempt {i + 1}/{attempts})")
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
            wait = 2.0 ** i
            log(f"[data] download failed ({e!r}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise last_err


def _check_tinyimagenet_integrity(Xtr, ytr, Xte, yte):
    """Fail with one clear sentence naming the actual counts, not a silent
    continue on a corrupt/truncated decode (evie-scalegen-spec.md sec 5)."""
    n_tr, n_te, n_cls = len(Xtr), len(Xte), len(np.unique(ytr))
    if n_tr != 100_000 or n_te != 10_000 or n_cls != 200:
        fail(f"tinyimagenet decode/cache integrity check failed: "
             f"train={n_tr} (want 100000)  val={n_te} (want 10000)  "
             f"classes={n_cls} (want 200)")


def _decode_tinyimagenet_zip(raw_bytes):
    _require("PIL", "Pillow")
    from PIL import Image
    import zipfile, io
    zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    names = zf.namelist()

    wnids = [w.strip() for w in
             zf.read("tiny-imagenet-200/wnids.txt").decode().splitlines() if w.strip()]
    assert len(wnids) == 200, f"expected 200 wnids, found {len(wnids)}"
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    def _read_img(n):
        with zf.open(n) as fh:
            img = Image.open(fh).convert("RGB")
        if img.size != (64, 64):
            img = img.resize((64, 64))
        return np.asarray(img, dtype=np.uint8)

    train_files = sorted(n for n in names
                         if n.startswith("tiny-imagenet-200/train/")
                         and n.lower().endswith(".jpeg"))
    Xtr = np.zeros((len(train_files), 64, 64, 3), dtype=np.uint8)
    ytr = np.zeros(len(train_files), dtype=np.int64)
    for i, n in enumerate(train_files):
        Xtr[i] = _read_img(n)
        ytr[i] = wnid_to_idx[n.split("/")[2]]           # .../train/<wnid>/images/...

    ann = zf.read("tiny-imagenet-200/val/val_annotations.txt").decode().splitlines()
    val_label = {}
    for line in ann:
        parts = line.split("\t")
        val_label[parts[0]] = wnid_to_idx[parts[1]]
    val_files = sorted(n for n in names
                       if n.startswith("tiny-imagenet-200/val/images/")
                       and n.lower().endswith(".jpeg"))
    Xte = np.zeros((len(val_files), 64, 64, 3), dtype=np.uint8)
    yte = np.zeros(len(val_files), dtype=np.int64)
    for i, n in enumerate(val_files):
        Xte[i] = _read_img(n)
        yte[i] = val_label[n.split("/")[-1]]
    return Xtr, ytr, Xte, yte


def _decode_tinyimagenet_hf():
    """Fallback source: HF mirror (evie-scalegen-spec.md sec 4.1) -- same
    try-URL-then-HF pattern PTB already uses in frag_language_data.py."""
    _require("datasets")
    import datasets as hfds
    last_err = None
    ds = None
    for i in range(3):
        try:
            ds = hfds.load_dataset("zh-plus/tiny-imagenet")
            break
        except Exception as e:
            last_err = e
            wait = 2.0 ** i
            log(f"[data] tinyimagenet HF mirror load failed ({e!r}); "
                f"retrying in {wait:.0f}s")
            time.sleep(wait)
    if ds is None:
        raise RuntimeError(f"HF mirror zh-plus/tiny-imagenet failed after 3 "
                           f"attempts: {last_err!r}")

    def _to_arrays(split):
        n = len(ds[split])
        X = np.zeros((n, 64, 64, 3), dtype=np.uint8)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            ex = ds[split][i]
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (64, 64):
                img = img.resize((64, 64))
            X[i] = np.asarray(img, dtype=np.uint8)
            y[i] = int(ex["label"])
        return X, y

    Xtr, ytr = _to_arrays("train")
    Xte, yte = _to_arrays("valid")
    return Xtr, ytr, Xte, yte


def _load_tinyimagenet(data_root):
    cache_dir = os.path.join(data_root, "tinyimagenet")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "tinyimagenet.npz")

    if os.path.exists(cache_path):
        try:
            npz = np.load(cache_path)
            Xtr, ytr, Xte, yte = (npz["Xtr"], npz["ytr"], npz["Xte"], npz["yte"])
            _check_tinyimagenet_integrity(Xtr, ytr, Xte, yte)
            log(f"[data] tinyimagenet: warm cache hit at {cache_path}, no network")
            return Xtr, ytr, Xte, yte, 200
        except SystemExit:
            raise
        except Exception as e:
            log(f"[data] tinyimagenet: cache at {cache_path} unreadable/corrupt "
                f"({e!r}); re-downloading")

    try:
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        raw = _download_with_retry(url)
        Xtr, ytr, Xte, yte = _decode_tinyimagenet_zip(raw)
    except Exception as e:
        log(f"[data] tinyimagenet: primary cs231n source failed ({e!r}); "
            f"falling back to the HF mirror")
        Xtr, ytr, Xte, yte = _decode_tinyimagenet_hf()

    _check_tinyimagenet_integrity(Xtr, ytr, Xte, yte)

    # atomic cache write (temp file + os.replace), same discipline as
    # atomic_torch_save/_write_json in frag_engine.py (evie-scalegen-spec sec 5):
    # a process killed mid-write must leave either no cache or the old good one,
    # never a half-written file a later run mistakes for complete.
    tmp = os.path.join(cache_dir, f".tinyimagenet.tmp.{os.getpid()}.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, cache_path)
    log(f"[data] tinyimagenet: wrote cache {cache_path} "
        f"(train={len(Xtr)} val={len(Xte)} classes=200)")
    return Xtr, ytr, Xte, yte, 200


class _VisionDataset(torch.utils.data.Dataset):
    def __init__(self, X_u8, y, mean, std, train, hflip):
        self.X = X_u8                                   # (N,32,32,3) uint8
        self.y = torch.from_numpy(y)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.train = train
        self.hflip = hflip

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = torch.from_numpy(np.ascontiguousarray(self.X[i])).permute(2, 0, 1).float() / 255.0
        if self.train:
            # size/pad generalized off TASK["image_size"] (default 32, matching
            # every non-64x64 task): pad = size // 8, crop range = randint(0, 2*
            # pad+1) -- reduces to exactly the pre-existing pad=4/randint(0,9)
            # behavior at size=32 (evie-scalegen-spec.md sec 3.1).
            size = TASK.get("image_size", 32)
            pad = size // 8
            img = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
            top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            img = img[:, top:top + size, left:left + size]
            if self.hflip and torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
        img = (img - self.mean) / self.std
        return img, self.y[i]


def _smoke_raw(name):
    rng = np.random.RandomState(0)
    ncls = 200 if name == "tinyimagenet" else (100 if name == "cifar100" else 10)
    size = 64 if name == "tinyimagenet" else 32
    Xtr = rng.randint(0, 256, size=(320, size, size, 3), dtype=np.uint8)
    ytr = rng.randint(0, ncls, size=320).astype(np.int64)
    Xte = rng.randint(0, 256, size=(128, size, size, 3), dtype=np.uint8)
    yte = rng.randint(0, ncls, size=128).astype(np.int64)
    return Xtr, ytr, Xte, yte, ncls


def prepare_data():
    name = TASK["dataset"]
    Xtr, ytr, Xte, yte, ncls = _smoke_raw(name) if SMOKE else _load_raw(name)
    rng = np.random.RandomState(2024)
    perm = rng.permutation(len(Xtr))
    n_val = max(500, int(round(len(Xtr) * 0.1)))
    if SMOKE:
        n_val = min(n_val, 64)
        keep = 256
        perm = perm[: keep + n_val]
        Xte, yte = Xte[:128], yte[:128]
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_s, ytr_s = Xtr[tr_idx], ytr[tr_idx]
    Xva_s, yva_s = Xtr[va_idx], ytr[va_idx]
    mean = (Xtr_s.reshape(-1, 3).mean(0) / 255.0).astype(np.float32)
    std = (Xtr_s.reshape(-1, 3).std(0) / 255.0 + 1e-6).astype(np.float32)
    _VISION_MEAN_STD[name] = (mean.tolist(), std.tolist())
    hflip = name != "svhn"
    log(f"[data] {name}: train={len(Xtr_s)} val={len(Xva_s)} test={len(Xte)} "
        f"classes={ncls}  mean={mean.round(4).tolist()} std={std.round(4).tolist()}")
    return {
        "name": name, "ncls": ncls, "mean": mean.tolist(), "std": std.tolist(),
        "hflip": hflip,
        "train": (Xtr_s, ytr_s), "val": (Xva_s, yva_s), "test": (Xte, yte),
    }


def lr_search_units(data):
    return ["full"]


def main_units(data):
    return ["full"]


def build_model(data, unit):
    ms = data.get("model_size")          # set only by the model-size sweep driver
    stem_pool = (TASK.get("image_size", 32) == 64)   # sec 3.1: 64x64 needs one pool
    if ms:
        return CifarStemResNet18(data["ncls"], widths=tuple(ms), stem_pool=stem_pool)
    return CifarStemResNet18(data["ncls"], stem_pool=stem_pool)


def make_criterion():
    return nn.CrossEntropyLoss()


def make_loaders(data, unit, batch_size, seed):
    m, s = data["mean"], data["std"]
    g = torch.Generator().manual_seed(seed)
    nw = 0 if SMOKE else 2
    tr = _VisionDataset(*data["train"], m, s, train=True, hflip=data["hflip"])
    va = _VisionDataset(*data["val"], m, s, train=False, hflip=False)
    te = _VisionDataset(*data["test"], m, s, train=False, hflip=False)
    train_loader = torch.utils.data.DataLoader(
        tr, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=nw, pin_memory=torch.cuda.is_available(), generator=g,
        persistent_workers=(nw > 0))
    val_loader = torch.utils.data.DataLoader(
        va, batch_size=256, shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), persistent_workers=(nw > 0))
    test_loader = torch.utils.data.DataLoader(
        te, batch_size=256, shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), persistent_workers=(nw > 0))
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
    tot, loss_sum, top1, top5 = 0, 0.0, 0, 0
    n_bins = 15
    bin_conf = np.zeros(n_bins); bin_acc = np.zeros(n_bins); bin_n = np.zeros(n_bins)
    for batch in loader:
        x, y = batch_to_device(batch)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item()) * x.size(0)
        tot += x.size(0)
        k = min(5, logits.size(1))
        conf, pred = logits.softmax(dim=1).topk(k, dim=1)
        correct = pred.eq(y.view(-1, 1))
        top1 += int(correct[:, 0].sum().item())
        top5 += int(correct.any(dim=1).sum().item())
        c = conf[:, 0].cpu().numpy(); ok = correct[:, 0].cpu().numpy().astype(float)
        idx = np.clip((c * n_bins).astype(int), 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.any():
                bin_conf[b] += c[m].sum(); bin_acc[b] += ok[m].sum(); bin_n[b] += m.sum()
    ece = float(np.sum(np.abs(bin_acc - bin_conf)[bin_n > 0]) / max(tot, 1))
    return 100.0 * top1 / max(tot, 1), {
        "top5": 100.0 * top5 / max(tot, 1),
        "test_loss": loss_sum / max(tot, 1),
        "ece": ece,   # expected calibration error (15 bins), descriptive
    }
