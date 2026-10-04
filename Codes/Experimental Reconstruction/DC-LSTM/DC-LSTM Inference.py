import re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tkinter import Tk, filedialog


C_VALUE = 696.0
ALPHA_VALUE = 0.44

INITIAL_TRANSIENT_SEC = 0.20
IGNORE_INITIAL_TRANSIENT_FOR_METRICS = True
APPLY_INITIAL_TRANSIENT_CLAMP = True

EPS = 1e-12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def tk_root():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def pick_model():
    root = tk_root()
    path = filedialog.askopenfilename(
        parent=root,
        title="Select trained DC-LSTM checkpoint",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return path or ""


def pick_test_files():
    root = tk_root()
    files = list(filedialog.askopenfilenames(
        parent=root,
        title="Select test CSV files",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
    ))
    root.destroy()
    return files


def pick_output_dir():
    root = tk_root()
    path = filedialog.askdirectory(parent=root, title="Select output folder")
    root.destroy()
    return path or ""


def is_state_dict(obj):
    if not isinstance(obj, (dict, OrderedDict)):
        return False

    has_tensor = any(isinstance(v, torch.Tensor) for v in obj.values())
    has_model_key = any(
        isinstance(k, str) and ("lstm." in k or "head" in k)
        for k in obj.keys()
    )

    return has_tensor and has_model_key


def infer_cfg(state_dict):
    w = state_dict.get("lstm.weight_ih_l0", None)

    if w is None:
        raise ValueError("Checkpoint does not contain lstm.weight_ih_l0.")

    hidden = int(w.shape[0] // 4)
    in_dim = int(w.shape[1])

    layers = []

    for k in state_dict.keys():
        m = re.match(r"lstm\.weight_ih_l(\d+)$", k)

        if m:
            layers.append(int(m.group(1)))

    return {
        "IN_DIM": in_dim,
        "OUT_DIM": 2,
        "HIDDEN": hidden,
        "LAYERS": max(layers) + 1 if layers else 1,
        "DROPOUT": 0.0,
    }


def extract_checkpoint_parts(ckpt):
    if is_state_dict(ckpt):
        return ckpt, infer_cfg(ckpt), None

    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format.")

    state = None

    for key in ["state", "model_state_dict", "state_dict", "model", "net"]:
        if key in ckpt and is_state_dict(ckpt[key]):
            state = ckpt[key]
            break

    if state is None:
        raise ValueError("Model state dictionary was not found in the checkpoint.")

    cfg = ckpt.get("cfg", None)

    if not isinstance(cfg, dict):
        cfg = infer_cfg(state)

    norm = None

    for key in ["norm", "normalizer", "stats"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            norm = ckpt[key]
            break

    return state, cfg, norm


def read_floor1_csv(path):
    df = pd.read_csv(path)

    if df.shape[1] < 10:
        raise ValueError(f"{Path(path).name}: expected at least 10 columns.")

    arr = df.iloc[:, :10].to_numpy(np.float64)

    t = arr[:, 0]
    d1 = arr[:, 1]
    f1 = arr[:, 4]
    v1 = arr[:, 7]

    mask = (
        np.isfinite(t) &
        np.isfinite(d1) &
        np.isfinite(f1) &
        np.isfinite(v1)
    )

    if mask.sum() < 20:
        raise ValueError(f"{Path(path).name}: fewer than 20 valid rows.")

    t = t[mask]
    d1 = d1[mask]
    f1 = f1[mask]
    v1 = v1[mask]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)

        t = t[idx]
        d1 = d1[idx]
        f1 = f1[idx]
        v1 = v1[idx]

    return (
        t.astype(np.float64),
        d1.astype(np.float32),
        f1.astype(np.float32),
        v1.astype(np.float32),
    )


def normalize(x, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), EPS)

    return ((x - mean) / std).astype(np.float32)


def denormalize(x, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), EPS)

    return (x * std + mean).astype(np.float32)


class SingleDamperLSTM(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden),
            num_layers=int(layers),
            dropout=float(dropout) if int(layers) > 1 else 0.0,
            batch_first=True,
        )

        self.headF = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.headV = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)

        f = self.headF(h)
        v = self.headV(h)

        return torch.nan_to_num(torch.cat([f, v], dim=-1))


def valid_pair(a, b, mask=None):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    m = np.isfinite(a) & np.isfinite(b)

    if mask is not None:
        m = m & np.asarray(mask).ravel().astype(bool)

    return a[m], b[m]


def pearson_r(a, b, mask=None):
    a, b = valid_pair(a, b, mask)

    if len(a) < 3:
        return np.nan

    a0 = a - a.mean()
    b0 = b - b.mean()

    den = np.sqrt((a0 * a0).sum()) * np.sqrt((b0 * b0).sum())

    if den <= 0:
        return np.nan

    return float((a0 * b0).sum() / den)


def transient_index(t, transient_sec=INITIAL_TRANSIENT_SEC):
    i = int(np.searchsorted(np.asarray(t), float(transient_sec), side="left"))
    return max(0, min(i, len(t) - 1))


def clamp_initial_transient(pred, i0):
    pred = np.asarray(pred).copy()

    if pred.size == 0:
        return pred

    i0 = int(np.clip(i0, 0, pred.size - 1))

    if i0 > 0:
        pred[:i0] = pred[i0]

    return pred


def main():
    model_path = pick_model()

    if not model_path:
        return

    test_files = pick_test_files()

    if not test_files:
        return

    out_dir = pick_output_dir()

    if not out_dir:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(model_path, map_location="cpu")
    state, cfg, norm = extract_checkpoint_parts(ckpt)

    if norm is None:
        raise ValueError("Checkpoint does not contain normalization statistics.")

    for key in ["X_mean", "X_std", "Y_mean", "Y_std"]:
        if key not in norm:
            raise ValueError(f"Missing normalization value: {key}")

    x_mean = np.asarray(norm["X_mean"], dtype=np.float32).reshape(-1)
    x_std = np.asarray(norm["X_std"], dtype=np.float32).reshape(-1)

    y_mean = np.asarray(norm["Y_mean"], dtype=np.float32).reshape(-1)
    y_std = np.asarray(norm["Y_std"], dtype=np.float32).reshape(-1)

    if len(x_mean) != 3:
        raise ValueError("The checkpoint input dimension must be 3: [D, C, alpha].")

    if len(y_mean) != 2:
        raise ValueError("The checkpoint output dimension must be 2: [F, V].")

    model = SingleDamperLSTM(
        in_dim=len(x_mean),
        hidden=int(cfg.get("HIDDEN", 128)),
        layers=int(cfg.get("LAYERS", 3)),
        dropout=float(cfg.get("DROPOUT", 0.05)),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []

    for file_path in test_files:
        file_path = Path(file_path)

        try:
            t, d1, f_true, v_true = read_floor1_csv(file_path)
        except Exception as exc:
            print(f"Skipped {file_path.name}: {exc}")
            continue

        i0 = transient_index(t)
        mask_after = t >= float(INITIAL_TRANSIENT_SEC)
        metric_mask = mask_after if IGNORE_INITIAL_TRANSIENT_FOR_METRICS else None

        c = np.full_like(d1, C_VALUE, dtype=np.float32)
        alpha = np.full_like(d1, ALPHA_VALUE, dtype=np.float32)

        x = np.column_stack([d1, c, alpha]).astype(np.float32)
        x_norm = normalize(x, x_mean, x_std)

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
            y_norm = model(x_tensor).squeeze(0).cpu().numpy()

        y_pred = denormalize(y_norm, y_mean, y_std)

        f_pred = y_pred[:, 0].copy()
        v_pred = y_pred[:, 1].copy()

        if APPLY_INITIAL_TRANSIENT_CLAMP:
            f_pred = clamp_initial_transient(f_pred, i0)
            v_pred = clamp_initial_transient(v_pred, i0)

        rows.append({
            "file": file_path.name,
            "r_F": pearson_r(f_true, f_pred, mask=metric_mask),
            "r_V": pearson_r(v_true, v_pred, mask=metric_mask),
        })

        print(f"{file_path.name}: evaluation completed.")

    if not rows:
        print("No files were evaluated.")
        return

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "evaluation_summary_floor1.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()