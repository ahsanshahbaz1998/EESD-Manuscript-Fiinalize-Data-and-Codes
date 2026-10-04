# -*- coding: utf-8 -*-

import re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tkinter import Tk, filedialog


INITIAL_TRANSIENT_SEC = 0.20
IGNORE_INITIAL_TRANSIENT_FOR_METRICS = True
APPLY_INITIAL_TRANSIENT_CLAMP = True

C0_VALUE = 696.0
ALPHA0_VALUE = 0.44

EPS = 1e-12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str, map_location="cpu"):
    try:
        import numpy as _np
        from torch.serialization import safe_globals

        allow = [
            _np.core.multiarray._reconstruct,
            _np.ndarray,
            _np.dtype,
        ]

        with safe_globals(allow):
            return torch.load(path, map_location=map_location, weights_only=False)

    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def tk_root():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def pick_model_pt():
    root = tk_root()

    path = filedialog.askopenfilename(
        parent=root,
        title="Select trained PhyLSTM model checkpoint",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )

    root.destroy()
    return path or ""


def pick_many_csv():
    root = tk_root()

    files = list(filedialog.askopenfilenames(
        parent=root,
        title="Select test case CSV files",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
    ))

    root.destroy()
    return files


def pick_out_dir():
    root = tk_root()

    path = filedialog.askdirectory(
        parent=root,
        title="Select output folder"
    )

    root.destroy()
    return path or ""


def find_col(df: pd.DataFrame, candidates):
    cols = [str(c) for c in df.columns]

    direct = {c: c for c in cols}

    for cand in candidates:
        if cand in direct:
            return cand

    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        key = cand.lower()

        if key in lower_map:
            return lower_map[key]

    return None


def read_case_csv_floor1(path: str, default_C=C0_VALUE, default_a=ALPHA0_VALUE):
    df = pd.read_csv(path)

    if df.shape[1] < 4:
        raise ValueError(f"CSV needs at least 4 columns. Got {df.shape[1]}.")

    cols = list(df.columns)
    first_col = str(cols[0]).strip().lower() if cols else ""
    has_unnamed_index = len(cols) >= 5 and first_col.startswith("unnamed")

    c_t = find_col(df, ["time_sec", "time"])
    c_D = find_col(df, ["D1_mm", "D1", "D_mm", "D"])
    c_V = find_col(df, ["V1_mps", "V1_mps_used", "V1", "V_mps", "V"])
    c_F = find_col(df, ["F_case_kN", "F_case", "Fcase_kN", "Fcase", "F_case_kN_used", "F_kN", "F"])

    c_C = find_col(df, ["C_used", "C", "C_case", "c"])
    c_a = find_col(df, ["alpha_used", "alpha", "alpha_case", "a_used", "a"])

    if c_t is None or c_D is None or c_V is None or c_F is None:
        start = 1 if has_unnamed_index else 0

        if df.shape[1] - start < 4:
            raise ValueError("Missing required columns and not enough columns for positional fallback.")

        arr = df.iloc[:, start:start + 4].to_numpy(np.float64)

        t = arr[:, 0]
        D = arr[:, 1]
        V = arr[:, 2]
        F_true = arr[:, 3]

    else:
        t = pd.to_numeric(df[c_t], errors="coerce").to_numpy(np.float64)
        D = pd.to_numeric(df[c_D], errors="coerce").to_numpy(np.float64)
        V = pd.to_numeric(df[c_V], errors="coerce").to_numpy(np.float64)
        F_true = pd.to_numeric(df[c_F], errors="coerce").to_numpy(np.float64)

    mask = (
        np.isfinite(t) &
        np.isfinite(D) &
        np.isfinite(V) &
        np.isfinite(F_true)
    )

    if mask.sum() < 20:
        raise ValueError("Too few finite rows.")

    t = t[mask]
    D = D[mask]
    V = V[mask]
    F_true = F_true[mask]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)

        t = t[idx]
        D = D[idx]
        V = V[idx]
        F_true = F_true[idx]

    n = len(t)

    C_seq = None
    alpha_seq = None

    if c_C is not None:
        try:
            C_col = pd.to_numeric(df[c_C], errors="coerce").to_numpy(np.float64)[mask]

            if np.isfinite(C_col).sum() >= 10:
                C_seq = C_col.astype(np.float32)

        except Exception:
            C_seq = None

    if c_a is not None:
        try:
            alpha_col = pd.to_numeric(df[c_a], errors="coerce").to_numpy(np.float64)[mask]

            if np.isfinite(alpha_col).sum() >= 10:
                alpha_seq = alpha_col.astype(np.float32)

        except Exception:
            alpha_seq = None

    if (C_seq is None or alpha_seq is None) and df.shape[1] >= 6:
        start = 1 if has_unnamed_index else 0

        try:
            arr6 = df.iloc[:, start:start + 6].to_numpy(np.float64)

            C_pos = arr6[:, 1][mask]
            alpha_pos = arr6[:, 2][mask]

            if C_seq is None and np.isfinite(C_pos).sum() >= 10:
                C_seq = C_pos.astype(np.float32)

            if alpha_seq is None and np.isfinite(alpha_pos).sum() >= 10:
                alpha_seq = alpha_pos.astype(np.float32)

        except Exception:
            pass

    if C_seq is None:
        C_seq = np.full((n,), float(default_C), dtype=np.float32)

    if alpha_seq is None:
        alpha_seq = np.full((n,), float(default_a), dtype=np.float32)

    return (
        t.astype(np.float64),
        D.astype(np.float32),
        V.astype(np.float32),
        F_true.astype(np.float32),
        C_seq.astype(np.float32),
        alpha_seq.astype(np.float32),
    )


def normalize(A, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.maximum(std, EPS)

    return ((A - mean) / std).astype(np.float32)


def denormalize(A, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.maximum(std, EPS)

    return (A * std + mean).astype(np.float32)


def initial_transient_index(t, transient_sec=INITIAL_TRANSIENT_SEC):
    t = np.asarray(t)

    i = int(np.searchsorted(t, float(transient_sec), side="left"))
    i = max(0, min(i, len(t) - 1))

    return i


def apply_initial_transient_clamp(pred, i0):
    pred = np.asarray(pred).copy()

    if pred.size == 0:
        return pred

    i0 = int(np.clip(i0, 0, pred.size - 1))

    if i0 > 0:
        pred[:i0] = pred[i0]

    return pred


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


def inverse_softplus(y: float):
    y = max(float(y), 1e-12)
    return float(np.log(np.expm1(y)))


class SingleDamperLSTM_HV(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, ks_init=1.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden),
            num_layers=int(layers),
            dropout=float(dropout) if int(layers) > 1 else 0.0,
            batch_first=True,
        )

        self.headF = nn.Sequential(
            nn.Linear(int(hidden), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.headV = nn.Sequential(
            nn.Linear(int(hidden), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._ks_raw = nn.Parameter(
            torch.tensor(inverse_softplus(ks_init), dtype=torch.float32)
        )

    def forward(self, x):
        h, _ = self.lstm(x)

        f = self.headF(h)
        v = self.headV(h)

        y = torch.cat([f, v], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def is_state_dict_like(obj):
    if not isinstance(obj, (dict, OrderedDict)):
        return False

    has_tensor = any(isinstance(v, torch.Tensor) for v in obj.values())

    has_key = any(
        isinstance(k, str) and ("lstm." in k or "head" in k)
        for k in obj.keys()
    )

    return has_tensor and has_key


def infer_cfg_from_state(state_dict):
    w = state_dict.get("lstm.weight_ih_l0", None)

    if w is None:
        raise ValueError("Cannot infer configuration: missing lstm.weight_ih_l0.")

    hidden = int(w.shape[0] // 4)
    in_dim = int(w.shape[1])

    layer_ids = []

    for k in state_dict.keys():
        m = re.match(r"lstm\.weight_ih_l(\d+)$", k)

        if m:
            layer_ids.append(int(m.group(1)))

    return {
        "IN_DIM": in_dim,
        "OUT_DIM": 2,
        "HIDDEN": hidden,
        "LAYERS": max(layer_ids) + 1 if layer_ids else 1,
        "DROPOUT": 0.0,
    }


def extract_state_cfg_norm(ckpt):
    if is_state_dict_like(ckpt):
        return ckpt, infer_cfg_from_state(ckpt), None

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a supported dictionary format.")

    state = None

    for key in ["state", "model_state_dict", "state_dict", "model", "net"]:
        if key in ckpt and is_state_dict_like(ckpt[key]):
            state = ckpt[key]
            break

    if state is None:
        raise ValueError("Could not find model state dictionary in checkpoint.")

    cfg = ckpt.get("cfg", None)

    if not isinstance(cfg, dict):
        cfg = infer_cfg_from_state(state)

    norm = None

    for key in ["norm", "normalizer", "stats"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            norm = ckpt[key]
            break

    return state, cfg, norm


def main():
    model_path = pick_model_pt()

    if not model_path:
        return

    files = pick_many_csv()

    if not files:
        return

    out_dir = pick_out_dir()

    if not out_dir:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = safe_torch_load(model_path, map_location="cpu")
    state, cfg, norm = extract_state_cfg_norm(ckpt)

    if norm is None:
        raise ValueError("Checkpoint does not contain normalization statistics.")

    for key in ["X_mean", "X_std", "Y_mean", "Y_std"]:
        if key not in norm:
            raise ValueError(f"Normalization dictionary missing {key}.")

    X_mean = np.asarray(norm["X_mean"], dtype=np.float32).reshape(-1)
    X_std = np.asarray(norm["X_std"], dtype=np.float32).reshape(-1)

    Y_mean = np.asarray(norm["Y_mean"], dtype=np.float32).reshape(-1)
    Y_std = np.asarray(norm["Y_std"], dtype=np.float32).reshape(-1)

    in_dim = int(cfg.get("IN_DIM", 3))
    out_dim = int(cfg.get("OUT_DIM", 2))
    hidden = int(cfg.get("HIDDEN", 128))
    layers = int(cfg.get("LAYERS", 3))
    dropout = float(cfg.get("DROPOUT", 0.05))

    if in_dim != 3:
        raise ValueError(f"This inference code requires IN_DIM = 3. Found {in_dim}.")

    if out_dim != 2:
        raise ValueError(f"This inference code requires OUT_DIM = 2. Found {out_dim}.")

    model = SingleDamperLSTM_HV(
        in_dim=in_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        ks_init=1.0,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []

    for fp in files:
        fp = Path(fp)

        try:
            t, D, V_true, F_true, C_seq, alpha_seq = read_case_csv_floor1(
                str(fp),
                default_C=C0_VALUE,
                default_a=ALPHA0_VALUE
            )

        except Exception as exc:
            print(f"Skipped {fp.name}: {exc}")
            continue

        X = np.column_stack([D, C_seq, alpha_seq]).astype(np.float32)
        X_norm = normalize(X, X_mean, X_std)

        with torch.no_grad():
            Xt = torch.from_numpy(X_norm).unsqueeze(0).to(device)
            Y_norm = model(Xt).squeeze(0).cpu().numpy()

        Y_pred = denormalize(Y_norm, Y_mean, Y_std)

        F_pred = Y_pred[:, 0].copy()
        V_pred = Y_pred[:, 1].copy()

        i0 = initial_transient_index(t)
        mask_after = t >= float(INITIAL_TRANSIENT_SEC)

        if APPLY_INITIAL_TRANSIENT_CLAMP:
            F_pred = apply_initial_transient_clamp(F_pred, i0)
            V_pred = apply_initial_transient_clamp(V_pred, i0)

        metric_mask = mask_after if IGNORE_INITIAL_TRANSIENT_FOR_METRICS else None

        rows.append({
            "file": fp.name,
            "r_F": pearson_r(F_true, F_pred, mask=metric_mask),
            "r_V": pearson_r(V_true, V_pred, mask=metric_mask),
        })

        print(f"{fp.name}: evaluation completed.")

    if not rows:
        print("No files processed successfully.")
        return

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:", summary_path)


if __name__ == "__main__":
    main()