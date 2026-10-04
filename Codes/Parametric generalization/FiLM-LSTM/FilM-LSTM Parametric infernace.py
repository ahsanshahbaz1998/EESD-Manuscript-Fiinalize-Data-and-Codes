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
        return torch.load(path, map_location=map_location)
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
        title="Select trained FiLM-LSTM model checkpoint",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return path or ""


def pick_many_csv():
    root = tk_root()
    files = list(filedialog.askopenfilenames(
        parent=root,
        title="Select test CSV files",
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


def is_state_dict_like(obj):
    if not isinstance(obj, (dict, OrderedDict)):
        return False

    has_tensor = any(isinstance(v, torch.Tensor) for v in obj.values())
    has_key = any(
        isinstance(k, str) and ("lstm." in k or "film." in k or "head" in k)
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
        "CA_DIM": 2,
        "HIDDEN": hidden,
        "LAYERS": max(layer_ids) + 1 if layer_ids else 1,
        "DROPOUT": 0.0,
    }


def extract_state_cfg_norm(ckpt):
    if is_state_dict_like(ckpt):
        return ckpt, infer_cfg_from_state(ckpt), None

    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format.")

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


def norm_colname(s: str):
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("-", "_")
    s = s.replace("/", "_")
    s = s.replace("(", "")
    s = s.replace(")", "")
    return s


def find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    norm_map = {norm_colname(c): c for c in cols}

    for cand in candidates:
        key = norm_colname(cand)

        if key in norm_map:
            return norm_map[key]

    return None


def read_case_csv(path: str, default_C=C0_VALUE, default_a=ALPHA0_VALUE):
    df = pd.read_csv(path)

    if df.shape[1] < 4:
        raise ValueError(f"CSV needs at least 4 columns. Got {df.shape[1]}.")

    cols = list(df.columns)
    first_col = str(cols[0]).strip().lower() if cols else ""
    has_unnamed_index = len(cols) >= 5 and first_col.startswith("unnamed")
    start = 1 if has_unnamed_index else 0

    c_t = find_col(df, ["time_sec", "time"])
    c_D = find_col(df, ["D_mm", "D1_mm", "D1", "D"])
    c_V = find_col(df, ["V_mps", "V1_mps", "V1", "V"])
    c_F = find_col(df, ["F_kN", "F_case_kN", "F_case", "Fcase_kN", "F"])

    c_C = find_col(df, ["C", "c", "C_used", "C_case"])
    c_a = find_col(df, ["alpha", "a", "alpha_used", "alpha_case"])

    if any(c is None for c in [c_t, c_D, c_V, c_F]):
        if df.shape[1] - start < 4:
            raise ValueError("Missing required columns and not enough columns for positional fallback.")

        arr4 = df.iloc[:, start:start + 4].to_numpy(np.float64)

        t = arr4[:, 0]
        D = arr4[:, 1]
        V = arr4[:, 2]
        F_true = arr4[:, 3]

    else:
        t = pd.to_numeric(df[c_t], errors="coerce").to_numpy(np.float64)
        D = pd.to_numeric(df[c_D], errors="coerce").to_numpy(np.float64)
        V = pd.to_numeric(df[c_V], errors="coerce").to_numpy(np.float64)
        F_true = pd.to_numeric(df[c_F], errors="coerce").to_numpy(np.float64)

    m = (
        np.isfinite(t) &
        np.isfinite(D) &
        np.isfinite(V) &
        np.isfinite(F_true)
    )

    if m.sum() < 20:
        raise ValueError("Too few finite rows.")

    t = t[m]
    D = D[m]
    V = V[m]
    F_true = F_true[m]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)

        t = t[idx]
        D = D[idx]
        V = V[idx]
        F_true = F_true[idx]

    C_arr = None
    a_arr = None

    if c_C is not None:
        try:
            C_tmp = pd.to_numeric(df[c_C], errors="coerce").to_numpy(np.float64)[m]

            if np.isfinite(C_tmp).sum() >= 10:
                C_arr = C_tmp

        except Exception:
            C_arr = None

    if c_a is not None:
        try:
            a_tmp = pd.to_numeric(df[c_a], errors="coerce").to_numpy(np.float64)[m]

            if np.isfinite(a_tmp).sum() >= 10:
                a_arr = a_tmp

        except Exception:
            a_arr = None

    if (C_arr is None or a_arr is None) and (df.shape[1] - start >= 6):
        try:
            arr6 = df.iloc[:, start:start + 6].to_numpy(np.float64)

            C_pos = arr6[:, 1][m]
            a_pos = arr6[:, 2][m]

            if C_arr is None and np.isfinite(C_pos).sum() >= 10:
                C_arr = C_pos

            if a_arr is None and np.isfinite(a_pos).sum() >= 10:
                a_arr = a_pos

        except Exception:
            pass

    C_scalar = float(default_C) if C_arr is None else float(np.nanmedian(C_arr))
    a_scalar = float(default_a) if a_arr is None else float(np.nanmedian(a_arr))

    return (
        t.astype(np.float64),
        D.astype(np.float32),
        V.astype(np.float32),
        F_true.astype(np.float32),
        np.float32(C_scalar),
        np.float32(a_scalar),
    )


def normalize(A, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), EPS)

    return ((A - mean) / std).astype(np.float32)


def denormalize(A, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), EPS)

    return (A * std + mean).astype(np.float32)


def initial_transient_index(t, transient_sec=INITIAL_TRANSIENT_SEC):
    i = int(np.searchsorted(np.asarray(t), float(transient_sec), side="left"))
    return max(0, min(i, len(t) - 1))


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


class FiLMSingleDamperLSTM(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, ca_dim=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden),
            num_layers=int(layers),
            dropout=float(dropout) if int(layers) > 1 else 0.0,
            batch_first=True,
        )

        self.film = nn.Sequential(
            nn.Linear(int(ca_dim), 64),
            nn.ReLU(),
            nn.Linear(64, 2 * int(hidden)),
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

    def forward(self, x, ca_norm):
        h, _ = self.lstm(x)

        gb = self.film(ca_norm)
        gamma, beta = torch.chunk(gb, 2, dim=1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        h = gamma * h + beta

        f = self.headF(h)
        v = self.headV(h)

        y = torch.cat([f, v], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


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
        raise ValueError("Checkpoint missing norm. Need X_mean, X_std, Y_mean, Y_std, CA_mean, CA_std.")

    for key in ["X_mean", "X_std", "Y_mean", "Y_std", "CA_mean", "CA_std"]:
        if key not in norm:
            raise ValueError(f"norm missing {key}.")

    X_mean = np.asarray(norm["X_mean"], dtype=np.float32).reshape(-1)
    X_std = np.asarray(norm["X_std"], dtype=np.float32).reshape(-1)

    Y_mean = np.asarray(norm["Y_mean"], dtype=np.float32).reshape(-1)
    Y_std = np.asarray(norm["Y_std"], dtype=np.float32).reshape(-1)

    CA_mean = np.asarray(norm["CA_mean"], dtype=np.float32).reshape(-1)
    CA_std = np.asarray(norm["CA_std"], dtype=np.float32).reshape(-1)

    in_dim = int(X_mean.shape[0])
    out_dim = int(Y_mean.shape[0])
    ca_dim = int(CA_mean.shape[0])

    if in_dim != 1:
        raise ValueError(f"This S2 FiLM inference code requires IN_DIM = 1. Found {in_dim}.")

    if out_dim != 2:
        raise ValueError(f"This S2 FiLM inference code requires OUT_DIM = 2. Found {out_dim}.")

    if ca_dim != 2:
        raise ValueError(f"This S2 FiLM inference code requires CA_DIM = 2. Found {ca_dim}.")

    inferred = infer_cfg_from_state(state)

    hidden = int(cfg.get("HIDDEN", inferred["HIDDEN"]))
    layers = int(cfg.get("LAYERS", inferred["LAYERS"]))
    dropout = float(cfg.get("DROPOUT", 0.05))

    model = FiLMSingleDamperLSTM(
        in_dim=in_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        ca_dim=ca_dim,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []

    for fp in files:
        fp = Path(fp)

        try:
            t, D, V_true, F_true, C_scalar, alpha_scalar = read_case_csv(str(fp))
        except Exception as exc:
            print(f"Skipped {fp.name}: {exc}")
            continue

        X = D.reshape(-1, 1).astype(np.float32)
        CA = np.array([C_scalar, alpha_scalar], dtype=np.float32)

        X_norm = normalize(X, X_mean, X_std)
        CA_norm = normalize(CA, CA_mean, CA_std)

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_norm).unsqueeze(0).to(device)
            CA_tensor = torch.from_numpy(CA_norm).unsqueeze(0).to(device)
            Y_norm = model(X_tensor, CA_tensor).squeeze(0).cpu().numpy()

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