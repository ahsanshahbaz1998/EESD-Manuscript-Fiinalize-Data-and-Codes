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

EPS = 1e-12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str, map_location="cpu"):
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
        title="Select trained DC-LSTM model checkpoint",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return path or ""


def pick_test_files():
    root = tk_root()
    files = list(filedialog.askopenfilenames(
        parent=root,
        title="Select 6-column test CSV files",
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


def infer_cfg_from_state(state_dict):
    w = state_dict.get("lstm.weight_ih_l0", None)

    if w is None:
        raise ValueError("Cannot infer model configuration: missing lstm.weight_ih_l0.")

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
        return ckpt, infer_cfg_from_state(ckpt), None

    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format.")

    state = None

    for key in ["state", "model_state_dict", "state_dict", "model", "net"]:
        if key in ckpt and is_state_dict(ckpt[key]):
            state = ckpt[key]
            break

    if state is None:
        raise ValueError("Model state dictionary was not found in checkpoint.")

    cfg = ckpt.get("cfg", None)

    if not isinstance(cfg, dict):
        cfg = infer_cfg_from_state(state)

    norm = None

    for key in ["norm", "normalizer", "stats"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            norm = ckpt[key]
            break

    return state, cfg, norm


def normalize_name(s: str):
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("-", "_")
    s = s.replace("(", "")
    s = s.replace(")", "")
    s = s.replace("/", "_")
    return s


def find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    norm_map = {normalize_name(c): c for c in cols}

    for cand in candidates:
        key = normalize_name(cand)

        if key in norm_map:
            return norm_map[key]

    return None


def read_6col_file(path: str):
    df = pd.read_csv(path)

    if df.shape[1] < 6:
        raise ValueError(f"CSV must have at least 6 columns. Got {df.shape[1]} columns.")

    c_t = find_col(df, ["time_sec", "time"])
    c_C = find_col(df, ["C", "c", "C_used", "C_case"])
    c_a = find_col(df, ["alpha", "Alpha", "a", "alpha_used", "alpha_case"])
    c_D = find_col(df, ["D_mm", "D1_mm", "D1", "D"])
    c_V = find_col(df, ["V_mps", "V1_mps", "V1", "V"])
    c_F = find_col(df, ["F_kN", "F_case_kN", "F_case", "Fcase_kN", "F"])

    if any(c is None for c in [c_t, c_C, c_a, c_D, c_V, c_F]):
        arr = df.iloc[:, :6].to_numpy(np.float64)

        t = arr[:, 0]
        C = arr[:, 1]
        alpha = arr[:, 2]
        D = arr[:, 3]
        V = arr[:, 4]
        F_true = arr[:, 5]

    else:
        t = pd.to_numeric(df[c_t], errors="coerce").to_numpy(np.float64)
        C = pd.to_numeric(df[c_C], errors="coerce").to_numpy(np.float64)
        alpha = pd.to_numeric(df[c_a], errors="coerce").to_numpy(np.float64)
        D = pd.to_numeric(df[c_D], errors="coerce").to_numpy(np.float64)
        V = pd.to_numeric(df[c_V], errors="coerce").to_numpy(np.float64)
        F_true = pd.to_numeric(df[c_F], errors="coerce").to_numpy(np.float64)

    mask = (
        np.isfinite(t) &
        np.isfinite(C) &
        np.isfinite(alpha) &
        np.isfinite(D) &
        np.isfinite(V) &
        np.isfinite(F_true)
    )

    if mask.sum() < 20:
        raise ValueError("Too few finite rows.")

    t = t[mask]
    C = C[mask]
    alpha = alpha[mask]
    D = D[mask]
    V = V[mask]
    F_true = F_true[mask]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)

        t = t[idx]
        C = C[idx]
        alpha = alpha[idx]
        D = D[idx]
        V = V[idx]
        F_true = F_true[idx]

    return (
        t.astype(np.float64),
        D.astype(np.float32),
        V.astype(np.float32),
        F_true.astype(np.float32),
        C.astype(np.float32),
        alpha.astype(np.float32),
    )


def normalize(A, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), EPS)

    return ((A - mean) / std).astype(np.float32)


def denormalize(A, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), EPS)

    return (A * std + mean).astype(np.float32)


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
            nn.Linear(int(hidden), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.headV = nn.Sequential(
            nn.Linear(int(hidden), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)

        Fp = self.headF(h)
        Vp = self.headV(h)

        y = torch.cat([Fp, Vp], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


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


def initial_transient_index(t, transient_sec=INITIAL_TRANSIENT_SEC):
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

    files = pick_test_files()

    if not files:
        return

    out_dir = pick_output_dir()

    if not out_dir:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(model_path, map_location="cpu")
    state, cfg, norm = extract_checkpoint_parts(ckpt)

    if norm is None:
        raise ValueError("Checkpoint missing normalization statistics.")

    for key in ["X_mean", "X_std", "Y_mean", "Y_std"]:
        if key not in norm:
            raise ValueError(f"Missing normalization value: {key}")

    X_mean = np.asarray(norm["X_mean"], dtype=np.float32).reshape(-1)
    X_std = np.asarray(norm["X_std"], dtype=np.float32).reshape(-1)

    Y_mean = np.asarray(norm["Y_mean"], dtype=np.float32).reshape(-1)
    Y_std = np.asarray(norm["Y_std"], dtype=np.float32).reshape(-1)

    if len(X_mean) != 3:
        raise ValueError("This inference code requires IN_DIM = 3: [D, C, alpha].")

    if len(Y_mean) != 2:
        raise ValueError("This inference code requires OUT_DIM = 2: [F, V].")

    model = SingleDamperLSTM(
        in_dim=3,
        hidden=int(cfg.get("HIDDEN", 128)),
        layers=int(cfg.get("LAYERS", 3)),
        dropout=float(cfg.get("DROPOUT", 0.05)),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []

    for file_path in files:
        file_path = Path(file_path)

        try:
            t, D, V_true, F_true, C_seq, alpha_seq = read_6col_file(str(file_path))
        except Exception as exc:
            print(f"Skipped {file_path.name}: {exc}")
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
            F_pred = clamp_initial_transient(F_pred, i0)
            V_pred = clamp_initial_transient(V_pred, i0)

        metric_mask = mask_after if IGNORE_INITIAL_TRANSIENT_FOR_METRICS else None

        rows.append({
            "file": file_path.name,
            "r_F": pearson_r(F_true, F_pred, mask=metric_mask),
            "r_V": pearson_r(V_true, V_pred, mask=metric_mask),
        })

        print(f"{file_path.name}: evaluation completed.")

    if not rows:
        print("No files were evaluated.")
        return

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:", summary_path)


if __name__ == "__main__":
    main()