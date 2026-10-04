# -*- coding: utf-8 -*-

import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tkinter import Tk, filedialog, messagebox


C0_VALUE = 696.0
C_MULTS = [0.5, 1.25]
A_CASES = [0.30, 0.80]

VAL_FRAC = 0.20
SEED = 42

IN_DIM = 1
CA_DIM = 2
OUT_DIM = 2

HIDDEN = 128
LAYERS = 3
DROPOUT = 0.05

LR = 3e-3
WD = 3e-4
BATCH = 16
EPOCHS = 2000
PATIENCE = 300
GRAD_CLIP = 1.0

W_FORCE = 1.0
W_VEL = 1.0

EPS = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def _tk_root():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def pick_many_csv(title):
    root = _tk_root()

    files = list(filedialog.askopenfilenames(
        parent=root,
        title=title,
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
    ))

    root.destroy()
    return files


def save_model_path():
    root = _tk_root()

    p = filedialog.asksaveasfilename(
        parent=root,
        title="Save trained model (.pt)",
        defaultextension=".pt",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )

    root.destroy()
    return p or ""


def build_cases_4(C0=C0_VALUE):
    cases = []

    for a in A_CASES:
        for mC in C_MULTS:
            Cc = float(C0) * float(mC)
            aa = float(a)

            short = f"C{int(round(Cc))}_a{int(round(aa * 100)):02d}"
            lab = f"C{Cc:.1f}_a{aa:.2f}"

            cases.append((Cc, aa, short, lab))

    assert len(cases) == 4
    return cases


CASES4 = build_cases_4()


def extract_base_id_from_stem(stem: str) -> str:
    m = re.search(r"^(.*?)__floor\d+__", stem, flags=re.IGNORECASE)

    if m:
        return m.group(1)

    s = re.sub(r"__C\d+_a\d+.*$", "", stem, flags=re.IGNORECASE)
    return s


def split_by_base_id(records, val_frac=VAL_FRAC, seed=SEED):
    base_ids = sorted(set(r["base_id"] for r in records))

    if len(base_ids) <= 1:
        return records, []

    rng = np.random.RandomState(seed)

    n_val = max(1, int(round(len(base_ids) * float(val_frac))))
    perm = rng.permutation(len(base_ids))

    val_ids = set(base_ids[i] for i in perm[:n_val])
    tr_ids = set(base_ids[i] for i in perm[n_val:])

    tr = [r for r in records if r["base_id"] in tr_ids]
    va = [r for r in records if r["base_id"] in val_ids]

    return tr, va


def n_unique_motions(rec_list):
    return len(set(r["base_id"] for r in rec_list))


def _norm_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("-", "_")
    s = s.replace("/", "_")
    s = s.replace("(", "")
    s = s.replace(")", "")
    return s


def find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    norm_map = {_norm_colname(c): c for c in cols}

    for cand in candidates:
        key = _norm_colname(cand)

        if key in norm_map:
            return norm_map[key]

    return None


def read_6col_with_CA(path: str):
    df = pd.read_csv(path)

    if df.shape[1] < 6:
        raise ValueError("Needs 6-column file with C and alpha inside the CSV.")

    cC = find_col(df, ["C", "c", "C_used", "C_case"])
    ca = find_col(df, ["alpha", "a", "alpha_used", "alpha_case"])
    cD = find_col(df, ["D_mm", "D1_mm", "D1", "D"])
    cV = find_col(df, ["V_mps", "V1_mps", "V1", "V"])
    cF = find_col(df, ["F_kN", "F_case_kN", "Fcase_kN", "F_case", "F"])

    if any(c is None for c in [cC, ca, cD, cV, cF]):
        arr = df.iloc[:, :6].to_numpy(np.float64)

        Cc = arr[:, 1]
        aa = arr[:, 2]
        D = arr[:, 3]
        V = arr[:, 4]
        Fv = arr[:, 5]

    else:
        Cc = pd.to_numeric(df[cC], errors="coerce").to_numpy(np.float64)
        aa = pd.to_numeric(df[ca], errors="coerce").to_numpy(np.float64)
        D = pd.to_numeric(df[cD], errors="coerce").to_numpy(np.float64)
        V = pd.to_numeric(df[cV], errors="coerce").to_numpy(np.float64)
        Fv = pd.to_numeric(df[cF], errors="coerce").to_numpy(np.float64)

    m = (
        np.isfinite(Cc) &
        np.isfinite(aa) &
        np.isfinite(D) &
        np.isfinite(V) &
        np.isfinite(Fv)
    )

    if m.sum() < 20:
        raise ValueError("too few finite rows (<20)")

    Cc = Cc[m]
    aa = aa[m]
    D = D[m]
    V = V[m]
    Fv = Fv[m]

    CA = np.array(
        [float(np.median(Cc)), float(np.median(aa))],
        dtype=np.float32
    )

    return (
        D.astype(np.float32),
        Fv.astype(np.float32),
        V.astype(np.float32),
        CA
    )


class FiLMCaseFileDataset(Dataset):
    def __init__(self, records):
        self.items = []
        self.skipped = []

        for r in records:
            fp = r["path"]

            try:
                D, Fv, V, CA = read_6col_with_CA(fp)
            except Exception as e:
                self.skipped.append((Path(fp).name, str(e)))
                continue

            X = D.reshape(-1, 1).astype(np.float32)
            Y = np.stack([Fv, V], axis=1).astype(np.float32)

            self.items.append((X, CA.astype(np.float32), Y))

        if len(self.items) == 0:
            msg = "No valid sequences built. This script requires 6-col files: time,C,alpha,D,V,F.\n\n"
            msg += "Skip reasons (first 10):\n"

            for nm, err in self.skipped[:10]:
                msg += f"- {nm}: {err}\n"

            raise ValueError(msg)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_fullseq(batch):
    B = len(batch)
    lengths = [len(x[0]) for x in batch]
    Lmax = max(lengths)

    Xb = torch.zeros((B, Lmax, IN_DIM), dtype=torch.float32)
    Yb = torch.zeros((B, Lmax, OUT_DIM), dtype=torch.float32)
    Mb = torch.zeros((B, Lmax, 1), dtype=torch.float32)
    CAb = torch.zeros((B, CA_DIM), dtype=torch.float32)

    for i, (X, CA, Y) in enumerate(batch):
        L = len(X)

        Xb[i, :L] = torch.from_numpy(X)
        Yb[i, :L] = torch.from_numpy(Y)
        Mb[i, :L, 0] = 1.0
        CAb[i] = torch.from_numpy(CA)

    return Xb, CAb, Yb, Mb


def streaming_mean_std(seqs, dim):
    n = 0
    mean = np.zeros((dim,), dtype=np.float64)
    M2 = np.zeros((dim,), dtype=np.float64)

    for A in seqs:
        A = np.asarray(A, dtype=np.float64)

        for row in A:
            n += 1

            delta = row - mean
            mean += delta / n

            delta2 = row - mean
            M2 += delta * delta2

    if n < 2:
        std = np.ones((dim,), dtype=np.float64)
    else:
        var = M2 / max(1, (n - 1))
        std = np.sqrt(np.maximum(var, 1e-12))

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_arr(A, mu, sd):
    sd = np.maximum(sd, 1e-12).astype(np.float32)
    return ((A - mu) / sd).astype(np.float32)


def compute_norm_from_train(ds_tr: FiLMCaseFileDataset):
    X_mu, X_sd = streaming_mean_std(
        [x for (x, _, _) in ds_tr.items],
        dim=IN_DIM
    )

    Y_mu, Y_sd = streaming_mean_std(
        [y for (_, _, y) in ds_tr.items],
        dim=OUT_DIM
    )

    CA_all = np.stack(
        [ca for (_, ca, _) in ds_tr.items],
        axis=0
    ).astype(np.float32)

    CA_mu = CA_all.mean(axis=0).astype(np.float32)
    CA_sd = (CA_all.std(axis=0) + 1e-12).astype(np.float32)

    X_sd[X_sd < 1e-12] = 1.0
    Y_sd[Y_sd < 1e-12] = 1.0
    CA_sd[CA_sd < 1e-12] = 1.0

    return {
        "X_mean": X_mu,
        "X_std": X_sd,
        "Y_mean": Y_mu,
        "Y_std": Y_sd,
        "CA_mean": CA_mu,
        "CA_std": CA_sd
    }


class FiLMSingleDamperLSTM(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden),
            num_layers=int(layers),
            dropout=float(dropout) if layers > 1 else 0.0,
            batch_first=True
        )

        self.film = nn.Sequential(
            nn.Linear(CA_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * hidden)
        )

        self.headF = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.headV = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, ca_norm):
        h, _ = self.lstm(x)

        gb = self.film(ca_norm)
        gamma, beta = torch.chunk(gb, 2, dim=1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        h_mod = gamma * h + beta

        Fp = self.headF(h_mod)
        Vp = self.headV(h_mod)

        y = torch.cat([Fp, Vp], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def train():
    all_records = []

    print("\n==================== SELECT DBE FILES (4 cases) ====================")

    for (_Cc, _aa, short, _lab) in CASES4:
        title = f"DBE: Select 6-col files for {short} (no filename checks)"
        files = pick_many_csv(title)

        if not files:
            root = _tk_root()
            messagebox.showerror(
                "Missing selection",
                f"No files selected for DBE {short}.",
                parent=root
            )
            root.destroy()
            return

        for fp in files:
            p = Path(fp)

            all_records.append(dict(
                hazard="DBE",
                case_short=short,
                path=str(p),
                stem=p.stem,
                base_id=extract_base_id_from_stem(p.stem),
            ))

    print("\n==================== SELECT MCE FILES (4 cases) ====================")

    for (_Cc, _aa, short, _lab) in CASES4:
        title = f"MCE: Select 6-col files for {short} (no filename checks)"
        files = pick_many_csv(title)

        if not files:
            root = _tk_root()
            messagebox.showerror(
                "Missing selection",
                f"No files selected for MCE {short}.",
                parent=root
            )
            root.destroy()
            return

        for fp in files:
            p = Path(fp)

            all_records.append(dict(
                hazard="MCE",
                case_short=short,
                path=str(p),
                stem=p.stem,
                base_id=extract_base_id_from_stem(p.stem),
            ))

    dbe_records = [r for r in all_records if r["hazard"] == "DBE"]
    mce_records = [r for r in all_records if r["hazard"] == "MCE"]

    dbe_tr, dbe_va = split_by_base_id(
        dbe_records,
        val_frac=VAL_FRAC,
        seed=SEED
    )

    mce_tr, mce_va = split_by_base_id(
        mce_records,
        val_frac=VAL_FRAC,
        seed=SEED
    )

    train_records = dbe_tr + mce_tr
    val_records = dbe_va + mce_va

    print("\n==================== SPLIT SUMMARY (by base_id) ====================")
    print(f"DBE: train={len(dbe_tr)} val={len(dbe_va)} | unique motions train={n_unique_motions(dbe_tr)} val={n_unique_motions(dbe_va)}")
    print(f"MCE: train={len(mce_tr)} val={len(mce_va)} | unique motions train={n_unique_motions(mce_tr)} val={n_unique_motions(mce_va)}")
    print(f"TOTAL: train={len(train_records)} val={len(val_records)}")
    print("====================================================================\n")

    print("Building training dataset.")
    ds_tr = FiLMCaseFileDataset(train_records)

    print("Computing normalization statistics from training data.")
    norm = compute_norm_from_train(ds_tr)

    X_mu = norm["X_mean"]
    X_sd = norm["X_std"]
    Y_mu = norm["Y_mean"]
    Y_sd = norm["Y_std"]
    CA_mu = norm["CA_mean"]
    CA_sd = norm["CA_std"]

    tr_items = []

    for (X, CA, Y) in ds_tr.items:
        tr_items.append((
            normalize_arr(X, X_mu, X_sd),
            normalize_arr(CA, CA_mu, CA_sd),
            normalize_arr(Y, Y_mu, Y_sd)
        ))

    if len(val_records) > 0:
        ds_va = FiLMCaseFileDataset(val_records)
        va_items = []

        for (X, CA, Y) in ds_va.items:
            va_items.append((
                normalize_arr(X, X_mu, X_sd),
                normalize_arr(CA, CA_mu, CA_sd),
                normalize_arr(Y, Y_mu, Y_sd)
            ))

    else:
        va_items = []

    tr_loader = DataLoader(
        tr_items,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collate_fullseq
    )

    va_loader = DataLoader(
        va_items,
        batch_size=BATCH,
        shuffle=False,
        collate_fn=collate_fullseq
    ) if va_items else None

    model = FiLMSingleDamperLSTM().to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD
    )

    mse = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")

    best_val = float("inf")
    patience = PATIENCE
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []

        for X, CA, Y, m in tr_loader:
            X = X.to(device)
            CA = CA.to(device)
            Y = Y.to(device)
            m = m.to(device)

            pred = model(X, CA)

            Fp = pred[:, :, 0:1]
            Vp = pred[:, :, 1:2]

            Ft = Y[:, :, 0:1]
            Vt = Y[:, :, 1:2]

            den = m.sum() + EPS

            L_force = (mse(Fp, Ft) * m).sum() / den
            L_vel = (huber(Vp, Vt) * m).sum() / den

            loss = W_FORCE * L_force + W_VEL * L_vel
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()

            tr_losses.append(float(loss.item()))

        tr_m = float(np.mean(tr_losses)) if tr_losses else float("inf")

        if va_loader is not None:
            model.eval()
            va_losses = []

            with torch.no_grad():
                for X, CA, Y, m in va_loader:
                    X = X.to(device)
                    CA = CA.to(device)
                    Y = Y.to(device)
                    m = m.to(device)

                    pred = model(X, CA)

                    Fp = pred[:, :, 0:1]
                    Vp = pred[:, :, 1:2]

                    Ft = Y[:, :, 0:1]
                    Vt = Y[:, :, 1:2]

                    den = m.sum() + EPS

                    L_force = (mse(Fp, Ft) * m).sum() / den
                    L_vel = (huber(Vp, Vt) * m).sum() / den

                    loss = W_FORCE * L_force + W_VEL * L_vel
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                    va_losses.append(float(loss.item()))

            va_m = float(np.mean(va_losses)) if va_losses else float("inf")

        else:
            va_m = tr_m

        print(f"Ep {ep:04d} | train={tr_m:.6e} | val={va_m:.6e}")

        if np.isfinite(va_m) and va_m < best_val:
            best_val = va_m

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience = PATIENCE

        else:
            patience -= 1

            if patience <= 0:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    sv = save_model_path()

    if not sv:
        return

    torch.save({
        "state": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "norm": norm,
        "notes": "No filename matching/checking; CA read only from CSV."
    }, sv)

    root = _tk_root()

    messagebox.showinfo(
        "Training complete",
        f"Saved model:\n{sv}\n\nBest val loss: {best_val:.6e}",
        parent=root
    )

    root.destroy()


if __name__ == "__main__":
    print("\nS2 FiLM training from 102.4 Hz 6-column CSV files.")
    print("Conditioning variables are read from the CSV file only.")
    train()