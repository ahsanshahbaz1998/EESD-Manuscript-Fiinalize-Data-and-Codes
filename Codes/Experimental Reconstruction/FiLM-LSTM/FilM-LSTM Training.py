
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tkinter import Tk, filedialog, messagebox
C_VALUE     = 696.0
ALPHA_VALUE = 0.44
VAL_FRAC = 0.20
SEED     = 42
FLOORS   = 3
IN_DIM   = 1    
CA_DIM   = 2   
OUT_DIM  = 2     
HIDDEN   = 128
LAYERS   = 3
DROPOUT  = 0.05
LR        = 3e-3
WD        = 3e-4
BATCH     = 16
EPOCHS    = 2000
PATIENCE  = 300
GRAD_CLIP = 1.0
W_FORCE  = 1.0
W_VEL    = 1.0
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
        title="Save trained FiLM model (.pt)",
        defaultextension=".pt",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return p or ""

def save_loss_csv_path():
    root = _tk_root()
    p = filedialog.asksaveasfilename(
        parent=root,
        title="Save epoch/train/val loss CSV",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return p or ""

def extract_base_id_from_stem(stem: str) -> str:
    """
    Keeps same spirit as your given FiLM code:
    if filename has __floor... pattern, strip it;
    else remove trailing case-like suffixes if present.
    """
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
    tr_ids  = set(base_ids[i] for i in perm[n_val:])

    tr = [r for r in records if r["base_id"] in tr_ids]
    va = [r for r in records if r["base_id"] in val_ids]
    return tr, va

def n_unique_motions(rec_list):
    return len(set(r["base_id"] for r in rec_list))

def read_10col_all_floors(path: str, floors=FLOORS):
    df = pd.read_csv(path)

    if df.shape[1] < 10:
        raise ValueError(f"Needs 10-column file. Got {df.shape[1]} columns.")

    arr = df.iloc[:, :10].to_numpy(np.float64)

    t  = arr[:, 0]
    Dm = arr[:, 1:1+floors]   # D1..D3
    Fm = arr[:, 4:4+floors]   # F1..F3
    Vm = arr[:, 7:7+floors]   # V1..V3

    m = (
        np.isfinite(t) &
        np.isfinite(Dm).all(axis=1) &
        np.isfinite(Fm).all(axis=1) &
        np.isfinite(Vm).all(axis=1)
    )

    if m.sum() < 20:
        raise ValueError("too few finite rows (<20)")

    t, Dm, Fm, Vm = t[m], Dm[m], Fm[m], Vm[m]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t, Dm, Fm, Vm = t[idx], Dm[idx], Fm[idx], Vm[idx]

    return (
        t.astype(np.float32),
        Dm.astype(np.float32),
        Fm.astype(np.float32),
        Vm.astype(np.float32)
    )

class FiLMCaseFileDataset(Dataset):
    """
    Each file becomes 3 independent samples:
      floor 1: X=[D1], CA=[C,alpha], Y=[F1,V1]
      floor 2: X=[D2], CA=[C,alpha], Y=[F2,V2]
      floor 3: X=[D3], CA=[C,alpha], Y=[F3,V3]
    """
    def __init__(self, records, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE):
        self.items = []
        self.skipped = []
        self.floors = int(floors)
        self.C = float(C)
        self.alpha = float(alpha)

        for r in records:
            fp = r["path"]
            try:
                _t, Dm, Fm, Vm = read_10col_all_floors(fp, floors=self.floors)
            except Exception as e:
                self.skipped.append((Path(fp).name, str(e)))
                continue

            CA = np.array([self.C, self.alpha], dtype=np.float32)

            for i in range(self.floors):
                D = Dm[:, i].reshape(-1, 1).astype(np.float32)      # (S,1)
                F = Fm[:, i].astype(np.float32)                     # (S,)
                V = Vm[:, i].astype(np.float32)                     # (S,)
                Y = np.stack([F, V], axis=1).astype(np.float32)     # (S,2)

                if len(D) >= 6:
                    self.items.append((D, CA.copy(), Y))

        if len(self.items) == 0:
            msg = "No valid sequences built. This script requires 10-col files:\n"
            msg += "time, D1-3, F1-3, V1-3\n\n"
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

    Xb  = torch.zeros((B, Lmax, IN_DIM), dtype=torch.float32)
    Yb  = torch.zeros((B, Lmax, OUT_DIM), dtype=torch.float32)
    Mb  = torch.zeros((B, Lmax, 1), dtype=torch.float32)
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
    M2   = np.zeros((dim,), dtype=np.float64)

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
    X_mu, X_sd = streaming_mean_std([x for (x, _, _) in ds_tr.items], dim=IN_DIM)
    Y_mu, Y_sd = streaming_mean_std([y for (_, _, y) in ds_tr.items], dim=OUT_DIM)

    CA_all = np.stack([ca for (_, ca, _) in ds_tr.items], axis=0).astype(np.float32)
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
            dropout=(float(dropout) if layers > 1 else 0.0),
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

        gb = self.film(ca_norm)              # (B, 2H)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        gamma = gamma.unsqueeze(1)           # (B,1,H)
        beta  = beta.unsqueeze(1)            # (B,1,H)

        h_mod = gamma * h + beta

        Fp = self.headF(h_mod)
        Vp = self.headV(h_mod)
        y = torch.cat([Fp, Vp], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

def train():
    print("\n==================== SELECT DBE FILES ====================")
    dbe_files = pick_many_csv("DBE: Select 10-column CSV files")
    if not dbe_files:
        root = _tk_root()
        messagebox.showerror("Missing selection", "No DBE files selected.", parent=root)
        root.destroy()
        return

    print("\n==================== SELECT MCE FILES ====================")
    mce_files = pick_many_csv("MCE: Select 10-column CSV files")
    if not mce_files:
        root = _tk_root()
        messagebox.showerror("Missing selection", "No MCE files selected.", parent=root)
        root.destroy()
        return

    dbe_records = [
        dict(
            hazard="DBE",
            path=str(Path(fp)),
            stem=Path(fp).stem,
            base_id=extract_base_id_from_stem(Path(fp).stem),
        )
        for fp in dbe_files
    ]

    mce_records = [
        dict(
            hazard="MCE",
            path=str(Path(fp)),
            stem=Path(fp).stem,
            base_id=extract_base_id_from_stem(Path(fp).stem),
        )
        for fp in mce_files
    ]

    dbe_tr, dbe_va = split_by_base_id(dbe_records, val_frac=VAL_FRAC, seed=SEED)
    mce_tr, mce_va = split_by_base_id(mce_records, val_frac=VAL_FRAC, seed=SEED)

    train_records = dbe_tr + mce_tr
    val_records   = dbe_va + mce_va

    print("\n==================== SPLIT SUMMARY (by base_id) ====================")
    print(f"DBE: train={len(dbe_tr)} val={len(dbe_va)} | unique motions train={n_unique_motions(dbe_tr)} val={n_unique_motions(dbe_va)}")
    print(f"MCE: train={len(mce_tr)} val={len(mce_va)} | unique motions train={n_unique_motions(mce_tr)} val={n_unique_motions(mce_va)}")
    print(f"TOTAL: train={len(train_records)} val={len(val_records)}")
    print("Each file becomes 3 independent single-damper sequences (one per floor).")
    print(f"Fixed FiLM conditioning: C={C_VALUE}, alpha={ALPHA_VALUE}")
    print("====================================================================\n")

    print(" Building TRAIN dataset...")
    ds_tr = FiLMCaseFileDataset(train_records, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE)

    print("Computing normalization stats from TRAIN only...")
    norm = compute_norm_from_train(ds_tr)
    X_mu, X_sd = norm["X_mean"], norm["X_std"]
    Y_mu, Y_sd = norm["Y_mean"], norm["Y_std"]
    CA_mu, CA_sd = norm["CA_mean"], norm["CA_std"]

    tr_items = []
    for (X, CA, Y) in ds_tr.items:
        tr_items.append((
            normalize_arr(X,  X_mu,  X_sd),
            normalize_arr(CA, CA_mu, CA_sd),
            normalize_arr(Y,  Y_mu,  Y_sd)
        ))

    if len(val_records) > 0:
        ds_va = FiLMCaseFileDataset(val_records, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE)
        va_items = []
        for (X, CA, Y) in ds_va.items:
            va_items.append((
                normalize_arr(X,  X_mu,  X_sd),
                normalize_arr(CA, CA_mu, CA_sd),
                normalize_arr(Y,  Y_mu,  Y_sd)
            ))
    else:
        va_items = []

    tr_loader = DataLoader(tr_items, batch_size=BATCH, shuffle=True,  collate_fn=collate_fullseq)
    va_loader = DataLoader(va_items, batch_size=BATCH, shuffle=False, collate_fn=collate_fullseq) if va_items else None

    model = FiLMSingleDamperLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    mse   = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")

    best_val = float("inf")
    patience = PATIENCE
    best_state = None

    loss_history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_losses, tr_F, tr_V = [], [], []

        for X, CA, Y, m in tr_loader:
            X  = X.to(device)
            CA = CA.to(device)
            Y  = Y.to(device)
            m  = m.to(device)

            pred = model(X, CA)
            Fp, Vp = pred[:, :, 0:1], pred[:, :, 1:2]
            Ft, Vt = Y[:, :, 0:1],    Y[:, :, 1:2]

            den = (m.sum() + EPS)

            L_force = (mse(Fp, Ft) * m).sum() / den
            L_vel   = (huber(Vp, Vt) * m).sum() / den
            loss    = W_FORCE * L_force + W_VEL * L_vel
            loss    = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            tr_losses.append(float(loss.item()))
            tr_F.append(float(L_force.item()))
            tr_V.append(float(L_vel.item()))

        tr_m  = float(np.mean(tr_losses)) if tr_losses else float("inf")
        tr_Fm = float(np.mean(tr_F)) if tr_F else float("inf")
        tr_Vm = float(np.mean(tr_V)) if tr_V else float("inf")

        if va_loader is not None:
            model.eval()
            va_losses, va_F, va_V = [], [], []

            with torch.no_grad():
                for X, CA, Y, m in va_loader:
                    X  = X.to(device)
                    CA = CA.to(device)
                    Y  = Y.to(device)
                    m  = m.to(device)

                    pred = model(X, CA)
                    Fp, Vp = pred[:, :, 0:1], pred[:, :, 1:2]
                    Ft, Vt = Y[:, :, 0:1],    Y[:, :, 1:2]

                    den = (m.sum() + EPS)

                    L_force = (mse(Fp, Ft) * m).sum() / den
                    L_vel   = (huber(Vp, Vt) * m).sum() / den
                    loss    = W_FORCE * L_force + W_VEL * L_vel
                    loss    = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                    va_losses.append(float(loss.item()))
                    va_F.append(float(L_force.item()))
                    va_V.append(float(L_vel.item()))

            va_m  = float(np.mean(va_losses)) if va_losses else float("inf")
            va_Fm = float(np.mean(va_F)) if va_F else float("inf")
            va_Vm = float(np.mean(va_V)) if va_V else float("inf")
        else:
            va_m, va_Fm, va_Vm = tr_m, tr_Fm, tr_Vm

        loss_history.append({
            "epoch": ep,
            "train_loss": tr_m,
            "val_loss": va_m
        })

        print(
            f"Ep {ep:04d} | "
            f"train={tr_m:.6e} (F={tr_Fm:.3e} V={tr_Vm:.3e}) | "
            f"val={va_m:.6e} (F={va_Fm:.3e} V={va_Vm:.3e})"
        )

        if np.isfinite(va_m) and va_m < best_val:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    csv_path = save_loss_csv_path()
    if csv_path:
        pd.DataFrame(loss_history, columns=["epoch", "train_loss", "val_loss"]).to_csv(csv_path, index=False)
        print("Loss CSV saved:", csv_path)

    sv = save_model_path()
    if not sv:
        return

    torch.save({
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": dict(
            IN_DIM=IN_DIM,
            CA_DIM=CA_DIM,
            OUT_DIM=OUT_DIM,
            HIDDEN=HIDDEN,
            LAYERS=LAYERS,
            DROPOUT=DROPOUT,
            FLOORS=FLOORS
        ),
        "hyper": dict(
            LR=LR,
            WD=WD,
            BATCH=BATCH,
            EPOCHS=EPOCHS,
            PATIENCE=PATIENCE,
            VAL_FRAC=VAL_FRAC,
            W_FORCE=W_FORCE,
            W_VEL=W_VEL,
            GRAD_CLIP=GRAD_CLIP,
            LOSS_FORCE="MSE",
            LOSS_VEL="SmoothL1"
        ),
        "norm": norm,
        "conditioning": dict(
            mode="FiLM",
            fixed=True,
            C=C_VALUE,
            alpha=ALPHA_VALUE
        ),
        "notes": (
            "Strategy 2 FiLM with fixed conditioning constants. "
            "10-column CSV input: time, D1-3, F1-3, V1-3. "
            "Each file creates 3 independent floor-wise samples: "
            "(D1->F1,V1), (D2->F2,V2), (D3->F3,V3). "
            "LSTM input is D only; FiLM conditioning uses CA=[C,alpha]."
        )
    }, sv)

    root = _tk_root()
    messagebox.showinfo(
        "Training complete",
        f"Saved model:\n{sv}\n\nBest val loss: {best_val:.6e}",
        parent=root
    )
    root.destroy()

if __name__ == "__main__":
    print("\nStrategy 2 — FiLM-LSTM (Fixed C/alpha, 10-col, 3 floors independent)")
    print("Each file -> 3 independent samples: (D1->F1,V1), (D2->F2,V2), (D3->F3,V3)")
    print("LSTM input : [D] only")
    print(f"FiLM CA    : [C, alpha] = [{C_VALUE}, {ALPHA_VALUE}]")
    print("Targets    : [F, V] from cols 5-10")
    print("Loss       : Force=MSE, Velocity=SmoothL1")
    if __name__ == "__main__":
        train()