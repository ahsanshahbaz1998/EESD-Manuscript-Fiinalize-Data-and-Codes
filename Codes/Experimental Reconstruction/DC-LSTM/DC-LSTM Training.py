
import os, re, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tkinter import Tk, filedialog


C_VALUE     = 696.0 
ALPHA_VALUE = 0.44   


HIDDEN   = 128
LAYERS   = 3
DROPOUT  = 0.05

FLOORS   = 3          
IN_DIM   = 3       
OUT_DIM  = 2       
LR       = 3e-3
WD       = 3e-4
BATCH    = 16
EPOCHS   = 2000
PATIENCE = 300
VAL_FRAC = 0.25
GRAD_CLIP = 1.0
# Loss weights
W_FORCE  = 1.0
W_VEL    = 1.0
EPS      = 1e-8
# ============================ Device ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
# ============================ File Pickers ============================
def pick_many_csv(title):
    root = Tk(); root.withdraw()
    return list(filedialog.askopenfilenames(title=title, filetypes=[("CSV","*.csv")]))
def save_model():
    root = Tk(); root.withdraw()
    return filedialog.asksaveasfilename(
        title="Save trained model",
        defaultextension=".pt",
        filetypes=[("PyTorch","*.pt")]
    )
def save_loss_csv():
    root = Tk(); root.withdraw()
    return filedialog.asksaveasfilename(
        title="Save training loss CSV",
        defaultextension=".csv",
        filetypes=[("CSV","*.csv")]
    )
# ============================ Dataset ============================
class DamperSingleSeqDataset(Dataset):
    """
    CSV format (>=10 columns):
      col 1   : time
      col 2-4 : D1–D3
      col 5-7 : F1–F3
      col 8-10: V1–V3 (TRUE)

    We create 3 independent samples per file (floors):
      X: (S,3) = [Di, C, alpha] at each timestep
      Y: (S,2) = [Fi, Vi]
    """
    def __init__(self, files, floors=3, C=C_VALUE, alpha=ALPHA_VALUE):
        self.items = []
        self.floors = floors
        self.C = float(C)
        self.alpha = float(alpha)

        for p in files:
            df = pd.read_csv(p)
            if df.shape[1] < 10:
                raise ValueError(f"{os.path.basename(p)} invalid (need >=10 cols)")

            t  = df.iloc[:, 0].to_numpy(np.float32)
            Dm = df.iloc[:, 1:1+floors].to_numpy(np.float32)     # D1..D3
            Fm = df.iloc[:, 4:4+floors].to_numpy(np.float32)     # F1..F3
            Vm = df.iloc[:, 7:7+floors].to_numpy(np.float32)     # V1..V3

            # keep finite rows
            m = (
                np.isfinite(t) &
                np.isfinite(Dm).all(axis=1) &
                np.isfinite(Fm).all(axis=1) &
                np.isfinite(Vm).all(axis=1)
            )
            if m.sum() < 6:
                continue

            t, Dm, Fm, Vm = t[m], Dm[m], Fm[m], Vm[m]

            # sort by time if needed
            if np.any(np.diff(t) <= 0):
                idx = np.argsort(t)
                t, Dm, Fm, Vm = t[idx], Dm[idx], Fm[idx], Vm[idx]

            # constants as per-timestep columns
            Ccol = np.full((len(t), 1), self.C, dtype=np.float32)
            Acol = np.full((len(t), 1), self.alpha, dtype=np.float32)

            # create independent samples per floor
            for i in range(floors):
                Di = Dm[:, i:i+1].astype(np.float32)              # (S,1)
                Fi = Fm[:, i:i+1].astype(np.float32)              # (S,1)
                Vi = Vm[:, i:i+1].astype(np.float32)              # (S,1)

                Xi = np.hstack([Di, Ccol, Acol]).astype(np.float32)  # (S,3)
                Yi = np.hstack([Fi, Vi]).astype(np.float32)          # (S,2)

                if len(Xi) >= 6:
                    self.items.append((Xi, Yi))

        if len(self.items) == 0:
            raise ValueError("No valid sequences found (need >=10 cols and length>=6).")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

# ============================ Collate ============================
def collate_fullseq(batch):
    B = len(batch)
    lengths = [len(x[0]) for x in batch]
    Lmax = max(lengths)

    Xb = torch.zeros((B, Lmax, IN_DIM), dtype=torch.float32)
    Yb = torch.zeros((B, Lmax, OUT_DIM), dtype=torch.float32)
    mask = torch.zeros((B, Lmax, 1), dtype=torch.float32)

    for i,(X,Y) in enumerate(batch):
        L = len(X)
        Xb[i,:L] = torch.from_numpy(X)
        Yb[i,:L] = torch.from_numpy(Y)
        mask[i,:L,0] = 1.0

    return Xb, Yb, mask

# ============================ Model (SEPARATE HEADS) ============================
class SingleDamperLSTM(nn.Module):
    """
    Shared LSTM trunk + two heads:
      headF -> 1 force
      headV -> 1 velocity
    Output concat: [F, V]
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=IN_DIM,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            dropout=(DROPOUT if LAYERS > 1 else 0.0),
            batch_first=True
        )
        self.headF = nn.Sequential(
            nn.Linear(HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.headV = nn.Sequential(
            nn.Linear(HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        h,_ = self.lstm(x)
        Fp = self.headF(h)
        Vp = self.headV(h)
        y = torch.cat([Fp, Vp], dim=-1)   # (B,S,2)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

# ============================ Utils ============================
def extract_base_id_from_stem(stem: str) -> str:
    m = re.search(r"^(.*?)__floor\d+__", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    s = re.sub(r"__C\d+_a\d+.*$", "", stem, flags=re.IGNORECASE)
    return s

def _base_id_from_file(fp: str) -> str:
    stem = os.path.splitext(os.path.basename(fp))[0]
    return extract_base_id_from_stem(stem)

def split_files(files, val_frac=VAL_FRAC, seed=42):
    files = list(files)
    if len(files) <= 1:
        return files, []

    base_ids = sorted(set(_base_id_from_file(fp) for fp in files))
    if len(base_ids) <= 1:
        return files, []

    n_val = max(1, int(round(len(base_ids) * float(val_frac))))

    rs = np.random.RandomState(int(seed))
    perm = rs.permutation(len(base_ids))

    val_ids = set(base_ids[i] for i in perm[:n_val])
    tr_ids  = set(base_ids[i] for i in perm[n_val:])

    tr  = [fp for fp in files if _base_id_from_file(fp) in tr_ids]
    val = [fp for fp in files if _base_id_from_file(fp) in val_ids]
    return tr, val

def compute_stats(dataset: DamperSingleSeqDataset):
    X = np.concatenate([x for x,_ in dataset.items], axis=0)  # (N,3)
    Y = np.concatenate([y for _,y in dataset.items], axis=0)  # (N,2)
    return {
        "X_mean": X.mean(axis=0).astype(np.float32),
        "X_std":  (X.std(axis=0) + EPS).astype(np.float32),
        "Y_mean": Y.mean(axis=0).astype(np.float32),
        "Y_std":  (Y.std(axis=0) + EPS).astype(np.float32),
    }

def norm_arr(A, mu, sd):
    return ((A - mu) / sd).astype(np.float32)

# ============================ Training ============================
def train():
    print("\n📥 Select DBE CSV files (10 cols: time, D1-3, F1-3, V1-3)")
    dbe = pick_many_csv("Select DBE CSV files (10 cols)")

    print("\n📥 Select MCE CSV files (10 cols: time, D1-3, F1-3, V1-3)")
    mce = pick_many_csv("Select MCE CSV files (10 cols)")

    dbe_tr, dbe_va = split_files(dbe, val_frac=VAL_FRAC, seed=42)
    mce_tr, mce_va = split_files(mce, val_frac=VAL_FRAC, seed=42)

    train_files = dbe_tr + mce_tr
    val_files   = dbe_va + mce_va

    print(f"DBE: train={len(dbe_tr)} val={len(dbe_va)}")
    print(f"MCE: train={len(mce_tr)} val={len(mce_va)}")
    print(f"TOTAL: train={len(train_files)} val={len(val_files)}")
    print("ℹ️ Each file becomes 3 independent single-damper sequences (one per floor).")
    print(f"ℹ️ Conditioning constants: C={C_VALUE}, alpha={ALPHA_VALUE}")

    # TRAIN dataset only (no leakage)
    ds_tr_raw = DamperSingleSeqDataset(train_files, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE)
    stats = compute_stats(ds_tr_raw)

    Xmu, Xsd = stats["X_mean"], stats["X_std"]
    Ymu, Ysd = stats["Y_mean"], stats["Y_std"]

    tr_list = [(norm_arr(X, Xmu, Xsd), norm_arr(Y, Ymu, Ysd)) for X,Y in ds_tr_raw.items]

    va_raw  = DamperSingleSeqDataset(val_files, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE) if len(val_files) else None
    va_list = [(norm_arr(X, Xmu, Xsd), norm_arr(Y, Ymu, Ysd)) for X,Y in (va_raw.items if va_raw else [])]

    tr_loader = DataLoader(tr_list, batch_size=BATCH, shuffle=True,  collate_fn=collate_fullseq)
    va_loader = DataLoader(va_list, batch_size=BATCH, shuffle=False, collate_fn=collate_fullseq) if len(va_list) else None

    model = SingleDamperLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    mse = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")   # <-- ONLY change: velocity uses Huber

    best_val = float("inf")
    patience = PATIENCE
    best_state = None

    loss_history = []

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_losses, tr_F, tr_V = [], [], []

        for X, Y, m in tr_loader:
            X = X.to(device)
            Y = Y.to(device)
            m = m.to(device)

            pred = model(X)
            Fp, Vp = pred[:, :, 0:1], pred[:, :, 1:2]
            Ft, Vt = Y[:, :, 0:1],    Y[:, :, 1:2]

            den = (m.sum() * 1.0 + EPS)

            L_force = (mse(Fp, Ft) * m).sum() / den
            L_vel   = (huber(Vp, Vt) * m).sum() / den

            loss = W_FORCE * L_force + W_VEL * L_vel
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            tr_losses.append(float(loss.item()))
            tr_F.append(float(L_force.item()))
            tr_V.append(float(L_vel.item()))

        tr_m  = float(np.mean(tr_losses)) if tr_losses else float("inf")
        tr_Fm = float(np.mean(tr_F)) if tr_F else float("inf")
        tr_Vm = float(np.mean(tr_V)) if tr_V else float("inf")

        # ---- Validation ----
        if va_loader is not None:
            model.eval()
            va_losses, va_F, va_V = [], [], []
            with torch.no_grad():
                for X, Y, m in va_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    m = m.to(device)

                    pred = model(X)
                    Fp, Vp = pred[:, :, 0:1], pred[:, :, 1:2]
                    Ft, Vt = Y[:, :, 0:1],    Y[:, :, 1:2]

                    den = (m.sum() * 1.0 + EPS)
                    L_force = (mse(Fp, Ft) * m).sum() / den
                    L_vel   = (huber(Vp, Vt) * m).sum() / den
                    loss = W_FORCE * L_force + W_VEL * L_vel
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

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

        print(f"Ep {ep:04d} | train={tr_m:.6e} (F={tr_Fm:.3e} V={tr_Vm:.3e}) | "
              f"val={va_m:.6e} (F={va_Fm:.3e} V={va_Vm:.3e})")

        if np.isfinite(va_m) and va_m < best_val:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("⏹️ Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    csv_path = save_loss_csv()
    if csv_path:
        pd.DataFrame(loss_history, columns=["epoch", "train_loss", "val_loss"]).to_csv(csv_path, index=False)
        print("💾 Loss CSV saved:", csv_path)

    sv = save_model()
    if sv:
        torch.save({
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "cfg": dict(HIDDEN=HIDDEN, LAYERS=LAYERS, DROPOUT=DROPOUT, IN_DIM=IN_DIM, OUT_DIM=OUT_DIM, FLOORS=FLOORS),
            "hyper": dict(
                LR=LR, WD=WD, BATCH=BATCH, EPOCHS=EPOCHS, PATIENCE=PATIENCE, VAL_FRAC=VAL_FRAC,
                W_FORCE=W_FORCE, W_VEL=W_VEL, GRAD_CLIP=GRAD_CLIP,
                DATA_LOSS="MSE"
            ),
            "norm": stats,
            "conditioning": dict(C=C_VALUE, alpha=ALPHA_VALUE),
            "notes": ("Strategy-1 Direct Concatenation: X=[D,C,alpha]. "
                      "Each floor treated independently (Di->Fi,Vi). Loss=MSE. No physics.")
        }, sv)
        print("💾 Model saved:", sv)

# ============================ Main ============================
if __name__ == "__main__":
    print("\nStrategy 1 — Direct Concatenation (Conditioned Single-Damper LSTM, MSE)")
    print("Each file -> 3 independent samples: (D1->F1,V1), (D2->F2,V2), (D3->F3,V3)")
    print("Inputs : [D, C, alpha]  (C and alpha are constants you provided)")
    print("Targets: [F, V]         (TRUE V from cols 8–10)")
    print("Loss   : MSE, mask-normalized")
    if __name__ == "__main__":
        train()