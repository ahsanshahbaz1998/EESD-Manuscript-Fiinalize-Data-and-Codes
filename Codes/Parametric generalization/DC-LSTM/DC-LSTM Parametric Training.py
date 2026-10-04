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

HIDDEN = 128
LAYERS = 3
DROPOUT = 0.05

IN_DIM = 3
OUT_DIM = 2

LR = 3e-3
WD = 3e-4
BATCH = 16
EPOCHS = 2000
PATIENCE = 300
GRAD_CLIP = 1.0

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


def _n_unique_base(rec_list):
    return len(set(r["base_id"] for r in rec_list))


def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "").replace("-", "_")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "_")
    return s


def find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    norm_map = {_norm_name(c): c for c in cols}

    for cand in candidates:
        key = _norm_name(cand)

        if key in norm_map:
            return norm_map[key]

    return None


def read_6col_file_use_DCA_VF(path: str):
    df = pd.read_csv(path)

    if df.shape[0] < 5:
        raise ValueError("too few rows")

    cC = find_col(df, ["C", "C_used", "C_case", "c"])
    cA = find_col(df, ["alpha", "alpha_used", "alpha_case", "a"])
    cD = find_col(df, ["D_mm", "D1_mm", "D", "disp_mm", "displacement_mm"])
    cV = find_col(df, ["V_mps", "V1_mps", "V", "vel_mps", "velocity_mps"])
    cF = find_col(df, ["F_kN", "F_case_kN", "Fcase_kN", "F_case", "F", "force_kn"])

    if any(c is None for c in [cC, cA, cD, cV, cF]):
        if df.shape[1] < 6:
            raise ValueError(f"need 6 columns, got {df.shape[1]} cols: {list(df.columns)}")

        arr = df.iloc[:, :6].to_numpy(np.float64)

        Cc = arr[:, 1]
        Aa = arr[:, 2]
        D = arr[:, 3]
        V = arr[:, 4]
        Fv = arr[:, 5]

    else:
        Cc = pd.to_numeric(df[cC], errors="coerce").to_numpy(np.float64)
        Aa = pd.to_numeric(df[cA], errors="coerce").to_numpy(np.float64)
        D = pd.to_numeric(df[cD], errors="coerce").to_numpy(np.float64)
        V = pd.to_numeric(df[cV], errors="coerce").to_numpy(np.float64)
        Fv = pd.to_numeric(df[cF], errors="coerce").to_numpy(np.float64)

    m = (
        np.isfinite(Cc) &
        np.isfinite(Aa) &
        np.isfinite(D) &
        np.isfinite(V) &
        np.isfinite(Fv)
    )

    if m.sum() < 20:
        raise ValueError("too few finite rows (<20)")

    Cc, Aa, D, V, Fv = Cc[m], Aa[m], D[m], V[m], Fv[m]

    return (
        D.astype(np.float32),
        Cc.astype(np.float32),
        Aa.astype(np.float32),
        V.astype(np.float32),
        Fv.astype(np.float32),
    )


class DVF_Dataset(Dataset):
    def __init__(self, records):
        self.items = []
        self.skipped = []

        for r in records:
            fp = r["path"]

            try:
                D, Cc, Aa, V, Fv = read_6col_file_use_DCA_VF(fp)
            except Exception as e:
                self.skipped.append((Path(fp).name, str(e)))
                continue

            if len(D) < 20:
                self.skipped.append((Path(fp).name, "sequence too short"))
                continue

            X = np.column_stack([D, Cc, Aa]).astype(np.float32)
            Y = np.column_stack([Fv, V]).astype(np.float32)

            self.items.append((X, Y))

        if len(self.items) == 0:
            msg = "No valid sequences found.\n\nTop skip reasons:\n"

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
    mask = torch.zeros((B, Lmax, 1), dtype=torch.float32)

    for i, (X, Y) in enumerate(batch):
        L = len(X)

        Xb[i, :L] = torch.from_numpy(X)
        Yb[i, :L] = torch.from_numpy(Y)
        mask[i, :L, 0] = 1.0

    return Xb, Yb, mask


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


class SingleDamperLSTM(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden),
            num_layers=int(layers),
            dropout=float(dropout) if layers > 1 else 0.0,
            batch_first=True
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

    def forward(self, x):
        h, _ = self.lstm(x)

        Fp = self.headF(h)
        Vp = self.headV(h)

        y = torch.cat([Fp, Vp], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def train():
    all_records = []

    print("\n==================== SELECT DBE FILES (4 cases) ====================")

    for (Cc, aa, short, lab) in CASES4:
        title = f"DBE: Select 6-col files for case {short} (C/alpha taken from file as direct input)"
        files = pick_many_csv(title)

        if not files:
            root = _tk_root()
            messagebox.showerror(
                "Missing selection",
                f"No files selected for DBE case {short}.",
                parent=root
            )
            root.destroy()
            return

        for fp in files:
            p = Path(fp)

            all_records.append(dict(
                hazard="DBE",
                case_short=short,
                case_label=lab,
                path=str(p),
                stem=p.stem,
                base_id=extract_base_id_from_stem(p.stem),
            ))

    print("\n==================== SELECT MCE FILES (4 cases) ====================")

    for (Cc, aa, short, lab) in CASES4:
        title = f"MCE: Select 6-col files for case {short} (C/alpha taken from file as direct input)"
        files = pick_many_csv(title)

        if not files:
            root = _tk_root()
            messagebox.showerror(
                "Missing selection",
                f"No files selected for MCE case {short}.",
                parent=root
            )
            root.destroy()
            return

        for fp in files:
            p = Path(fp)

            all_records.append(dict(
                hazard="MCE",
                case_short=short,
                case_label=lab,
                path=str(p),
                stem=p.stem,
                base_id=extract_base_id_from_stem(p.stem),
            ))

    dbe_records = [r for r in all_records if r["hazard"] == "DBE"]
    mce_records = [r for r in all_records if r["hazard"] == "MCE"]

    dbe_tr, dbe_va = split_by_base_id(dbe_records, val_frac=VAL_FRAC, seed=SEED)
    mce_tr, mce_va = split_by_base_id(mce_records, val_frac=VAL_FRAC, seed=SEED)

    train_records = dbe_tr + mce_tr
    val_records = dbe_va + mce_va

    print("\n==================== SPLIT SUMMARY (by base_id) ====================")
    print(f"DBE: train files={len(dbe_tr)} val files={len(dbe_va)} | unique motions train={_n_unique_base(dbe_tr)} val={_n_unique_base(dbe_va)}")
    print(f"MCE: train files={len(mce_tr)} val files={len(mce_va)} | unique motions train={_n_unique_base(mce_tr)} val={_n_unique_base(mce_va)}")
    print(f"TOTAL: train={len(train_records)} val={len(val_records)}")
    print("====================================================================\n")

    print("Building training dataset...")
    ds_tr = DVF_Dataset(train_records)

    if ds_tr.skipped:
        print(f"Skipped {len(ds_tr.skipped)} train files (showing up to 8):")

        for nm, err in ds_tr.skipped[:8]:
            print("  -", nm, "|", err)

    print("Computing normalization statistics from training data...")
    X_mu, X_sd = streaming_mean_std([x for x, _ in ds_tr.items], dim=IN_DIM)
    Y_mu, Y_sd = streaming_mean_std([y for _, y in ds_tr.items], dim=OUT_DIM)

    tr_items = [
        (normalize_arr(X, X_mu, X_sd), normalize_arr(Y, Y_mu, Y_sd))
        for (X, Y) in ds_tr.items
    ]

    if len(val_records) > 0:
        print("Building validation dataset...")
        ds_va = DVF_Dataset(val_records)

        if ds_va.skipped:
            print(f"Skipped {len(ds_va.skipped)} val files (showing up to 8):")

            for nm, err in ds_va.skipped[:8]:
                print("  -", nm, "|", err)

        va_items = [
            (normalize_arr(X, X_mu, X_sd), normalize_arr(Y, Y_mu, Y_sd))
            for (X, Y) in ds_va.items
        ]

    else:
        ds_va = None
        va_items = []

    print(f"Sequences: train={len(tr_items)} val={len(va_items)}")

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
    ) if len(va_items) else None

    model = SingleDamperLSTM().to(device)

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
        tr_F = []
        tr_V = []

        for X, Y, m in tr_loader:
            X = X.to(device)
            Y = Y.to(device)
            m = m.to(device)

            pred = model(X)

            Fp = pred[:, :, 0:1]
            Vp = pred[:, :, 1:2]

            Ft = Y[:, :, 0:1]
            Vt = Y[:, :, 1:2]

            den = m.sum() + EPS

            L_force = (mse(Fp, Ft) * m).sum() / den
            L_vel = (huber(Vp, Vt) * m).sum() / den

            loss = L_force + L_vel
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()

            tr_losses.append(float(loss.item()))
            tr_F.append(float(L_force.item()))
            tr_V.append(float(L_vel.item()))

        tr_m = float(np.mean(tr_losses)) if tr_losses else float("inf")
        tr_Fm = float(np.mean(tr_F)) if tr_F else float("inf")
        tr_Vm = float(np.mean(tr_V)) if tr_V else float("inf")

        if va_loader is not None:
            model.eval()

            va_losses = []
            va_F = []
            va_V = []

            with torch.no_grad():
                for X, Y, m in va_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    m = m.to(device)

                    pred = model(X)

                    Fp = pred[:, :, 0:1]
                    Vp = pred[:, :, 1:2]

                    Ft = Y[:, :, 0:1]
                    Vt = Y[:, :, 1:2]

                    den = m.sum() + EPS

                    L_force = (mse(Fp, Ft) * m).sum() / den
                    L_vel = (huber(Vp, Vt) * m).sum() / den

                    loss = L_force + L_vel
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                    va_losses.append(float(loss.item()))
                    va_F.append(float(L_force.item()))
                    va_V.append(float(L_vel.item()))

            va_m = float(np.mean(va_losses)) if va_losses else float("inf")
            va_Fm = float(np.mean(va_F)) if va_F else float("inf")
            va_Vm = float(np.mean(va_V)) if va_V else float("inf")

        else:
            va_m = tr_m
            va_Fm = tr_Fm
            va_Vm = tr_Vm

        print(
            f"Ep {ep:04d} | "
            f"train={tr_m:.6e} (F={tr_Fm:.3e} V={tr_Vm:.3e}) | "
            f"val={va_m:.6e} (F={va_Fm:.3e} V={va_Vm:.3e})"
        )

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
        print("No save path selected. Exit without saving.")
        return

    norm = dict(
        X_mean=X_mu.astype(np.float32),
        X_std=np.maximum(X_sd, 1e-12).astype(np.float32),
        Y_mean=Y_mu.astype(np.float32),
        Y_std=np.maximum(Y_sd, 1e-12).astype(np.float32),
    )

    split_info = dict(
        VAL_FRAC=VAL_FRAC,
        SEED=SEED,
        dbe_unique_motions_train=_n_unique_base(dbe_tr),
        dbe_unique_motions_val=_n_unique_base(dbe_va),
        mce_unique_motions_train=_n_unique_base(mce_tr),
        mce_unique_motions_val=_n_unique_base(mce_va),
        train_files=len(train_records),
        val_files=len(val_records),
    )

    torch.save({
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": dict(
            HIDDEN=HIDDEN,
            LAYERS=LAYERS,
            DROPOUT=DROPOUT,
            IN_DIM=IN_DIM,
            OUT_DIM=OUT_DIM
        ),
        "hyper": dict(
            LR=LR,
            WD=WD,
            BATCH=BATCH,
            EPOCHS=EPOCHS,
            PATIENCE=PATIENCE,
            GRAD_CLIP=GRAD_CLIP,
            VAL_FRAC=VAL_FRAC
        ),
        "norm": norm,
        "conditioning": dict(
            floor="FLOOR-1 ONLY",
            X="[D_mm, C, alpha]",
            Y="[F_kN, V_mps]"
        ),
        "cases": [
            (float(Cc), float(aa), short, lab)
            for (Cc, aa, short, lab) in CASES4
        ],
        "split_info": split_info,
        "notes": (
            "Trained S1 LSTM-HV on 102.4 Hz 6-column files using direct conditioned input from file. "
            "Inputs X=[D_mm, C, alpha]. Targets Y=[F_kN, V_mps]. "
            "Files selected per case (DBE 4 then MCE 4). "
            "Val split by base_id separately for DBE/MCE."
        )
    }, sv)

    root = _tk_root()

    messagebox.showinfo(
        "Training complete",
        f"Saved model:\n{sv}\n\nBest val loss: {best_val:.6e}\n\nSplit info:\n{split_info}",
        parent=root
    )

    root.destroy()
    print("Model saved:", sv)


if __name__ == "__main__":
    print("\nS1 LSTM HV — Train from 102.4 Hz 6-col CSVs | FLOOR-1 ONLY")
    print("C and alpha are taken from the file as direct conditioned input.")
    print("Input is [D_mm, C, alpha].")
    train()