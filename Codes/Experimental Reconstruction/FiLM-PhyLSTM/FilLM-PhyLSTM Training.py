import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tkinter import Tk, filedialog, messagebox


C_VALUE = 696.0
ALPHA_VALUE = 0.44

USE_TRUE_V_IN_LAW = False

HIDDEN = 128
LAYERS = 3
DROPOUT = 0.05

FLOORS = 3
IN_DIM = 3
CA_DIM = 2
OUT_DIM = 2

LR = 3e-3
WD = 3e-4
BATCH = 16
EPOCHS = 2000
PATIENCE = 300
VAL_FRAC = 0.25
GRAD_CLIP = 1.0

W_FORCE = 1.0
W_VEL = 1.0
LAMBDA_LAW = 0.5

LOG_EXP_CLAMP = 50.0
SGN_EPS = 1e-12
ALPHA_MIN_CLAMP = 0.15
ALPHA_MAX_CLAMP = 1.20

EPS = 1e-8
SEED = 42

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
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
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


def save_model_dialog(title="Save trained model (.pt)"):
    root = _tk_root()
    p = filedialog.asksaveasfilename(
        parent=root,
        title=title,
        defaultextension=".pt",
        filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return p or ""


def save_loss_csv_dialog(title="Save epoch/train/val loss CSV"):
    root = _tk_root()
    p = filedialog.asksaveasfilename(
        parent=root,
        title=title,
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return p or ""


def extract_base_id_from_stem(stem: str) -> str:
    m = re.search(r"^(.*?)__floor\d+__", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return re.sub(r"__C\d+_a\d+.*$", "", stem, flags=re.IGNORECASE)


def _base_id_from_file(fp: str) -> str:
    stem = os.path.splitext(os.path.basename(fp))[0]
    return extract_base_id_from_stem(stem)


def split_files(files, val_frac=VAL_FRAC, seed=SEED):
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
    tr_ids = set(base_ids[i] for i in perm[n_val:])

    tr = [fp for fp in files if _base_id_from_file(fp) in tr_ids]
    val = [fp for fp in files if _base_id_from_file(fp) in val_ids]

    return tr, val


def n_unique_motions(files):
    return len(set(_base_id_from_file(fp) for fp in files))


def median_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    if t.size < 3:
        return float("nan")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        return float("nan")

    return float(np.median(dt))


def read_10col_all_floors(csv_path: str, floors=FLOORS):
    df = pd.read_csv(csv_path)

    if df.shape[1] < 10:
        raise ValueError(f"{os.path.basename(csv_path)} needs >=10 columns.")

    arr = df.iloc[:, :10].to_numpy(np.float64)

    t = arr[:, 0]
    Dm = arr[:, 1:1+floors]
    Fm = arr[:, 4:4+floors]
    Vm = arr[:, 7:7+floors]

    m = (
        np.isfinite(t) &
        np.isfinite(Dm).all(axis=1) &
        np.isfinite(Fm).all(axis=1) &
        np.isfinite(Vm).all(axis=1)
    )

    if m.sum() < 20:
        raise ValueError("Too few valid rows (<20)")

    t, Dm, Fm, Vm = t[m], Dm[m], Fm[m], Vm[m]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t, Dm, Fm, Vm = t[idx], Dm[idx], Fm[idx], Vm[idx]

    dt_med = median_dt(t)
    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 1.0 / 102.4

    return (
        t.astype(np.float64),
        Dm.astype(np.float32),
        Fm.astype(np.float32),
        Vm.astype(np.float32),
        float(dt_med),
    )


class FiLMPhyDataset10Col(Dataset):
    def __init__(self, files, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE):
        self.items = []
        self.skipped = []
        self.floors = int(floors)
        self.C = float(C)
        self.alpha = float(np.clip(alpha, ALPHA_MIN_CLAMP, ALPHA_MAX_CLAMP))

        for fp in files:
            try:
                t, Dm, Fm, Vm, dt_med = read_10col_all_floors(fp, floors=self.floors)
            except Exception as e:
                self.skipped.append((os.path.basename(fp), str(e)))
                continue

            L = len(t)
            Ccol = np.full((L, 1), self.C, dtype=np.float32)
            Acol = np.full((L, 1), self.alpha, dtype=np.float32)

            for i in range(self.floors):
                D = Dm[:, i:i+1].astype(np.float32)
                Ftrue = Fm[:, i:i+1].astype(np.float32)
                Vtrue = Vm[:, i:i+1].astype(np.float32)

                X = np.hstack([D, Ccol, Acol]).astype(np.float32)
                Y = np.hstack([Ftrue, Vtrue]).astype(np.float32)

                meta = dict(
                    file=os.path.basename(fp),
                    floor=int(i + 1),
                    C_used=float(self.C),
                    alpha_used=float(self.alpha)
                )

                if len(X) >= 6:
                    self.items.append((X, Y, float(dt_med), meta))

        if len(self.items) == 0:
            msg = "No valid sequences built. Need 10-column files:\n"
            msg += "time, D1-3, F1-3, V1-3\n\n"
            msg += "Skip reasons:\n"
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
    dtb = torch.zeros((B, 1), dtype=torch.float32)

    metas = []

    for i, (X, Y, dt, meta) in enumerate(batch):
        L = len(X)
        Xb[i, :L] = torch.from_numpy(X)
        Yb[i, :L] = torch.from_numpy(Y)
        Mb[i, :L, 0] = 1.0
        dtb[i, 0] = float(dt)
        metas.append(meta)

    return Xb, Yb, Mb, dtb, metas


def compute_stats_from_items(items):
    X = np.concatenate([x for (x, _, _, _) in items], axis=0)
    Y = np.concatenate([y for (_, y, _, _) in items], axis=0)

    return {
        "X_mean": X.mean(axis=0).astype(np.float32),
        "X_std": (X.std(axis=0) + EPS).astype(np.float32),
        "Y_mean": Y.mean(axis=0).astype(np.float32),
        "Y_std": (Y.std(axis=0) + EPS).astype(np.float32),
    }


def norm_arr(A, mu, sd):
    mu = np.asarray(mu, dtype=np.float32)
    sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ((A - mu) / sd).astype(np.float32)


def _inv_softplus(y: float):
    y = max(float(y), 1e-12)
    return float(np.log(np.expm1(y)))


class FiLMPhyLSTM_HV(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, ks_init=1.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
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

        self._ks_raw = nn.Parameter(torch.tensor(_inv_softplus(ks_init), dtype=torch.float32))

    def ks_value(self):
        return F.softplus(self._ks_raw) + 1e-12

    def forward(self, x):
        h, _ = self.lstm(x)

        ca_norm = x[:, 0, 1:3]
        gb = self.film(ca_norm)
        gamma, beta = torch.chunk(gb, 2, dim=1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        h_mod = gamma * h + beta

        Fp = self.headF(h_mod)
        Vp = self.headV(h_mod)

        y = torch.cat([Fp, Vp], dim=-1)

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def law_loss_relmse_learnableKs(
    F_phys,
    V_pred_phys,
    V_true_phys,
    C_phys,
    A_phys,
    mask,
    dtb,
    ks_scalar
):
    Vuse = V_true_phys if USE_TRUE_V_IN_LAW else V_pred_phys

    mp = mask[:, 1:, :] * mask[:, :-1, :]
    dt = dtb.view(-1, 1, 1)
    dt = torch.clamp(dt, min=1e-12)

    Fdot = (F_phys[:, 1:, :] - F_phys[:, :-1, :]) / dt

    Cc = torch.clamp(C_phys, min=1e-12)
    Aa = torch.clamp(A_phys, min=ALPHA_MIN_CLAMP, max=ALPHA_MAX_CLAMP)

    Fnow = F_phys[:, :-1, :]
    Vnow = Vuse[:, :-1, :]

    absF = torch.abs(Fnow) + 1e-24
    log_absF = torch.log(absF)
    logC = torch.log(torch.clamp(Cc[:, :-1, :], min=1e-12))
    inv_a = 1.0 / Aa[:, :-1, :]

    expo = inv_a * (log_absF - logC)
    expo = torch.clamp(expo, -LOG_EXP_CLAMP, LOG_EXP_CLAMP)

    mag = torch.exp(expo)
    sgn = Fnow / (torch.abs(Fnow) + SGN_EPS)
    vc = sgn * mag

    ks = ks_scalar.view(1, 1, 1).to(F_phys.dtype).to(F_phys.device)
    rhs = ks * (Vnow - vc)

    r = Fdot - rhs
    denom = torch.abs(Fdot) + torch.abs(rhs) + EPS
    rrel = r / torch.clamp(denom, min=1e-6)

    per = rrel * rrel
    den = mp.sum() + EPS

    return (per * mp).sum() / den


def train():
    print("\nSelect DBE CSV files")
    dbe = pick_many_csv("Select DBE CSV files")

    print("\nSelect MCE CSV files")
    mce = pick_many_csv("Select MCE CSV files")

    if len(dbe) == 0 and len(mce) == 0:
        print("No files selected.")
        return

    dbe_tr, dbe_va = split_files(dbe, val_frac=VAL_FRAC, seed=SEED)
    mce_tr, mce_va = split_files(mce, val_frac=VAL_FRAC, seed=SEED)

    train_files = dbe_tr + mce_tr
    val_files = dbe_va + mce_va

    print("\nSplit summary")
    print(f"DBE: train={len(dbe_tr)} val={len(dbe_va)} | unique train={n_unique_motions(dbe_tr)} val={n_unique_motions(dbe_va)}")
    print(f"MCE: train={len(mce_tr)} val={len(mce_va)} | unique train={n_unique_motions(mce_tr)} val={n_unique_motions(mce_va)}")
    print(f"Total: train={len(train_files)} val={len(val_files)}")

    ds_tr_raw = FiLMPhyDataset10Col(train_files, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE)
    stats = compute_stats_from_items(ds_tr_raw.items)

    Xmu, Xsd = stats["X_mean"], stats["X_std"]
    Ymu, Ysd = stats["Y_mean"], stats["Y_std"]

    tr_list = [
        (norm_arr(X, Xmu, Xsd), norm_arr(Y, Ymu, Ysd), dt, meta)
        for (X, Y, dt, meta) in ds_tr_raw.items
    ]

    va_list = []
    if len(val_files) > 0:
        ds_va_raw = FiLMPhyDataset10Col(val_files, floors=FLOORS, C=C_VALUE, alpha=ALPHA_VALUE)
        va_list = [
            (norm_arr(X, Xmu, Xsd), norm_arr(Y, Ymu, Ysd), dt, meta)
            for (X, Y, dt, meta) in ds_va_raw.items
        ]

    tr_loader = DataLoader(tr_list, batch_size=BATCH, shuffle=True, collate_fn=collate_fullseq)
    va_loader = DataLoader(va_list, batch_size=BATCH, shuffle=False, collate_fn=collate_fullseq) if len(va_list) else None

    model = FiLMPhyLSTM_HV(ks_init=1.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    mse = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")

    Xmu_t = torch.tensor(Xmu, dtype=torch.float32, device=device).view(1, 1, IN_DIM)
    Xsd_t = torch.tensor(Xsd, dtype=torch.float32, device=device).view(1, 1, IN_DIM)
    Ymu_t = torch.tensor(Ymu, dtype=torch.float32, device=device).view(1, 1, OUT_DIM)
    Ysd_t = torch.tensor(Ysd, dtype=torch.float32, device=device).view(1, 1, OUT_DIM)

    best_val = float("inf")
    patience = PATIENCE
    best_state = None

    loss_history = []

    for ep in range(1, EPOCHS + 1):
        model.train()

        tr_losses = []
        tr_data_losses = []
        tr_law_losses = []

        for Xn, Yn, m, dtb, _ in tr_loader:
            Xn = Xn.to(device)
            Yn = Yn.to(device)
            m = m.to(device)
            dtb = dtb.to(device)

            pred_n = model(Xn)

            Fp_n = pred_n[:, :, 0:1]
            Vp_n = pred_n[:, :, 1:2]

            Ft_n = Yn[:, :, 0:1]
            Vt_n = Yn[:, :, 1:2]

            den = m.sum() + EPS

            Lf = (mse(Fp_n, Ft_n) * m).sum() / den
            Lv = (huber(Vp_n, Vt_n) * m).sum() / den
            L_data = W_FORCE * Lf + W_VEL * Lv

            X_phys = Xn * Xsd_t + Xmu_t
            Yp_phys = pred_n * Ysd_t + Ymu_t
            Yt_phys = Yn * Ysd_t + Ymu_t

            C_phys = X_phys[:, :, 1:2]
            A_phys = X_phys[:, :, 2:3]

            F_phys = Yp_phys[:, :, 0:1]
            V_predphys = Yp_phys[:, :, 1:2]
            V_truephys = Yt_phys[:, :, 1:2]

            L_law = law_loss_relmse_learnableKs(
                F_phys,
                V_predphys,
                V_truephys,
                C_phys,
                A_phys,
                m,
                dtb,
                model.ks_value()
            )

            loss = L_data + LAMBDA_LAW * L_law
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            tr_losses.append(float(loss.item()))
            tr_data_losses.append(float(L_data.item()))
            tr_law_losses.append(float(L_law.item()))

        tr_m = float(np.mean(tr_losses)) if tr_losses else float("inf")
        tr_dm = float(np.mean(tr_data_losses)) if tr_data_losses else float("inf")
        tr_lwm = float(np.mean(tr_law_losses)) if tr_law_losses else 0.0

        if va_loader is not None:
            model.eval()

            va_losses = []
            va_data_losses = []
            va_law_losses = []

            with torch.no_grad():
                for Xn, Yn, m, dtb, _ in va_loader:
                    Xn = Xn.to(device)
                    Yn = Yn.to(device)
                    m = m.to(device)
                    dtb = dtb.to(device)

                    pred_n = model(Xn)

                    Fp_n = pred_n[:, :, 0:1]
                    Vp_n = pred_n[:, :, 1:2]

                    Ft_n = Yn[:, :, 0:1]
                    Vt_n = Yn[:, :, 1:2]

                    den = m.sum() + EPS

                    Lf = (mse(Fp_n, Ft_n) * m).sum() / den
                    Lv = (huber(Vp_n, Vt_n) * m).sum() / den
                    L_data = W_FORCE * Lf + W_VEL * Lv

                    X_phys = Xn * Xsd_t + Xmu_t
                    Yp_phys = pred_n * Ysd_t + Ymu_t
                    Yt_phys = Yn * Ysd_t + Ymu_t

                    C_phys = X_phys[:, :, 1:2]
                    A_phys = X_phys[:, :, 2:3]

                    F_phys = Yp_phys[:, :, 0:1]
                    V_predphys = Yp_phys[:, :, 1:2]
                    V_truephys = Yt_phys[:, :, 1:2]

                    L_law = law_loss_relmse_learnableKs(
                        F_phys,
                        V_predphys,
                        V_truephys,
                        C_phys,
                        A_phys,
                        m,
                        dtb,
                        model.ks_value()
                    )

                    loss = L_data + LAMBDA_LAW * L_law
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                    va_losses.append(float(loss.item()))
                    va_data_losses.append(float(L_data.item()))
                    va_law_losses.append(float(L_law.item()))

            va_m = float(np.mean(va_losses)) if va_losses else float("inf")
            va_dm = float(np.mean(va_data_losses)) if va_data_losses else float("inf")
            va_lwm = float(np.mean(va_law_losses)) if va_law_losses else 0.0
        else:
            va_m = tr_m
            va_dm = tr_dm
            va_lwm = tr_lwm

        loss_history.append({
            "epoch": ep,
            "train_loss": tr_m,
            "val_loss": va_m
        })

        print(
            f"Ep {ep:04d} | "
            f"train={tr_m:.6e} data={tr_dm:.6e} law={tr_lwm:.6e} | "
            f"val={va_m:.6e} data={va_dm:.6e} law={va_lwm:.6e} | "
            f"ks={float(model.ks_value().item()):.6g}"
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

    csv_path = save_loss_csv_dialog("Save epoch/train/val loss CSV")
    if csv_path:
        pd.DataFrame(loss_history, columns=["epoch", "train_loss", "val_loss"]).to_csv(csv_path, index=False)
        print("Loss CSV saved:", csv_path)

    sv = save_model_dialog("Save trained FiLM-PhyLSTM model (.pt)")
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
            GRAD_CLIP=GRAD_CLIP,
            W_FORCE=W_FORCE,
            W_VEL=W_VEL,
            LOSS_FORCE="MSE",
            LOSS_VEL="SmoothL1"
        ),
        "norm": stats,
        "physics": dict(
            USE_TRUE_V_IN_LAW=USE_TRUE_V_IN_LAW,
            LAMBDA_LAW=LAMBDA_LAW,
            LOG_EXP_CLAMP=LOG_EXP_CLAMP,
            ALPHA_MIN_CLAMP=ALPHA_MIN_CLAMP,
            ALPHA_MAX_CLAMP=ALPHA_MAX_CLAMP,
            learned_ks=float(model.ks_value().detach().cpu().item())
        ),
        "conditioning": dict(
            mode="FiLM-PhyLSTM",
            fixed=True,
            C=C_VALUE,
            alpha=ALPHA_VALUE
        ),
        "notes": (
            "FiLM-PhyLSTM with fixed conditioning constants. "
            "10-column CSV input: time, D1-3, F1-3, V1-3. "
            "Each file creates 3 independent floor-wise samples. "
            "C and alpha are used both as direct input and FiLM conditioning. "
            "Physics law uses learnable ks."
        )
    }, sv)

    print("Model saved:", sv)

    root = _tk_root()
    messagebox.showinfo(
        "Training complete",
        f"Saved model:\n{sv}\n\n"
        f"Best val loss: {best_val:.6e}\n"
        f"Learned ks: {float(model.ks_value().detach().cpu().item()):.6g}",
        parent=root
    )
    root.destroy()


if __name__ == "__main__":
    print("\nFiLM-PhyLSTM — Fixed C/alpha, 10-column, 3 floors independent")
    print("Each file -> 3 independent samples: (D1->F1,V1), (D2->F2,V2), (D3->F3,V3)")
    print(f"Inputs : [D, C, alpha] with fixed constants C={C_VALUE}, alpha={ALPHA_VALUE}")
    print("FiLM   : separate conditioning from [C, alpha]")
    print("Targets: [F, V] from cols 5-10")
    print("Loss   : MSE(F) + SmoothL1(V) + physics law")
    print(f"Physics: lambda_law={LAMBDA_LAW}, use_true_v_in_law={USE_TRUE_V_IN_LAW}")
    train()