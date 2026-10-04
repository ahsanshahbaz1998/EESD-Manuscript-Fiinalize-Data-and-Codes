# -*- coding: utf-8 -*-

import os
import random
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tkinter import Tk, filedialog


C0_VALUE = 696.0
ALPHA0_VALUE = 0.44

USE_TRUE_V_IN_LAW = False

C_MULTS = [0.5, 1.25]
A_CASES = [0.30, 0.80]

HIDDEN = 128
LAYERS = 3
DROPOUT = 0.05

IN_DIM = 3
CA_DIM = 2
OUT_DIM = 2

LR = 3e-3
WD = 3e-4
BATCH = 16
EPOCHS = 2000
PATIENCE = 300
VAL_FRAC = 0.20
GRAD_CLIP = 1.0

W_FORCE = 1.0
W_VEL = 1.0

LAMBDA_LAW = 0.5

LOG_EXP_CLAMP = 50.0
SGN_EPS = 1e-12
ALPHA_MIN_CLAMP = 0.15
ALPHA_MAX_CLAMP = 1.20

EPS = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def _tk_dialog(callable_dialog):
    root = Tk()
    root.withdraw()

    try:
        root.update_idletasks()

        try:
            root.attributes("-topmost", True)
            root.update()
        except Exception:
            pass

        return callable_dialog(root)

    finally:
        try:
            root.destroy()
        except Exception:
            pass


def pick_many_csv(title):
    paths = _tk_dialog(lambda r: filedialog.askopenfilenames(
        title=title,
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
    ))

    return list(paths) if paths else []


def pick_out_dir(title="Select output folder (logs)"):
    p = _tk_dialog(lambda r: filedialog.askdirectory(title=title))
    return p if p else ""


def save_model_dialog(title="Save trained model (.pt)"):
    p = _tk_dialog(lambda r: filedialog.asksaveasfilename(
        title=title,
        defaultextension=".pt",
        filetypes=[("PyTorch", "*.pt")]
    ))

    return p if p else ""


def make_case_list(C0=C0_VALUE):
    cases = []

    for a in A_CASES:
        for mC in C_MULTS:
            Cc = float(C0) * float(mC)
            aa = float(a)
            lab = f"C{Cc:.1f}_a{aa:.2f}"
            cases.append((Cc, aa, lab))

    assert len(cases) == 4
    return cases


CASES4 = make_case_list()

print("Cases:")
for i, (Cc, aa, lab) in enumerate(CASES4, 1):
    print(f"  case_{i:02d}: {lab}")


def extract_base_id_from_stem(stem: str) -> str:
    m = re.search(r"^(.*?)__floor\d+__", stem, flags=re.IGNORECASE)

    if m:
        return m.group(1)

    s = re.sub(r"__C\d+_a\d+.*$", "", stem, flags=re.IGNORECASE)
    return s


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


def find_col(df: pd.DataFrame, candidates):
    cols = [str(c) for c in df.columns]
    direct = {c: c for c in cols}

    for cand in candidates:
        if cand in direct:
            return cand

    low = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]

    return None


def median_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)

    if t.size < 3:
        return float("nan")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        return float("nan")

    return float(np.median(dt))


def read_case_file_floor1_6col(csv_path: str):
    df = pd.read_csv(csv_path)

    if df.shape[1] < 6:
        raise ValueError("This FiLM-PhyLSTM script requires 6-column files containing C and alpha in the CSV.")

    c_time = find_col(df, ["time_sec", "time"])
    c_C = find_col(df, ["C", "c", "C_used", "C_case"])
    c_a = find_col(df, ["alpha", "Alpha", "a", "alpha_used", "alpha_case"])
    c_D = find_col(df, ["D_mm", "D1_mm", "D1", "D"])
    c_V = find_col(df, ["V_mps", "V1_mps", "V1", "V"])
    c_F = find_col(df, ["F_kN", "F_case_kN", "Fcase_kN", "F_case", "F"])

    if any(c is None for c in [c_time, c_C, c_a, c_D, c_V, c_F]):
        arr = df.iloc[:, :6].to_numpy(np.float64)

        t = arr[:, 0]
        Cc = arr[:, 1]
        aa = arr[:, 2]
        D = arr[:, 3]
        V = arr[:, 4]
        Fc = arr[:, 5]

    else:
        t = pd.to_numeric(df[c_time], errors="coerce").to_numpy(np.float64)
        Cc = pd.to_numeric(df[c_C], errors="coerce").to_numpy(np.float64)
        aa = pd.to_numeric(df[c_a], errors="coerce").to_numpy(np.float64)
        D = pd.to_numeric(df[c_D], errors="coerce").to_numpy(np.float64)
        V = pd.to_numeric(df[c_V], errors="coerce").to_numpy(np.float64)
        Fc = pd.to_numeric(df[c_F], errors="coerce").to_numpy(np.float64)

    m = (
        np.isfinite(t) &
        np.isfinite(Cc) &
        np.isfinite(aa) &
        np.isfinite(D) &
        np.isfinite(V) &
        np.isfinite(Fc)
    )

    if m.sum() < 20:
        raise ValueError("Too few valid rows (<20)")

    t = t[m]
    Cc = Cc[m]
    aa = aa[m]
    D = D[m]
    V = V[m]
    Fc = Fc[m]

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)

        t = t[idx]
        Cc = Cc[idx]
        aa = aa[idx]
        D = D[idx]
        V = V[idx]
        Fc = Fc[idx]

    dt_med = median_dt(t)

    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 1.0 / 102.4

    C_used = float(np.median(Cc))
    a_used = float(np.median(aa))
    a_used = float(np.clip(a_used, ALPHA_MIN_CLAMP, ALPHA_MAX_CLAMP))

    return (
        t.astype(np.float64),
        D.astype(np.float32),
        V.astype(np.float32),
        Fc.astype(np.float32),
        float(dt_med),
        float(C_used),
        float(a_used),
    )


def collect_case_files_one_by_one(tag: str, cases4):
    collected = []

    for i, (_Cc, _aa, lab) in enumerate(cases4, start=1):
        title = f"[{tag}] Select 6-col CSV files for case_{i:02d}  {lab}"
        files = pick_many_csv(title)

        if not files:
            print(f"{tag} case_{i:02d} skipped because no files were selected.")
            continue

        for fp in files:
            collected.append((fp, str(lab), str(tag)))

        print(f"{tag} case_{i:02d}: selected {len(files)} file(s)")

    return collected


class CaseFilesDatasetFloor1(Dataset):
    def __init__(self, file_tuples):
        self.items = []

        for (fp, lab, tag) in file_tuples:
            try:
                t, D, Vtrue, Fcase, dt_med, C_used, a_used = read_case_file_floor1_6col(fp)
            except Exception as e:
                print("Skip:", os.path.basename(fp), "|", str(e))
                continue

            L = len(t)

            Ccol = np.full((L, 1), float(C_used), dtype=np.float32)
            Acol = np.full((L, 1), float(a_used), dtype=np.float32)

            X = np.hstack([
                D.reshape(-1, 1),
                Ccol,
                Acol
            ]).astype(np.float32)

            Y = np.hstack([
                Fcase.reshape(-1, 1),
                Vtrue.reshape(-1, 1)
            ]).astype(np.float32)

            meta = dict(
                file=os.path.basename(fp),
                tag=tag,
                case=lab,
                C_used=C_used,
                alpha_used=a_used
            )

            self.items.append((X, Y, float(dt_med), meta))

        if len(self.items) == 0:
            raise ValueError("No valid sequences built. Use 6-column files: time,C,alpha,D,V,F.")

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
            dropout=dropout if layers > 1 else 0.0,
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

        self._ks_raw = nn.Parameter(
            torch.tensor(_inv_softplus(ks_init), dtype=torch.float32)
        )

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
    dbe_tuples = collect_case_files_one_by_one("DBE", CASES4)
    mce_tuples = collect_case_files_one_by_one("MCE", CASES4)

    if len(dbe_tuples) == 0 and len(mce_tuples) == 0:
        print("No files selected. Exit.")
        return

    dbe_files_only = [t[0] for t in dbe_tuples]
    mce_files_only = [t[0] for t in mce_tuples]

    dbe_tr_files, dbe_va_files = split_files(
        dbe_files_only,
        VAL_FRAC,
        seed=SEED
    )

    mce_tr_files, mce_va_files = split_files(
        mce_files_only,
        VAL_FRAC,
        seed=SEED
    )

    def filter_tuples(tuples, keep_files_set):
        keep = set(keep_files_set)
        return [t for t in tuples if t[0] in keep]

    train_tuples = (
        filter_tuples(dbe_tuples, dbe_tr_files) +
        filter_tuples(mce_tuples, mce_tr_files)
    )

    val_tuples = (
        filter_tuples(dbe_tuples, dbe_va_files) +
        filter_tuples(mce_tuples, mce_va_files)
    )

    ds_tr_raw = CaseFilesDatasetFloor1(train_tuples)

    stats = compute_stats_from_items(ds_tr_raw.items)

    Xmu = stats["X_mean"]
    Xsd = stats["X_std"]
    Ymu = stats["Y_mean"]
    Ysd = stats["Y_std"]

    tr_list = [
        (norm_arr(X, Xmu, Xsd), norm_arr(Y, Ymu, Ysd), dt, meta)
        for (X, Y, dt, meta) in ds_tr_raw.items
    ]

    va_list = []

    if len(val_tuples) > 0:
        ds_va_raw = CaseFilesDatasetFloor1(val_tuples)

        va_list = [
            (norm_arr(X, Xmu, Xsd), norm_arr(Y, Ymu, Ysd), dt, meta)
            for (X, Y, dt, meta) in ds_va_raw.items
        ]

    def collate_norm(batch):
        return collate_fullseq([
            (x, y, dt, meta)
            for (x, y, dt, meta) in batch
        ])

    tr_loader = DataLoader(
        tr_list,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collate_norm
    )

    va_loader = DataLoader(
        va_list,
        batch_size=BATCH,
        shuffle=False,
        collate_fn=collate_norm
    ) if len(va_list) else None

    model = FiLMPhyLSTM_HV(ks_init=1.0).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD
    )

    mse = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")

    Xmu_t = torch.tensor(
        Xmu,
        dtype=torch.float32,
        device=device
    ).view(1, 1, IN_DIM)

    Xsd_t = torch.tensor(
        Xsd,
        dtype=torch.float32,
        device=device
    ).view(1, 1, IN_DIM)

    Ymu_t = torch.tensor(
        Ymu,
        dtype=torch.float32,
        device=device
    ).view(1, 1, OUT_DIM)

    Ysd_t = torch.tensor(
        Ysd,
        dtype=torch.float32,
        device=device
    ).view(1, 1, OUT_DIM)

    best_val = float("inf")
    patience = PATIENCE
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []

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
            V_phys = Yp_phys[:, :, 1:2]
            Vtrue_phys = Yt_phys[:, :, 1:2]

            ks_learn = model.ks_value()

            L_law = law_loss_relmse_learnableKs(
                F_phys,
                V_phys,
                Vtrue_phys,
                C_phys,
                A_phys,
                m,
                dtb,
                ks_learn
            )

            loss = L_data + LAMBDA_LAW * L_law

            loss = torch.nan_to_num(
                loss,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            opt.step()

            tr_losses.append(float(loss.item()))

        tr_m = float(np.mean(tr_losses)) if tr_losses else float("inf")

        if va_loader is not None:
            model.eval()
            va_losses = []

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
                    V_phys = Yp_phys[:, :, 1:2]
                    Vtrue_phys = Yt_phys[:, :, 1:2]

                    ks_learn = model.ks_value()

                    L_law = law_loss_relmse_learnableKs(
                        F_phys,
                        V_phys,
                        Vtrue_phys,
                        C_phys,
                        A_phys,
                        m,
                        dtb,
                        ks_learn
                    )

                    loss = L_data + LAMBDA_LAW * L_law

                    va_losses.append(float(loss.item()))

            va_m = float(np.mean(va_losses)) if va_losses else float("inf")

        else:
            va_m = tr_m

        print(
            f"Ep {ep:04d} | "
            f"train={tr_m:.6e} | "
            f"val={va_m:.6e} | "
            f"ks={float(model.ks_value().item()):.6g}"
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

    out_dir = pick_out_dir("Select output folder (logs)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sv = save_model_dialog("Save trained model (.pt)")

    if not sv:
        return

    torch.save({
        "state": model.state_dict(),
        "norm": stats,
        "notes": "FiLM-PhyLSTM: C and alpha used both as direct input [D,C,alpha] and as separate FiLM conditioning."
    }, sv)

    print("Model saved:", sv)


if __name__ == "__main__":
    train()