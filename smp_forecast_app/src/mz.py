"""한 달 예측 모델 비교 — 기준일 I 15시 판단, 7~34일 뒤 일별 4개 지표(일평균·한낮·저녁·한낮 최저).

후보
  기준: same4(최근 같은 유형일 4일 평균), seas(작년 계절비 보정)
  표 형식 (목표 = 실제 ÷ same4, 예측 = same4 × 비율): lgb·xgb·ridge(월 재학습), cat·rf·et·ebm·mlp·tabpfn(분기)
  시계열: Chronos-2(휴무일 공변량)·Chronos-Bolt — 지표별 일 시계열, 최근 512일 문맥, 34일 예측
          LSTM·GRU — 최근 56일 4개 지표 + 앞으로 34일 달력 → 28일 × 4개 지표 한 번에 (분기 재학습)
  앙상블: wz_ens.ens_candidates와 같은 유형 + '기준 + λ×(앙상블 − 기준)' 섞기
선택: 2025 상반기로 맞춤 → 2025 하반기 비교 → 2025 전체로 재적합 → 2026 한 번 채점(한 달 예측 3차 시험)
범위: model_monthly.interval (주차 × 휴무일, 고정 반년/갱신 90·180일, 가산·배수형), 2025 하반기로 선택
실행: python -m smp_forecast_app.src.mz tab lgb xgb ridge ... | chronos | seq | ens
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT, SEED
from smp_forecast_app.src.features_daily import FEATURES, STATS, daily_stats, load_daily

warnings.filterwarnings("ignore")
OUT = PROCESSED_DIR / "smp_forecast_app" / "mzoo"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
START = "2025-01-01"
FEATS = FEATURES + ["base_ratio_l28"]
KEY = ["base_date", "H"]


def data():
    df = load_daily()
    return df[df["date"] >= "2022-03-01"].reset_index(drop=True)


def periods(df, freq):
    ps = pd.date_range(START, df["base_date"].max(), freq=freq)
    return list(ps) + [pd.Timestamp("2100-01-01")]


def save(name, df):
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(OUT / f"{name}.csv.gz", index=False)


# ---------------------------------------------------------------- 표 형식
def _prep(a, b, onehot):
    X, Xt = a.copy(), b.copy()
    med = X.median()
    X, Xt = X.fillna(med), Xt.fillna(med)
    mu, sd = X.mean(), X.std().replace(0, 1)
    return (X - mu) / sd, (Xt - mu) / sd


def make(name):
    from catboost import CatBoostRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    return {"lgb": lambda: LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=60, subsample=0.8,
                                         subsample_freq=1, colsample_bytree=0.8, objective="l1", random_state=SEED, verbose=-1),
            "xgb": lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                        objective="reg:absoluteerror", tree_method="hist", random_state=SEED, n_jobs=4),
            "cat": lambda: CatBoostRegressor(iterations=400, depth=6, learning_rate=0.06, loss_function="MAE", random_seed=SEED,
                                             verbose=0, thread_count=4),
            "rf": lambda: RandomForestRegressor(300, min_samples_leaf=10, max_features=0.5, random_state=SEED, n_jobs=4),
            "et": lambda: ExtraTreesRegressor(300, min_samples_leaf=10, max_features=0.5, random_state=SEED, n_jobs=4),
            "ridge": lambda: Ridge(alpha=10),
            "ebm": lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, outer_bags=4, n_jobs=2),
            "mlp": lambda: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=200, early_stopping=True, random_state=SEED)}[name]


FREQ = {"lgb": "MS", "xgb": "MS", "ridge": "MS", "cat": "QS", "rf": "QS", "et": "QS", "ebm": "QS", "mlp": "QS", "tabpfn": "QS"}
SCALE = {"ridge", "mlp", "rf", "et", "ebm"}


def run_tab(name):
    df = data()
    out = df[KEY + ["date"]].copy()
    for s in STATS:
        d = df.assign(base_ratio_l28=df[f"{s}_same4"] / df[f"{s}_l28"].clip(lower=10))
        base = d[f"{s}_same4"].clip(lower=10)
        yt = (d[f"y_{s}"] / base).clip(0, 3)
        pred = pd.Series(np.nan, index=d.index)
        ps = periods(d, FREQ[name])
        for a, b in zip(ps[:-1], ps[1:]):
            tr = (d["date"] < a) & yt.notna()
            te = (d["base_date"] >= a) & (d["base_date"] < b)
            if te.sum() == 0:
                continue
            A, B = d.loc[tr, FEATS], d.loc[te, FEATS]
            if name == "tabpfn":
                from tabpfn import TabPFNRegressor
                idx = d.index[tr][-4000:]
                p = TabPFNRegressor(device="cpu", random_state=SEED, ignore_pretraining_limits=True).fit(
                    d.loc[idx, FEATS], yt[idx]).predict(B)
            else:
                if name in SCALE:
                    A, B = _prep(A, B, False)
                p = make(name)().fit(A, yt[tr]).predict(B)
            pred[te] = p * base[te].values
        out[f"pred_{s}"] = pred
        print(f"{name} {s} 완료", flush=True)
    save(name, out[out["base_date"] >= START])


def run_base():
    df = data()
    for name, f in (("same4", lambda s: df[f"{s}_same4"]), ("seas", lambda s: df[f"{s}_same4"] * df["seas_smp"].clip(0.6, 1.6).fillna(1))):
        out = df[KEY + ["date"]].copy()
        for s in STATS:
            out[f"pred_{s}"] = f(s)
        save(name, out[out["base_date"] >= START])


# ---------------------------------------------------------------- Chronos (일 시계열)
def run_chronos(tags=("c2", "bolt")):
    import torch
    from chronos import BaseChronosPipeline
    from common.calendar import build_calendar
    st = daily_stats()
    days = st.index
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=40)).set_index("date")
    off = cal["is_offday"].astype("float32")
    df = data()
    bases = sorted(df.loc[df["base_date"] >= START, "base_date"].unique())
    for tag, mname in (("c2", "amazon/chronos-2"), ("bolt", "amazon/chronos-bolt-small")):
        if tag not in tags or (OUT / f"chr_{tag}.csv.gz").exists():
            continue
        pipe = BaseChronosPipeline.from_pretrained(mname, device_map="cpu", torch_dtype=torch.float32)
        res = {}
        for s in STATS:
            ser = st[s].astype("float32")
            for k in range(0, len(bases), 64):
                batch = bases[k:k + 64]
                inputs = []
                for I in batch:
                    ctx = ser[:I].values[-512:]
                    if tag == "c2":
                        cidx = ser[:I].index[-512:]
                        fut = pd.date_range(I + pd.Timedelta(days=1), periods=34)
                        inputs.append({"target": ctx, "past_covariates": {"off": off.reindex(cidx).values},
                                       "future_covariates": {"off": off.reindex(fut).values}})
                    else:
                        inputs.append(torch.tensor(ctx))
                q, _ = pipe.predict_quantiles(inputs, prediction_length=34, quantile_levels=[0.5])
                for I, qq in zip(batch, q):
                    qq = (qq[0] if qq.dim() == 3 else qq).numpy()[:, 0]
                    for H in range(7, 35):
                        res.setdefault((pd.Timestamp(I), H), {})[f"pred_{s}"] = float(qq[H - 1])
            print(f"chronos {tag} {s} 완료", flush=True)
        out = pd.DataFrame([{"base_date": k[0], "H": k[1], **v} for k, v in res.items()])
        out["date"] = out["base_date"] + pd.to_timedelta(out["H"], unit="D")
        save(f"chr_{tag}", out)


# ---------------------------------------------------------------- LSTM·GRU (일별)
def seq_data():
    """기준일별 입력: 최근 56일 4개 지표(÷최근 28일 일평균) + 7~34일 뒤 달력 → 목표 28일 × 4개 지표(÷같은 수준)."""
    from common.calendar import build_calendar
    st = daily_stats()
    days = st.index
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=40)).set_index("date")
    cv = pd.DataFrame({"off": cal["is_offday"].astype(float), "dt": cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}) / 3,
                       "mj": cal["is_myeongjeol"].astype(float), "dow_s": np.sin(2 * np.pi * cal.index.dayofweek / 7),
                       "dow_c": np.cos(2 * np.pi * cal.index.dayofweek / 7), "doy_s": np.sin(2 * np.pi * cal.index.dayofyear / 365.25),
                       "doy_c": np.cos(2 * np.pi * cal.index.dayofyear / 365.25)}, index=cal.index)
    V = st[STATS].values.astype("float32")
    pos = {d: i for i, d in enumerate(days)}
    Xp, Xf, Y, L, B = [], [], [], [], []
    for I in days[60:]:
        i = pos[I]
        lvl = np.nanmean(V[i - 27:i + 1, 0])
        past = V[i - 55:i + 1] / lvl
        if np.isnan(past).any():
            continue
        fut = cv.reindex(pd.date_range(I + pd.Timedelta(days=7), periods=28)).values
        tgt = np.full((28, 4), np.nan, dtype="float32")
        for h in range(28):
            j = i + 7 + h
            if j < len(V):
                tgt[h] = V[j] / lvl
        Xp.append(past); Xf.append(np.nan_to_num(fut)); Y.append(tgt); L.append(lvl); B.append(I)
    return (np.array(Xp, "float32"), np.array(Xf, "float32"), np.array(Y, "float32"), np.array(L, "float32"),
            pd.DatetimeIndex(B))


def seq_net(kind, nf):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(4, 64, 2, batch_first=True, dropout=0.1)
            self.dec = nn.Sequential(nn.Linear(64 + nf, 64), nn.ReLU(), nn.Linear(64, 4))

        def forward(self, xp, xf):
            _, h = self.rnn(xp)
            h = (h[0] if kind == "lstm" else h)[-1]
            return self.dec(torch.cat([h.unsqueeze(1).expand(-1, xf.shape[1], -1), xf], -1))
    return Net()


def seq_fit(kind, Xp, Xf, Y, tr, seed):
    """tr(학습 기준일 위치, 시간 순)로 학습, 마지막 10%로 조기 종료 → 학습된 모델."""
    import torch
    torch.set_num_threads(4)
    t = lambda a: torch.tensor(a)
    torch.manual_seed(seed); np.random.seed(seed)
    m = seq_net(kind, Xf.shape[2])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    nv = max(30, len(tr) // 10)
    tri, vai = tr[:-nv], tr[-nv:]
    best, state, bad = 1e9, None, 0
    for ep in range(80):
        m.train()
        for k in range(0, len(tri), 64):
            bb = np.random.permutation(tri)[k:k + 64]
            y = t(Y[bb]); mk = ~torch.isnan(y)
            loss = (m(t(Xp[bb]), t(Xf[bb])) - torch.nan_to_num(y)).abs()[mk].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            y = t(Y[vai]); mk = ~torch.isnan(y)
            v = (m(t(Xp[vai]), t(Xf[vai])) - torch.nan_to_num(y)).abs()[mk].mean().item()
        if v < best - 1e-4:
            best, state, bad = v, {k2: x.clone() for k2, x in m.state_dict().items()}, 0
        elif (bad := bad + 1) >= 10:
            break
    m.load_state_dict(state); m.eval()
    return m


def seq_predict(models, Xp, Xf, idx):
    import torch
    with torch.no_grad():
        return np.mean([m(torch.tensor(Xp[idx]), torch.tensor(Xf[idx])).numpy() for m in models], axis=0)


def run_seq(kinds=("lstm", "gru")):
    Xp, Xf, Y, L, B = seq_data()
    ps = periods(pd.DataFrame({"base_date": B}), "QS")
    ps = [pd.Timestamp("2024-12-25")] + ps[1:] if ps[0] > pd.Timestamp("2024-12-25") else ps
    for kind in kinds:
        if (OUT / kind).with_suffix(".csv.gz").exists():
            continue
        rows = []
        for a, b in zip(ps[:-1], ps[1:]):
            tr = np.where(B + pd.Timedelta(days=34) < a)[0]
            te = np.where((B >= a) & (B < b))[0]
            if len(te) == 0:
                continue
            ms = [seq_fit(kind, Xp, Xf, Y, tr, SEED + sd) for sd in range(3)]
            p = seq_predict(ms, Xp, Xf, te) * L[te, None, None]
            for j, I in enumerate(B[te]):
                for h in range(28):
                    rows.append({"base_date": I, "H": h + 7, **{f"pred_{s}": float(p[j, h, k]) for k, s in enumerate(STATS)}})
            print(f"{kind} 기간 {a.date()} 완료", flush=True)
        out = pd.DataFrame(rows)
        out["date"] = out["base_date"] + pd.to_timedelta(out["H"], unit="D")
        save(kind, out[out["base_date"] >= START])


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "tab":
        for n in args:
            if n == "base":
                run_base()
            elif not (OUT / f"{n}.csv.gz").exists():
                t0 = time.time(); run_tab(n); print(f"{n} ({time.time()-t0:.0f}s)", flush=True)
    elif cmd == "chronos":
        run_chronos(tuple(args) or ("c2", "bolt"))
    elif cmd == "seq":
        run_seq(tuple(args) or ("lstm", "gru"))
