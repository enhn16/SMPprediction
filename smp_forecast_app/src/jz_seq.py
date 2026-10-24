"""제주 일주일 예측 — 시계열 신경망 (168시간 한 번에). 육지 wz_seq와 같은 구조, 기준일 I 15시 판단.

입력 (모두 I일까지 확정된 값 / 사전 확정 값)
  과거: I-6 ~ I일 시간별 제주 SMP·제주 하루전 수요예측·육지 SMP (168시간), SMP는 최근 7일 제주 평균(20원 하한)으로 나눔
  미래(I+1 ~ I+7, 시간별): 시간·요일 순환값 + 제주 피처 표(jz_features H별)의 달력·제주 예보·Open-Meteo 풍속·일사,
       같은 기준일의 육지 일주일 예측(L_base, 수준으로 나눔)
출력: I+1 ~ I+7 시간별 제주 SMP ÷ 수준 (L1 손실)
재학습: 분기마다 (기준일 기간 [a, b)의 모델은 대상일 < a 실적으로 학습). 시드 2개 평균. 학습 기준일 2023-01 ~ (육지 예측이 있는 기간)
실행: python -m smp_forecast_app.src.jz_seq jlstm jgru jcnn jtransformer
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.config import SEED
from common.loaders import load_smp_actual, load_smp_forecast_api
from smp_forecast_app.src.jz_common import HS, OUT, feats, save

torch.set_num_threads(6)
PAST = 7   # 2026-10-21: 14일은 CPU에서 너무 느려(구간당 40분+) 7일로 줄임
FUT_COLS = ["offday", "daytype", "is_myeongjeol", "prev_offday", "next_offday", "jf_sky_pm", "jf_pop_pm", "jf_sky_am",
            "kpx_peak_rel", "fc_lng", "jo_rad_mid", "jo_ws100_day", "jo_ws100_mid"]
NCH = 3   # 과거 통로: 제주 SMP, 제주 수요, 육지 SMP
PERIODS = ["2024-12-25", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01"]


def build_tensors():
    s = load_smp_actual("제주").pivot_table(index="date", columns="hour", values="smp")
    ls = load_smp_actual("육지").pivot_table(index="date", columns="hour", values="smp")
    d = load_smp_forecast_api()
    d = d[d["region"] == "제주"].pivot_table(index="date", columns="hour", values="demand_fcst")
    days = pd.date_range(s.index.min(), s.index.max() + pd.Timedelta(days=7))
    S, Dm, LS = s.reindex(days).values, d.reindex(days).values, ls.reindex(days).values
    # 미래 공변량: 지평 H 표에서 대상일(I+H) 행
    fut = {}
    for H in HS:
        f = feats(H).set_index(["date", "hour"])
        fut[H] = f[FUT_COLS + ["L_base"]]
    bases = pd.date_range("2023-01-01", s.index.max())
    pos = {d: i for i, d in enumerate(days)}
    Xp, Xf, Y, L, B = [], [], [], [], []
    hour = np.arange(1, 25)
    hs, hc = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    stats = {c: (pd.concat([fut[H][c] for H in HS]).mean(), pd.concat([fut[H][c] for H in HS]).std() or 1) for c in FUT_COLS}
    for I in bases:
        i = pos[I]
        past_s, past_d = S[i - PAST + 1:i + 1].reshape(-1), Dm[i - PAST + 1:i + 1].reshape(-1)
        past_l = LS[i - PAST + 1:i + 1].reshape(-1)
        if np.isnan(past_s).any() or np.isnan(past_l).any():
            continue
        lvl = max(np.nanmean(S[i - 6:i + 1]), 20.0)
        dl = np.nanmean(Dm[i - 6:i + 1])
        past_d = np.nan_to_num(past_d / dl, nan=1.0)
        rows, ys = [], []
        for H in HS:
            D = I + pd.Timedelta(days=H)
            dow = D.dayofweek
            try:
                fv = fut[H].loc[D].reindex(hour)
            except KeyError:
                fv = pd.DataFrame(np.nan, index=hour, columns=FUT_COLS + ["L_base"])
            if fv["L_base"].isna().any():
                break
            z = np.column_stack([(fv[c].values - stats[c][0]) / stats[c][1] for c in FUT_COLS])
            z = np.nan_to_num(z, nan=0.0)
            cal = np.column_stack([hs, hc, np.full(24, np.sin(2 * np.pi * dow / 7)), np.full(24, np.cos(2 * np.pi * dow / 7)),
                                   np.full(24, H / 7)])
            rows.append(np.hstack([cal, z, (fv["L_base"].values / lvl)[:, None]]))
            ys.append(S[pos[D]] / lvl if D in pos else np.full(24, np.nan))
        if len(rows) < len(HS):   # 육지 예측이 없는 기준일은 제외
            continue
        Xp.append(np.stack([past_s / lvl, past_d, past_l / lvl], axis=1))
        Xf.append(np.vstack(rows))
        Y.append(np.concatenate(ys))
        L.append(lvl)
        B.append(I)
    return (np.array(Xp, dtype="float32"), np.array(Xf, dtype="float32"), np.array(Y, dtype="float32"),
            np.array(L, dtype="float32"), pd.DatetimeIndex(B))


class Net(nn.Module):
    def __init__(self, kind, nf, hid=48):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.enc = (nn.LSTM if kind == "lstm" else nn.GRU)(NCH, hid, num_layers=1, batch_first=True)
        elif kind == "cnn":
            self.enc = nn.Sequential(nn.Conv1d(NCH, hid, 5, padding=2), nn.ReLU(), nn.Conv1d(hid, hid, 5, padding=4, dilation=2),
                                     nn.ReLU(), nn.Conv1d(hid, hid, 5, padding=8, dilation=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        else:
            self.inp = nn.Linear(NCH, hid)
            self.pos = nn.Parameter(torch.randn(1, PAST * 24, hid) * 0.02)
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid, 4, 96, 0.1, batch_first=True), 1)
        self.dec = nn.Sequential(nn.Linear(hid + nf, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xp, xf):
        if self.kind in ("lstm", "gru"):
            _, h = self.enc(xp)
            h = (h[0] if self.kind == "lstm" else h)[-1]
        elif self.kind == "cnn":
            h = self.enc(xp.transpose(1, 2)).squeeze(-1)
        else:
            h = self.enc(self.inp(xp) + self.pos).mean(1)
        h = h.unsqueeze(1).expand(-1, xf.shape[1], -1)
        return self.dec(torch.cat([h, xf], -1)).squeeze(-1)


def fit_predict(kind, Xp, Xf, Y, tr, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Net(kind, Xf.shape[2])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    idx = np.where(tr)[0]
    n_val = max(30, len(idx) // 10)
    tri, vai = idx[:-n_val], idx[-n_val:]            # 시간 순서: 마지막 10%로 조기 종료
    t = lambda a: torch.tensor(a)
    best, best_state, bad = 1e9, None, 0
    for ep in range(30):
        m.train()
        perm = np.random.permutation(tri)
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            y = t(Y[b])
            mask = ~torch.isnan(y)
            loss = (m(t(Xp[b]), t(Xf[b])) - torch.nan_to_num(y)).abs()[mask].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            y = t(Y[vai])
            mask = ~torch.isnan(y)
            v = (m(t(Xp[vai]), t(Xf[vai])) - torch.nan_to_num(y)).abs()[mask].mean().item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return m(t(Xp[te]), t(Xf[te])).numpy()


def main(kinds):
    t0 = time.time()
    Xp, Xf, Y, L, B = build_tensors()
    print(f"표본 {len(B)} 기준일, 입력 {Xp.shape}, 미래 {Xf.shape} ({time.time()-t0:.0f}s)", flush=True)
    ps = [pd.Timestamp(p) for p in PERIODS] + [pd.Timestamp("2100-01-01")]
    last_target = {I: I + pd.Timedelta(days=7) for I in B}
    for kind in kinds:
        if (OUT / f"{kind}.csv.gz").exists():
            print(f"{kind}: 이미 있음 → 건너뜀")
            continue
        rows = []
        for a, b in zip(ps[:-1], ps[1:]):
            tr = np.array([last_target[I] < a for I in B])          # 대상일 전부 a 이전 = 학습 가능
            te = np.where((B >= a) & (B < b))[0]
            if len(te) == 0:
                continue
            preds = np.mean([fit_predict(kind[1:], Xp, Xf, Y, tr, te, SEED + k) for k in range(2)], axis=0)
            preds = preds * L[te, None]
            for j, I in zip(range(len(te)), B[te]):
                for H in HS:
                    D = I + pd.Timedelta(days=H)
                    for h in range(24):
                        rows.append((D, h + 1, H, float(preds[j, (H - 1) * 24 + h])))
            print(f"{kind} 기간 {a.date()} ({len(te)}일, {time.time()-t0:.0f}s)", flush=True)
        out = pd.DataFrame(rows, columns=["date", "hour", "H", "pred"])
        save(kind, out[out["date"] >= "2025-01-01"])
        print(f"{kind} 완료 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["jlstm", "jgru", "jcnn", "jtransformer"])
