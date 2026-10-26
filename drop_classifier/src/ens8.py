"""팀원식 8개 결합을 직접 재현해 현재 모델과 같은 조건으로 비교 (육지 급락, 1차 15시).

조건 (현재 모델 final_model.py와 동일)
  피처: 기본+주간전망+보강+태양광설비 (BASE+SOLAR). 확장 변수 = + 연료비 + 최근 7일 SMP 수준
  검증: 2024 학습 → 2025 예측 → 문턱 (주의 = 2025 급락 90%, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨)
  시험: 2024~2025 학습 → 2026-01-01 ~ 10-18, 휴일·주말만 채점 (team_scorecard.py 지표)
8개 구성원 (모두 분류 확률, 규제 강하게 고정 — 튜닝 없음)
  LightGBM, LightGBM(확장 변수), LSTM(시간별 SMP 168시간 + 표 피처), 랜덤포레스트, 엑스트라트리,
  로지스틱, 의사결정나무, 결합(앞 7개 예측을 받는 로지스틱 메타모델, 학습 구간 마지막 25%로 시간 순서 적합)
  최종 = 8개 확률 같은 가중치 평균. 확률 보정은 2025 검증 예측에 로지스틱.
주의: 2026 시험 구간은 이미 열람됨 → 이 비교는 모델 선택이 아니라 발표용 참고 비교.
팀 규칙 (--team, 2026-10-26 팀 통일): 2023~2024 학습 → 2025 문턱, 2023~2025 학습 → 2026. 문턱·채점 모두 휴일·주말만.
  2023 결측(날씨 예보·주간전망)은 학습 구간 중앙값 (부스팅은 빈칸 그대로). 데이터는 team_select.data23().
출력: outputs/tables/20_ens8_vs_current_육지{,_team}.csv, 20_ens8_daily_육지{,_team}.csv
실행: python -m drop_classifier.src.ens8 [--team]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

from common.config import SEED
from drop_classifier.src.fair_select import fcst15
from drop_classifier.src.final import threshold_for
from drop_classifier.src.fuel_level import BASE, FUEL, LEVEL, fuel_features, level_features
from drop_classifier.src.search import day_table
from drop_classifier.src.supply_level import SOLAR, supply_features
from drop_classifier.src.team_select import boot, data23, imp
from drop_classifier.src.two_tier import OUT_TAB, strong_threshold
from drop_classifier.src.zoo import BASES as ZOO, add_seq_and_chronos
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
COLS = BASE + SOLAR
COLS_X = COLS + FUEL + LEVEL
TR_VAL, VALID = ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")
TR_FIN, TEST = ("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-10-18")
TEAM = {"TR_VAL": ("2023-01-01", "2024-12-31"), "TR_FIN": ("2023-01-01", "2025-12-31")}


def data(team=False) -> pd.DataFrame:
    if team:
        df = data23()
        df = pd.concat([df, level_features(df["date"])], axis=1)
    else:
        df = day_table().merge(fcst15(), on="date", how="left")
        df = pd.concat([df.reset_index(drop=True), supply_features(df["date"]), fuel_features(df["date"]),
                        level_features(df["date"])], axis=1)
    df = add_seq_and_chronos(df)
    df["offday"] = df["day_type"] != "평일"
    return df


def sk(make, cols=COLS):
    def f(tr, te):
        return make().fit(tr[cols], tr["label"]).predict_proba(te[cols])[:, 1]
    return f


class LSTMCls(torch.nn.Module):
    def __init__(self, n_tab, h=16):
        super().__init__()
        self.rnn = torch.nn.LSTM(1, h, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(h + n_tab, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                        torch.nn.Linear(16, 1))

    def forward(self, seq, tab):
        return self.head(torch.cat([self.rnn(seq.unsqueeze(-1))[0][:, -1], tab], 1)).squeeze(-1)


def lstm(tr, te, seeds=(0, 1, 2)):
    """규제: 은닉 16, 드롭아웃 0.3, 가중치 감쇠 1e-2, 학습 구간 마지막 1/7로 조기 종료, 시드 3개 평균."""
    sx = SimpleImputer(strategy="median").fit(tr[COLS])
    sc = StandardScaler().fit(sx.transform(tr[COLS]))
    tab = lambda d: torch.tensor(sc.transform(sx.transform(d[COLS])), dtype=torch.float32)
    allseq = np.stack(tr["seq"])
    mu, sd = np.nanmean(allseq), np.nanstd(allseq)
    seq = lambda d: torch.nan_to_num(torch.tensor((np.stack(d["seq"]) - mu) / sd, dtype=torch.float32))
    y = torch.tensor(tr["label"].values, dtype=torch.float32)
    Xs, Xt, Ts, Tt = seq(tr), tab(tr), seq(te), tab(te)
    n_val = max(30, len(tr) // 7)
    pw = torch.tensor((y[:-n_val] == 0).sum() / max((y[:-n_val] == 1).sum(), 1))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    preds = []
    for s in seeds:
        torch.manual_seed(SEED + s)
        net = LSTMCls(Xt.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        best, state, bad = 1e9, None, 0
        for ep in range(150):
            net.train()
            perm = torch.randperm(len(y) - n_val)
            for b in range(0, len(perm), 32):
                i = perm[b:b + 32]
                opt.zero_grad()
                loss_fn(net(Xs[i], Xt[i]), y[i]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xs[-n_val:], Xt[-n_val:]), y[-n_val:]).item()
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 15:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(net(Ts, Tt)).numpy())
    return np.mean(preds, axis=0)


LGBM = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5, class_weight="balanced", random_state=SEED, verbose=-1)
MEMBERS = {
    "LightGBM": sk(lambda: LGBMClassifier(**LGBM)),
    "LightGBM(확장 변수)": sk(lambda: LGBMClassifier(**LGBM), COLS_X),
    "LSTM": lstm,
    "랜덤포레스트": sk(lambda: imp(RandomForestClassifier(500, min_samples_leaf=5, max_features="sqrt",
                                                      class_weight="balanced_subsample", random_state=SEED, n_jobs=4))),
    "엑스트라트리": sk(lambda: imp(ExtraTreesClassifier(500, min_samples_leaf=5, max_features="sqrt",
                                                    class_weight="balanced_subsample", random_state=SEED, n_jobs=4))),
    "로지스틱": sk(lambda: imp(LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))),
    "의사결정나무": sk(lambda: imp(DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced",
                                                     random_state=SEED))),
}


def ens8(tr, te):
    """7개 구성원 + 결합(스태킹 메타모델) → 8개 확률 평균."""
    p = {k: f(tr, te) for k, f in MEMBERS.items()}
    cut = tr["date"].quantile(0.75)                               # 학습 구간 마지막 25%로 메타모델 적합 (시간 순서)
    a, b = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    print(f"    결합 메타모델 적합 구간 {b['date'].min():%Y-%m-%d}~, 급락 {int(b['label'].sum())}일", flush=True)
    meta_X = np.column_stack([f(a, b) for f in MEMBERS.values()])
    meta = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit(meta_X, b["label"])
    p["결합"] = meta.predict_proba(np.column_stack(list(p.values())))[:, 1]
    return np.mean(list(p.values()), axis=0), p


def current(tr, te):
    """현재 모델: Ridge+EBM+랜덤포레스트 회귀 평균 (zoo.py 정의 그대로), 점수 = −예상 한낮 최저."""
    return np.mean([ZOO[m](tr, te) for m in ("Ridge", "EBM(회귀)", "랜덤포레스트(회귀)")], axis=0)


def evaluate(name, fn, df, unit, team=False):
    t0 = time.time()
    tr_val, tr_fin = (TEAM["TR_VAL"], TEAM["TR_FIN"]) if team else (TR_VAL, TR_FIN)
    tr, va = df[df["date"].between(*tr_val)], df[df["date"].between(*VALID)]
    sv = fn(tr, va)
    sv = sv[0] if isinstance(sv, tuple) else sv
    yv = va["label"].values
    m = va["offday"].values if team else np.ones(len(va), bool)     # 팀 규칙: 휴일·주말만, 기존: 2025 전체 날
    warn, strong = threshold_for(yv[m], sv[m], 0.9), strong_threshold(yv[m], sv[m])
    cal = LogisticRegression(C=1e6).fit(sv[m].reshape(-1, 1), yv[m])
    trf, te = df[df["date"].between(*tr_fin)], df[df["date"].between(*TEST)]
    out = fn(trf, te)
    st, members = (out if isinstance(out, tuple) else (out, None))
    sec = time.time() - t0
    vo, to = va["offday"].values, te["offday"].values
    y5, s5, y6, s6 = yv[vo], sv[vo], te["label"].values[to], st[to]
    v, t, bs = drop_stats(y5, s5), drop_stats(y6, s6), boot(y6, s6)
    h = te["date"].values[to] < np.datetime64("2026-07-01")
    w6, g6, w5, g5 = s6 >= warn, s6 >= strong, s5 >= warn, s5 >= strong
    hit = lambda a, y: int((a & (y == 1)).sum())
    fa = lambda a, y: int((a & (y == 0)).sum())
    row = {"모델": name,
           "PR-AUC": t["PR-AUC"], "PR-AUC 95% 구간": f"{bs['PR-AUC'].quantile(.025):.2f} ~ {bs['PR-AUC'].quantile(.975):.2f}",
           "ROC-AUC": t["ROC-AUC"], "Brier (낮을수록 좋음)": brier_score_loss(y6, cal.predict_proba(s6.reshape(-1, 1))[:, 1]),
           "헛경보@재현율80%": t["헛경보@재현율80%"],
           "헛경보@재현율80% 95% 구간": f"{bs['헛경보@재현율80%'].quantile(.025):.0f} ~ {bs['헛경보@재현율80%'].quantile(.975):.0f}",
           "헛경보@재현율90%": t["헛경보@재현율90%"],
           "주의 이상 적중/18": hit(w6, y6), "주의 이상 헛경보": fa(w6, y6), "주의 이상 정밀도": hit(w6, y6) / max(w6.sum(), 1),
           "경보 적중/18": hit(g6, y6), "경보 헛경보": fa(g6, y6), "경보 정밀도": hit(g6, y6) / max(g6.sum(), 1),
           "PR-AUC 2026 1~6월": drop_stats(y6[h], s6[h])["PR-AUC"], "PR-AUC 2026 7월~": drop_stats(y6[~h], s6[~h])["PR-AUC"],
           "2025 검증 PR-AUC": v["PR-AUC"], "2025 검증 헛경보@재현율80%": v["헛경보@재현율80%"],
           "검증→시험 PR-AUC 변화": t["PR-AUC"] - v["PR-AUC"],
           "2025 검증 주의 이상 (적중·헛경보)": f"{hit(w5, y5)} · {fa(w5, y5)}",
           "문턱 (주의 · 경보)": f"{unit(warn)} · {unit(strong)}",
           "검증+시험 학습·예측 시간(초)": round(sec)}
    daily = te[["date", "label", "offday"]].assign(model=name, score=st)
    if members:
        for k, p in members.items():
            row[f"구성원 PR-AUC: {k}"] = drop_stats(y6, p[to])["PR-AUC"]
    return row, daily


def main(team=False):
    df = data(team)
    rows, days = [], []
    for name, fn, unit in (("팀원식 8개 결합 (직접 재현)", ens8, lambda s: round(s, 3)),
                           ("현재 모델 (Ridge+EBM+RF, 예상 한낮 최저)", current, lambda s: f"{-s:.1f}원")):
        r, d = evaluate(name, fn, df, unit, team)
        rows.append(r)
        days.append(d)
        print(f"{name}: {r['검증+시험 학습·예측 시간(초)']}초", flush=True)
    res = pd.DataFrame(rows).set_index("모델").T
    tag = "_team" if team else ""
    res.to_csv(OUT_TAB / f"20_ens8_vs_current_육지{tag}.csv", encoding="utf-8-sig")
    pd.concat(days).round(4).to_csv(OUT_TAB / f"20_ens8_daily_육지{tag}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.map(lambda x: round(x, 3) if isinstance(x, float) else x).to_string())


if __name__ == "__main__":
    main("--team" in sys.argv)
