"""모델 대규모 비교 (육지, 판단 시각 D-1 15:00) — 재현율 우선 기준.

후보 (피처는 현 최종 피처 = 기본+주간전망+보강+태양광설비로 통일, 시계열 모델은 시간별 SMP 168시간 추가)
  선형     : Ridge, Lasso, ElasticNet, Huber, 로지스틱 회귀
  트리     : 의사결정나무(회귀·분류), 랜덤포레스트, 엑스트라트리, GradientBoosting, HistGB,
             XGBoost(회귀·분류), LightGBM(회귀·분류), CatBoost(회귀·분류)
  GAM      : EBM(회귀·분류)
  기타     : SVR, KNN, MLP, TabPFN(설치·인증 될 때만)
  시계열   : 기준선(전날 한낮 최저 그대로), Chronos-Bolt·Chronos-2(제로샷), LSTM, GRU, 1D-CNN, Transformer
  앙상블   : 3개 이상 평균·순위평균 (구성은 아래 ENSEMBLES에 미리 고정)
검증 (2025년 이전 자료만): F1 2024 학습 → 2025-01~06 예측, F2 2024-01~2025-06 학습 → 2025-07~12 예측, 이어 붙여 채점
채점 (재현율 우선, 2026-10-21 사전 고정): 1순위 급락 90%를 잡을 때 헛경보, 2순위 100%일 때 헛경보, 3순위 PR-AUC
채택: 현 모델(Ridge+EBM)보다 1순위가 15% 이상 작을 때만 교체. 2026은 결정 후 1회 채점
실행: python -m drop_classifier.src.zoo
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from common.config import PROCESSED_DIR, SEED
from common.loaders import load_smp_excel
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.explore2 import data2
from drop_classifier.src.fuel_level import BASE
from drop_classifier.src.supply_level import SOLAR
from drop_classifier.src.two_tier import OUT_TAB

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
COLS = BASE + SOLAR
FOLDS = [(("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-06-30")),
         (("2024-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"))]
CLIP = (0, 150)
SEQ = 168


def imp(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def reg(make):
    def f(tr, te):
        return -make().fit(tr[COLS], tr["min_smp_11_15"].clip(*CLIP)).predict(te[COLS])
    return f


CHR = ["bolt_mid_med", "bolt_mid_q10", "c2_mid_med", "c2_mid_q10"]


def reg_chr(make):
    """Chronos 예측값을 피처로 추가한 회귀."""
    def f(tr, te):
        cols = COLS + CHR
        return -make().fit(tr[cols], tr["min_smp_11_15"].clip(*CLIP)).predict(te[cols])
    return f


def cls(make):
    def f(tr, te):
        return make().fit(tr[COLS], tr["label"]).predict_proba(te[COLS])[:, 1]
    return f


def col_score(col):
    """학습 없이 열 값을 그대로 점수로 (낮을수록 위험)."""
    def f(tr, te):
        return -te[col].values
    return f


# ---------- 시계열 신경망 (시간별 SMP 168시간 + 표 피처) ----------
class SeqNet(torch.nn.Module):
    def __init__(self, kind, n_tab, h=32):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.rnn = (torch.nn.LSTM if kind == "lstm" else torch.nn.GRU)(1, h, batch_first=True)
        elif kind == "cnn":
            self.rnn = torch.nn.Sequential(torch.nn.Conv1d(1, h, 24, stride=6), torch.nn.ReLU(),
                                           torch.nn.Conv1d(h, h, 5, stride=2), torch.nn.ReLU(),
                                           torch.nn.AdaptiveAvgPool1d(1))
        else:  # transformer: 시간 단위를 하루(24시간) 토큰으로 묶음
            self.proj = torch.nn.Linear(24, h)
            self.rnn = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(h, 4, 64, 0.1, batch_first=True), 2)
        self.head = torch.nn.Sequential(torch.nn.Linear(h + n_tab, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                        torch.nn.Linear(32, 1))

    def forward(self, seq, tab):
        if self.kind in ("lstm", "gru"):
            z = self.rnn(seq.unsqueeze(-1))[0][:, -1]
        elif self.kind == "cnn":
            z = self.rnn(seq.unsqueeze(1)).squeeze(-1)
        else:
            z = self.rnn(self.proj(seq.view(seq.shape[0], -1, 24))).mean(1)
        return self.head(torch.cat([z, tab], 1)).squeeze(-1)


def seq_model(kind, seeds=(0, 1, 2)):
    def f(tr, te):
        sx = SimpleImputer(strategy="median").fit(tr[COLS])
        sc = StandardScaler().fit(sx.transform(tr[COLS]))
        tab = lambda d: torch.tensor(sc.transform(sx.transform(d[COLS])), dtype=torch.float32)
        mu, sd = np.nanmean(np.stack(tr["seq"])), np.nanstd(np.stack(tr["seq"]))
        seq = lambda d: torch.tensor((np.stack(d["seq"]) - mu) / sd, dtype=torch.float32)
        y = torch.tensor(tr["min_smp_11_15"].clip(*CLIP).values / 100, dtype=torch.float32)
        Xs, Xt, Ts, Tt = seq(tr), tab(tr), seq(te), tab(te)
        n_val = max(30, len(tr) // 7)            # 학습 구간 마지막 부분으로 조기 종료 (시간 순서 유지)
        preds = []
        for s in seeds:
            torch.manual_seed(SEED + s)
            net = SeqNet(kind, Xt.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
            best, best_state, bad = 1e9, None, 0
            for ep in range(200):
                net.train()
                perm = torch.randperm(len(y) - n_val)
                for b in range(0, len(perm), 32):
                    i = perm[b:b + 32]
                    opt.zero_grad()
                    loss = torch.nn.functional.mse_loss(net(Xs[i], Xt[i]), y[i])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = torch.nn.functional.mse_loss(net(Xs[-n_val:], Xt[-n_val:]), y[-n_val:]).item()
                if v < best - 1e-5:
                    best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 20:
                        break
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                preds.append(net(Ts, Tt).numpy() * 100)
        return -np.mean(preds, axis=0)
    return f


def tabpfn_reg(tr, te):
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device="cpu", random_state=SEED)
    X = SimpleImputer(strategy="median").fit(tr[COLS])
    return -m.fit(X.transform(tr[COLS]), tr["min_smp_11_15"].clip(*CLIP)).predict(X.transform(te[COLS]))


BASES = {
    # 선형
    "Ridge": reg(lambda: imp(Ridge(alpha=3))),
    "Lasso": reg(lambda: imp(Lasso(alpha=0.5, max_iter=20000))),
    "ElasticNet": reg(lambda: imp(ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000))),
    "Huber": reg(lambda: imp(HuberRegressor(alpha=1e-3, max_iter=2000))),
    "로지스틱 회귀": cls(lambda: imp(LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))),
    # 트리
    "의사결정나무(회귀)": reg(lambda: imp(DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=SEED))),
    "의사결정나무(분류)": cls(lambda: imp(DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced",
                                                             random_state=SEED))),
    "랜덤포레스트(회귀)": reg(lambda: imp(RandomForestRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=4))),
    "랜덤포레스트(분류)": cls(lambda: imp(RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample",
                                                            random_state=SEED, n_jobs=4))),
    "엑스트라트리": reg(lambda: imp(ExtraTreesRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=4))),
    "GradientBoosting": reg(lambda: imp(GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                                                  subsample=0.8, random_state=SEED))),
    "HistGB": reg(lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,
                                                        min_samples_leaf=10, random_state=SEED)),
    "XGBoost(회귀)": reg(lambda: XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                             colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=4)),
    "XGBoost(분류)": cls(lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                              colsample_bytree=0.8, scale_pos_weight=15, random_state=SEED, n_jobs=4)),
    "LightGBM(회귀)": reg(lambda: LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                                               random_state=SEED, verbose=-1)),
    "LightGBM(분류)": cls(lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                                class_weight="balanced", random_state=SEED, verbose=-1)),
    "CatBoost(회귀)": reg(lambda: CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)),
    "CatBoost(분류)": cls(lambda: CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05,
                                                    auto_class_weights="Balanced", random_seed=SEED, verbose=0)),
    # GAM
    "EBM(회귀)": reg(lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5,
                                                         max_bins=64, n_jobs=1)),
    "EBM(분류)": cls(lambda: ExplainableBoostingClassifier(random_state=SEED, interactions=5, min_samples_leaf=5,
                                                          max_bins=64, n_jobs=1)),
    # 기타
    "SVR": reg(lambda: imp(SVR(C=30, epsilon=3))),
    "KNN": reg(lambda: imp(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
    "MLP": reg(lambda: imp(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=3000, early_stopping=True, random_state=SEED))),
    "TabPFN": tabpfn_reg,
    # Chronos 예측을 피처로 넣은 모델 (2026-10-21 추가)
    "Ridge+Chronos피처": reg_chr(lambda: imp(Ridge(alpha=3))),
    "EBM+Chronos피처": reg_chr(lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5,
                                                                   max_bins=64, n_jobs=1)),
    "랜덤포레스트+Chronos피처": reg_chr(lambda: imp(RandomForestRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=4))),
    # 시계열
    "기준선(전날 한낮 최저)": col_score("prev_day_min_smp"),
    "Chronos-Bolt(제로샷)": col_score("bolt_mid_q10"),
    "Chronos-2(제로샷)": col_score("c2_mid_q10"),
    "LSTM": seq_model("lstm"),
    "GRU": seq_model("gru"),
    "1D-CNN": seq_model("cnn"),
    "Transformer": seq_model("tf"),
}

# 앙상블 (사전 고정). 'mean' = 예측 원(元) 평균(회귀끼리), 'rank' = 순위 평균(회귀·분류 섞을 때)
ENSEMBLES = {
    "Ridge+EBM (현 모델)": ("mean", ["Ridge", "EBM(회귀)"]),
    "Ridge+EBM+CatBoost": ("mean", ["Ridge", "EBM(회귀)", "CatBoost(회귀)"]),
    "Ridge+EBM+XGBoost": ("mean", ["Ridge", "EBM(회귀)", "XGBoost(회귀)"]),
    "Ridge+EBM+LightGBM": ("mean", ["Ridge", "EBM(회귀)", "LightGBM(회귀)"]),
    "부스팅 3종 (XGB+LGBM+CatBoost)": ("mean", ["XGBoost(회귀)", "LightGBM(회귀)", "CatBoost(회귀)"]),
    "Ridge+EBM+부스팅 3종": ("mean", ["Ridge", "EBM(회귀)", "XGBoost(회귀)", "LightGBM(회귀)", "CatBoost(회귀)"]),
    "Ridge+EBM+랜덤포레스트": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)"]),
    "Ridge+EBM+LSTM": ("mean", ["Ridge", "EBM(회귀)", "LSTM"]),
    "Ridge+EBM+Chronos-Bolt": ("mean", ["Ridge", "EBM(회귀)", "Chronos-Bolt(제로샷)"]),
    "신경망 4종 (LSTM+GRU+CNN+Transformer)": ("mean", ["LSTM", "GRU", "1D-CNN", "Transformer"]),
    "회귀 전체 평균": ("mean", None),
    "순위평균(Ridge+EBM, 로지스틱, CatBoost분류)": ("rank", ["Ridge", "EBM(회귀)", "로지스틱 회귀", "CatBoost(분류)"]),
    "순위평균(Ridge+EBM, EBM분류, LightGBM분류)": ("rank", ["Ridge", "EBM(회귀)", "EBM(분류)", "LightGBM(분류)"]),
    "순위평균(Ridge+EBM, LSTM, Chronos-Bolt)": ("rank", ["Ridge", "EBM(회귀)", "LSTM", "Chronos-Bolt(제로샷)"]),
    # 2026-10-21 추가 (1차 결과를 본 뒤 추가한 후보 — 보고 시 구분 표기)
    "Ridge+EBM+랜덤포레스트+Chronos-Bolt": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "Chronos-Bolt(제로샷)"]),
    "Ridge+EBM+랜덤포레스트+Chronos-2": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "Chronos-2(제로샷)"]),
    "Ridge+EBM+Chronos-Bolt+Chronos-2": ("mean", ["Ridge", "EBM(회귀)", "Chronos-Bolt(제로샷)", "Chronos-2(제로샷)"]),
    "Ridge+EBM+랜덤포레스트+TabPFN": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "TabPFN"]),
    "Ridge+EBM+랜덤포레스트+XGBoost+LSTM": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "XGBoost(회귀)", "LSTM"]),
    "다 넣기 (Ridge+EBM+RF+XGB+LSTM+GRU+TabPFN+Chronos 2종)": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "XGBoost(회귀)",
                                                         "LSTM", "GRU", "TabPFN", "Chronos-Bolt(제로샷)", "Chronos-2(제로샷)"]),
    "Chronos피처 Ridge+EBM": ("mean", ["Ridge+Chronos피처", "EBM+Chronos피처"]),
    "Chronos피처 Ridge+EBM+랜덤포레스트": ("mean", ["Ridge+Chronos피처", "EBM+Chronos피처", "랜덤포레스트+Chronos피처"]),
    "순위평균 다 넣기 (회귀 상위+분류+Chronos)": ("rank", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "XGBoost(회귀)", "LSTM",
                                               "로지스틱 회귀", "랜덤포레스트(분류)", "Chronos-Bolt(제로샷)", "Chronos-2(제로샷)"]),
}
ADDED_LATER = {"Ridge+Chronos피처", "EBM+Chronos피처", "랜덤포레스트+Chronos피처"} | {
    k for k in ENSEMBLES if k.startswith(("Ridge+EBM+랜덤포레스트+", "Ridge+EBM+Chronos-Bolt+", "다 넣기", "Chronos피처", "순위평균 다 넣기"))}
CLS_NAMES = {k for k in BASES if "분류" in k or k == "로지스틱 회귀"}


def add_seq_and_chronos(df):
    s = load_smp_excel("육지").sort_values(["date", "hour"])
    vals = s["smp"].values.astype("float32")
    pos = {pd.Timestamp(d): i * 24 for i, d in enumerate(s["date"].values[::24])}
    df["seq"] = [vals[pos[d] - SEQ:pos[d]] if d in pos and pos[d] >= SEQ else np.full(SEQ, np.nan, "float32")
                 for d in df["date"]]
    for tag in ("bolt", "c2"):
        p = PROCESSED_DIR / "drop_classifier" / f"chronos_{tag}.csv"
        if p.exists():
            df = df.merge(pd.read_csv(p, parse_dates=["date"]), on="date", how="left")
        else:
            df[f"{tag}_mid_q10"] = np.nan
    return df


def score(y, s):
    return {"헛경보@80%": false_alarms_at(y, s, 0.8)[0], "헛경보@90%": false_alarms_at(y, s, 0.9)[0],
            "헛경보@100%": false_alarms_at(y, s, 1.0)[0], "PR-AUC": average_precision_score(y, s),
            "ROC-AUC": roc_auc_score(y, s)}


def main():
    df = add_seq_and_chronos(data2())
    preds, rows = {}, []
    for name, fn in BASES.items():
        t0 = time.time()
        try:
            parts = []
            for (a, b), (c, d) in FOLDS:
                tr = df[df["date"].between(a, b)]
                te = df[df["date"].between(c, d)].copy()
                te["s"] = fn(tr, te)
                te["s_rank"] = te["s"].rank(pct=True)
                parts.append(te[["date", "label", "s", "s_rank"]])
            p = pd.concat(parts)
            if p["s"].isna().all():
                raise ValueError("예측 없음")
            p["s"] = p["s"].fillna(p["s"].median())
            preds[name] = p
            r = score(p["label"].values, p["s"].values)
            rows.append({"구분": "단일" + ("(추가)" if name in ADDED_LATER else ""), "모델": name, **r, "초": round(time.time() - t0)})
            print(f"{name}: 헛경보@90% {r['헛경보@90%']}, @100% {r['헛경보@100%']}, PR-AUC {r['PR-AUC']:.3f} "
                  f"({time.time()-t0:.0f}초)", flush=True)
        except Exception as e:
            print(f"{name}: 실패 — {repr(e)[:150]}", flush=True)

    regs = [k for k in preds if k not in CLS_NAMES]
    for name, (how, members) in ENSEMBLES.items():
        members = regs if members is None else members
        if any(m not in preds for m in members):
            print(f"{name}: 구성 모델 누락으로 건너뜀", flush=True)
            continue
        key = "s" if how == "mean" else "s_rank"
        base = preds[members[0]][["date", "label"]].copy()
        base["s"] = np.mean([preds[m][key].values for m in members], axis=0)
        r = score(base["label"].values, base["s"].values)
        rows.append({"구분": "앙상블" + ("(추가)" if name in ADDED_LATER else ""), "모델": name, **r, "초": 0})
        print(f"[앙상블] {name}: 헛경보@90% {r['헛경보@90%']}, @100% {r['헛경보@100%']}, PR-AUC {r['PR-AUC']:.3f}", flush=True)

    res = pd.DataFrame(rows).sort_values(["헛경보@90%", "헛경보@100%", "PR-AUC"], ascending=[True, True, False])
    res.round(3).to_csv(OUT_TAB / "17_zoo_cv2025_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n", res.round(3).to_string(index=False))
    cur = res.loc[res["모델"] == "Ridge+EBM (현 모델)", "헛경보@90%"].iloc[0]
    best = res.iloc[0]
    print(f"\n현 모델 헛경보@90% {cur} / 1위 {best['모델']} {best['헛경보@90%']} → 교체 기준(15%↓) "
          f"{'충족' if best['헛경보@90%'] <= cur * 0.85 else '미달'}")


if __name__ == "__main__":
    main()
