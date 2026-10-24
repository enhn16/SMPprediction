"""제주 다음 날 급락 판별 — 모델 대규모 비교 (육지 zoo.py와 같은 절차, 판단 시각 D-1 15:00).

급락 = 내일 제주 11~15시 최저 SMP < 30원. 표 = jeju_drop.day_table() (하루 한 행, 학습 2024-01 ~).
피처 묶음
  S4 +제주예측 : 1차 선택 묶음 (달력·제주 SMP·수요예측·구역예보·Open-Meteo·육지 정보·육지/제주 예측 한낮 최저)
  S5 +계통실적 : S4 + 제주 5분 계통 실적(판단일 14:25까지 풍력·태양광·수요·순부하·신재생 비중, 전날·최근 7일)
  S5c 핵심     : S5에서 급락 원리(낮은 수요·많은 태양광·강한 바람·최근 급락·예측 한낮 최저)로 고른 27개 (결과를 보기 전 고정)
후보 (묶음마다)
  선형   : Ridge, Lasso, ElasticNet, Huber, 로지스틱(L2), 로지스틱(L1), LDA
  트리   : 의사결정나무, 랜덤포레스트, 엑스트라트리, GradientBoosting, HistGB, XGBoost, LightGBM, CatBoost (회귀·분류 각각)
  GAM    : EBM(회귀·분류)
  기타   : SVR, SVC, KNN(회귀·분류), MLP(회귀·분류), 나이브 베이즈, TabPFN(회귀·분류)
  시계열 : LSTM, GRU, 1D-CNN, Transformer (제주 시간별 SMP 168시간 + 표 피처, 한낮 최저 회귀, 시드 3개 평균)
  규칙   : 학습 없는 열 그대로 (전날 제주 한낮 최저, 육지 일주일 예측, 제주 Chronos-2 중앙·하위 10%)
  앙상블 : 아래 ENSEMBLES (구성 미리 고정). 회귀끼리는 점수 평균, 회귀·분류가 섞이면 순위 평균
검증 (2025년 이전 자료만): F1 2024 학습 → 2025-01~06, F2 2024-01~2025-06 학습 → 2025-07~12, 이어 붙여 채점
순위 (사전 고정, 육지와 같음): 1순위 급락 90%를 잡을 때 헛경보 → 2순위 100%일 때 헛경보 → 3순위 PR-AUC
안정성 (2026-10-25 추가, 사용자 확인): F1·F2 상위 20개를 2024 학습 → 2025 전체로도 채점, 두 검증 중 나쁜 쪽 헛경보@90%로 최종 순위
보조: 상위 후보 매월 재학습 비교, 1차 선택(S4 랜덤포레스트 분류) 대비 헛경보 차이 부트스트랩, F1로만 고른 상위 k 앙상블의 F2 성적
출력: drop_classifier/outputs/tables/J4_zoo_2025.csv, J5_zoo_monthly_2025.csv, J6_zoo_bootstrap.csv, J7_zoo_nested_F2.csv, J8_zoo_stability_2025.csv,
      J_zoo_pick.json (jeju_drop.py가 읽어 문턱·확률·결과 파일을 만든다)
실행: python -m drop_classifier.src.jeju_zoo [--only S5]   (2026은 쓰지 않음)
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import PROCESSED_DIR, SEED
from common.loaders import load_smp_actual
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src import jeju_drop as jd

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
TAB = jd.TAB
CACHE = PROCESSED_DIR / "drop_classifier" / "jeju_zoo_preds"   # 묶음별 표본 밖 점수 (재실행 시 재사용)
CLIP = (0, 150)
SEQ = 168
NJ = 6

SUKUB = ["js_wind_i_am", "js_solar_i_mid", "js_dem_i_mid", "js_net_i_mid", "js_share_i_mid", "js_wind_i1_day",
         "js_solar_i1_mid", "js_net_i1_mid", "js_share_i1_mid", "js_solar_max7_mid", "js_wind7", "js_net_min7_mid"]
S4 = jd.SETS["S4 +제주예측"]
CORE = ["offday", "daytype", "dow", "month_sin", "month_cos", "off_block_len", "s_d1_midmin", "s_same_midmin", "j_min7_mid",
        "j_drop14", "j_drop28", "j_only28", "d_d1_mid", "d_rel28", "jf_sky_pm", "jo_rad_mid", "jo_cloud_mid", "jo_ws100_day",
        "jo_ws100_mid", "L_pred_midmin", "J_c2sx_midmin", "J_c2sx_qlo_midmin", "js_wind_i_am", "js_solar_i_mid",
        "js_net_i_mid", "js_share_i_mid", "js_net_min7_mid"]
SETS = {"S4 +제주예측": S4, "S5 +계통실적": S4 + SUKUB, "S5c 핵심": CORE}


# ---------- 공통 ----------
def imp(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def reg(make):
    def f(tr, te, c):
        return -make().fit(tr[c], tr["min11_15"].clip(*CLIP)).predict(te[c])
    f.kind = "reg"
    return f


def cls(make):
    def f(tr, te, c):
        return make().fit(tr[c], tr["label"].astype(int)).predict_proba(te[c])[:, 1]
    f.kind = "cls"
    return f


def rule(col):
    def f(tr, te, c):
        v = te[col].astype(float)
        return -v.fillna(v.median()).values
    f.kind = "rule"
    return f


def _pos_w(tr):
    y = tr["label"].astype(int)
    return float((y == 0).sum() / max(1, (y == 1).sum()))


def tabpfn(kind):
    def f(tr, te, c):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        x = SimpleImputer(strategy="median").fit(tr[c])
        if kind == "reg":
            m = TabPFNRegressor(device="cpu", random_state=SEED)
            return -m.fit(x.transform(tr[c]), tr["min11_15"].clip(*CLIP)).predict(x.transform(te[c]))
        m = TabPFNClassifier(device="cpu", random_state=SEED)
        return m.fit(x.transform(tr[c]), tr["label"].astype(int)).predict_proba(x.transform(te[c]))[:, 1]
    f.kind = kind
    return f


# ---------- 시계열 신경망 (제주 시간별 SMP 168시간 = D-7 ~ D-1, + 표 피처) ----------
_SEQ = {}


def seq_for(dates):
    """대상일 D → D-7 1시 ~ D-1 24시 제주 SMP 168개 (D-1 SMP는 D-2 저녁 하루전 시장에서 확정되어 판단 시각에 알 수 있음)."""
    if not _SEQ:
        s = load_smp_actual("제주").sort_values(["date", "hour"])
        _SEQ["v"] = s["smp"].values.astype("float32")
        _SEQ["pos"] = {pd.Timestamp(d): i * 24 for i, d in enumerate(s["date"].values[::24])}
        _SEQ["next"] = pd.Timestamp(s["date"].max()) + pd.Timedelta(days=1)   # 내일(실적 없음) = 마지막 168시간
    v, pos = _SEQ["v"], _SEQ["pos"]
    out = []
    for d in pd.to_datetime(dates):
        p = len(v) if d == _SEQ["next"] else pos.get(d)
        out.append(v[p - SEQ:p] if p is not None and p >= SEQ else np.full(SEQ, np.nan, "float32"))
    return np.stack(out)


class SeqNet(torch.nn.Module):
    def __init__(self, kind, n_tab, h=32):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.rnn = (torch.nn.LSTM if kind == "lstm" else torch.nn.GRU)(1, h, batch_first=True)
        elif kind == "cnn":
            self.rnn = torch.nn.Sequential(torch.nn.Conv1d(1, h, 24, stride=6), torch.nn.ReLU(),
                                           torch.nn.Conv1d(h, h, 5, stride=2), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool1d(1))
        else:   # transformer: 하루(24시간) 단위 토큰
            self.proj = torch.nn.Linear(24, h)
            self.rnn = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(h, 4, 64, 0.1, batch_first=True), 2)
        self.head = torch.nn.Sequential(torch.nn.Linear(h + n_tab, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(32, 1))

    def forward(self, seq, tab):
        if self.kind in ("lstm", "gru"):
            z = self.rnn(seq.unsqueeze(-1))[0][:, -1]
        elif self.kind == "cnn":
            z = self.rnn(seq.unsqueeze(1)).squeeze(-1)
        else:
            z = self.rnn(self.proj(seq.view(seq.shape[0], -1, 24))).mean(1)
        return self.head(torch.cat([z, tab], 1)).squeeze(-1)


def seq_model(kind, seeds=(0, 1, 2)):
    def f(tr, te, c):
        sx = SimpleImputer(strategy="median").fit(tr[c])
        sc = StandardScaler().fit(sx.transform(tr[c]))
        tab = lambda d: torch.tensor(sc.transform(sx.transform(d[c])), dtype=torch.float32)
        s_tr, s_te = seq_for(tr["date"]), seq_for(te["date"])
        mu, sd = np.nanmean(s_tr), np.nanstd(s_tr)
        seq = lambda a: torch.tensor(np.nan_to_num((a - mu) / sd), dtype=torch.float32)
        y = torch.tensor(tr["min11_15"].clip(*CLIP).values / 100, dtype=torch.float32)
        Xs, Xt, Ts, Tt = seq(s_tr), tab(tr), seq(s_te), tab(te)
        n_val = max(30, len(tr) // 7)            # 학습 구간 마지막 부분으로 조기 종료 (시간 순서 유지)
        preds = []
        for s in seeds:
            torch.manual_seed(SEED + s)
            net = SeqNet(kind, Xt.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
            best, best_state, bad = 1e9, None, 0
            for _ in range(200):
                net.train()
                perm = torch.randperm(len(y) - n_val)
                for b in range(0, len(perm), 32):
                    i = perm[b:b + 32]
                    opt.zero_grad()
                    torch.nn.functional.mse_loss(net(Xs[i], Xt[i]), y[i]).backward()
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
    f.kind = "reg"
    return f


def _bases():
    from catboost import CatBoostClassifier, CatBoostRegressor
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                                  HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LogisticRegression, Ridge
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from xgboost import XGBClassifier, XGBRegressor

    def xgbc(tr, te, c):
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=_pos_w(tr), random_state=SEED, n_jobs=NJ, verbosity=0)
        return m.fit(tr[c], tr["label"].astype(int)).predict_proba(te[c])[:, 1]
    xgbc.kind = "cls"
    return {
        # 선형
        "Ridge": reg(lambda: imp(Ridge(alpha=3))),
        "Lasso": reg(lambda: imp(Lasso(alpha=0.5, max_iter=20000))),
        "ElasticNet": reg(lambda: imp(ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000))),
        "Huber": reg(lambda: imp(HuberRegressor(alpha=1e-3, max_iter=2000))),
        "로지스틱(L2)": cls(lambda: imp(LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))),
        "로지스틱(L1)": cls(lambda: imp(LogisticRegression(C=0.1, penalty="l1", solver="liblinear", class_weight="balanced"))),
        "LDA": cls(lambda: imp(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
        # 트리
        "의사결정나무(회귀)": reg(lambda: imp(DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=SEED))),
        "의사결정나무(분류)": cls(lambda: imp(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced",
                                                                 random_state=SEED))),
        "랜덤포레스트(회귀)": reg(lambda: imp(RandomForestRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=NJ))),
        "랜덤포레스트(분류)": cls(lambda: imp(RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample",
                                                                 random_state=SEED, n_jobs=NJ))),
        "엑스트라트리(회귀)": reg(lambda: imp(ExtraTreesRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=NJ))),
        "엑스트라트리(분류)": cls(lambda: imp(ExtraTreesClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample",
                                                                 random_state=SEED, n_jobs=NJ))),
        "GradientBoosting(회귀)": reg(lambda: imp(GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                                                            subsample=0.8, random_state=SEED))),
        "GradientBoosting(분류)": cls(lambda: imp(GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                                                             subsample=0.8, random_state=SEED))),
        "HistGB(회귀)": reg(lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=SEED)),
        "HistGB(분류)": cls(lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                                   class_weight="balanced", random_state=SEED)),
        "XGBoost(회귀)": reg(lambda: XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=SEED, n_jobs=NJ, verbosity=0)),
        "XGBoost(분류)": xgbc,
        "LightGBM(회귀)": reg(lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                                    verbose=-1, n_jobs=NJ)),
        "LightGBM(분류)": cls(lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, is_unbalance=True,
                                                     random_state=SEED, verbose=-1, n_jobs=NJ)),
        "CatBoost(회귀)": reg(lambda: CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0,
                                                        thread_count=NJ)),
        "CatBoost(분류)": cls(lambda: CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
                                                         random_seed=SEED, verbose=0, thread_count=NJ)),
        # GAM
        "EBM(회귀)": reg(lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64,
                                                              n_jobs=NJ)),
        "EBM(분류)": cls(lambda: ExplainableBoostingClassifier(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64,
                                                               n_jobs=NJ)),
        # 기타
        "SVR": reg(lambda: imp(SVR(C=30, epsilon=3))),
        "SVC": cls(lambda: imp(SVC(C=1.0, probability=True, class_weight="balanced", random_state=SEED))),
        "KNN(회귀)": reg(lambda: imp(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
        "KNN(분류)": cls(lambda: imp(KNeighborsClassifier(n_neighbors=25, weights="distance"))),
        "MLP(회귀)": reg(lambda: imp(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=3000, early_stopping=True,
                                                  random_state=SEED))),
        "MLP(분류)": cls(lambda: imp(MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-1, max_iter=3000, random_state=SEED))),
        "나이브 베이즈": cls(lambda: imp(GaussianNB())),
        "TabPFN(회귀)": tabpfn("reg"),
        "TabPFN(분류)": tabpfn("cls"),
        # 시계열
        "LSTM": seq_model("lstm"),
        "GRU": seq_model("gru"),
        "1D-CNN": seq_model("cnn"),
        "Transformer": seq_model("tf"),
        # 규칙 (학습 없음)
        "규칙: 전날 제주 한낮 최저": rule("s_d1_midmin"),
        "규칙: 육지 일주일 예측 한낮 최저": rule("L_pred_midmin"),
        "규칙: 제주 Chronos-2 한낮 최저": rule("J_c2sx_midmin"),
        "규칙: 제주 Chronos-2 하위 10% 한낮 최저": rule("J_c2sx_qlo_midmin"),
    }


BASES = _bases()
TREE_C = ["랜덤포레스트(분류)", "엑스트라트리(분류)", "GradientBoosting(분류)", "HistGB(분류)", "XGBoost(분류)", "LightGBM(분류)",
          "CatBoost(분류)"]
TREE_R = ["랜덤포레스트(회귀)", "엑스트라트리(회귀)", "GradientBoosting(회귀)", "HistGB(회귀)", "XGBoost(회귀)", "LightGBM(회귀)",
          "CatBoost(회귀)"]
NN = ["LSTM", "GRU", "1D-CNN", "Transformer"]
# 구성 미리 고정 (2026-10-25, 결과 보기 전). how: mean = 점수 평균(회귀끼리), rank = 순위 평균
ENSEMBLES = {
    "Ridge+EBM+RF (육지 최종 구성)": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)"]),
    "회귀 부스팅 3종 (XGB+LGBM+CatBoost)": ("mean", ["XGBoost(회귀)", "LightGBM(회귀)", "CatBoost(회귀)"]),
    "회귀 트리 7종": ("mean", TREE_R),
    "신경망 4종 (LSTM+GRU+CNN+Transformer)": ("mean", NN),
    "Ridge+EBM+RF+LSTM": ("mean", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "LSTM"]),
    "분류 부스팅 3종 (XGB+LGBM+CatBoost)": ("rank", ["XGBoost(분류)", "LightGBM(분류)", "CatBoost(분류)"]),
    "분류 트리 7종": ("rank", TREE_C),
    "RF+ET (분류)": ("rank", ["랜덤포레스트(분류)", "엑스트라트리(분류)"]),
    "RF+EBM+로지스틱 (분류)": ("rank", ["랜덤포레스트(분류)", "EBM(분류)", "로지스틱(L2)"]),
    "RF+CatBoost+EBM (분류)": ("rank", ["랜덤포레스트(분류)", "CatBoost(분류)", "EBM(분류)"]),
    "RF+ET+CatBoost+EBM (분류)": ("rank", ["랜덤포레스트(분류)", "엑스트라트리(분류)", "CatBoost(분류)", "EBM(분류)"]),
    "TabPFN 회귀+분류": ("rank", ["TabPFN(회귀)", "TabPFN(분류)"]),
    "RF(분류)+TabPFN(분류)": ("rank", ["랜덤포레스트(분류)", "TabPFN(분류)"]),
    "RF(분류)+LightGBM(회귀)": ("rank", ["랜덤포레스트(분류)", "LightGBM(회귀)"]),
    "RF(분류)+RF(회귀)": ("rank", ["랜덤포레스트(분류)", "랜덤포레스트(회귀)"]),
    "RF(분류)+LSTM": ("rank", ["랜덤포레스트(분류)", "LSTM"]),
    "RF(분류)+Chronos 하위 10%": ("rank", ["랜덤포레스트(분류)", "규칙: 제주 Chronos-2 하위 10% 한낮 최저"]),
    "분류 트리 7종+회귀 트리 7종": ("rank", TREE_C + TREE_R),
    "분류 상위형 (RF·ET·Cat·EBM·TabPFN·로지스틱)": ("rank", ["랜덤포레스트(분류)", "엑스트라트리(분류)", "CatBoost(분류)", "EBM(분류)",
                                                    "TabPFN(분류)", "로지스틱(L2)"]),
    "다 넣기 (Ridge·EBM·RF·XGB·LSTM·GRU·TabPFN 회귀 + RF·Cat·EBM·TabPFN 분류)": (
        "rank", ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)", "XGBoost(회귀)", "LSTM", "GRU", "TabPFN(회귀)",
                 "랜덤포레스트(분류)", "CatBoost(분류)", "EBM(분류)", "TabPFN(분류)"]),
    "분류 전부 순위 평균": ("rank", "cls"),
    "회귀 전부 순위 평균": ("rank", "reg"),
    "학습 모델 전부 순위 평균": ("rank", "all"),
}
INCUMBENT = ("S4 +제주예측", "랜덤포레스트(분류)")


def _rank(a):
    return pd.Series(a).rank(pct=True).values


def members_of(name):
    how, ms = ENSEMBLES[name]
    if isinstance(ms, str):
        ms = [k for k, f in BASES.items() if f.kind != "rule" and (ms == "all" or f.kind == ms)]
    return how, ms


def score_fn(name, cols):
    """후보 이름 → f(tr, te) 위험 점수 (높을수록 위험). jeju_drop·jz_ops 운영이 그대로 쓴다."""
    if name in BASES:
        return lambda tr, te: BASES[name](tr, te, cols)
    how, ms = members_of(name)

    def f(tr, te):
        s = [BASES[m](tr, te, cols) for m in ms]
        return np.mean(s, axis=0) if how == "mean" else np.mean([_rank(x) for x in s], axis=0)
    return f


def is_reg(name):
    """점수가 '−예상 한낮 최저(원)'인가 (회귀 단일·회귀 점수 평균 앙상블)."""
    if name in BASES:
        return BASES[name].kind == "reg"
    how, ms = members_of(name) if name in ENSEMBLES else ("rank", [])
    return how == "mean" and all(BASES[m].kind == "reg" for m in ms)


def ranking(y, s):
    fa90, hit90 = false_alarms_at(y, s, 0.9)
    return {"헛경보@80%": false_alarms_at(y, s, 0.8)[0], "헛경보@90%": fa90, "적중@90%": hit90,
            "헛경보@100%": false_alarms_at(y, s, 1.0)[0], "PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s)}


ORDER = (["헛경보@90%", "헛경보@100%", "PR-AUC"], [True, True, False])


def run_set(d, sname, cols, scheme=jd.folds):
    """묶음 하나: 단일 후보별 F1·F2 표본 밖 점수 → {이름: DataFrame[date, label, fold, s, s_rank]}."""
    preds = {}
    for name, fn in BASES.items():
        t0 = time.time()
        try:
            parts = []
            for i, (tr, te) in enumerate(scheme(d)):
                te = te[["date", "label"]].copy()
                te["s"] = fn(tr, d.loc[te.index], cols)
                te["fold"] = i
                te["s_rank"] = te["s"].rank(pct=True)
                parts.append(te)
            p = pd.concat(parts)
            if p["s"].isna().all():
                raise ValueError("예측 없음")
            p["s"] = p["s"].fillna(p["s"].median())
            preds[name] = p
            r = ranking(p["label"].values.astype(int), p["s"].values)
            print(f"  [{sname}] {name}: 헛경보@90% {r['헛경보@90%']} · @100% {r['헛경보@100%']} · PR-AUC {r['PR-AUC']:.3f}"
                  f" ({time.time() - t0:.0f}초)", flush=True)
        except Exception as e:
            print(f"  [{sname}] {name}: 실패 — {repr(e)[:160]}", flush=True)
    for name in ENSEMBLES:
        how, ms = members_of(name)
        if any(m not in preds for m in ms):
            print(f"  [{sname}] {name}: 구성 모델 누락으로 건너뜀", flush=True)
            continue
        p = preds[ms[0]][["date", "label", "fold"]].copy()
        p["s"] = np.mean([preds[m]["s" if how == "mean" else "s_rank"].values for m in ms], axis=0)
        p["s_rank"] = p.groupby("fold")["s"].rank(pct=True)
        preds[name] = p
    return preds


def bootstrap(preds_all, top, n=2000):
    """1차 선택 대비 헛경보@90% 차이 — 날짜 단위 복원 추출(급락일·평상일 따로) 95% 구간."""
    rng = np.random.default_rng(SEED)
    inc = preds_all[INCUMBENT]
    y = inc["label"].values.astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]
    fa_inc = np.array([false_alarms_at(y[i], inc["s"].values[i], 0.9)[0] for i in idx])
    rows = []
    for key in top:
        s = preds_all[key]["s"].values
        fa = np.array([false_alarms_at(y[i], s[i], 0.9)[0] for i in idx])
        dlt = fa - fa_inc
        rows.append({"피처": key[0], "모델": key[1], "헛경보@90% 차이 중앙(후보−1차)": float(np.median(dlt)),
                     "95% 하한": float(np.percentile(dlt, 2.5)), "95% 상한": float(np.percentile(dlt, 97.5)),
                     "1차보다 적을 확률": float((dlt < 0).mean())})
    return pd.DataFrame(rows)


def nested_topk(preds_all):
    """보조: F1(2025 상반기) 성적만으로 상위 k개 단일 모델을 골라 순위 평균 → F2(하반기) 성적. 같은 F2에서 단일 후보와 비교."""
    rows = []
    for sname in SETS:
        singles = {k[1]: v for k, v in preds_all.items() if k[0] == sname and k[1] in BASES and BASES[k[1]].kind != "rule"}
        f1 = []
        for m, p in singles.items():
            q = p[p["fold"] == 0]
            f1.append((m, *[ranking(q["label"].values.astype(int), q["s"].values)[c] for c in ORDER[0]]))
        f1 = pd.DataFrame(f1, columns=["모델", *ORDER[0]]).sort_values(ORDER[0], ascending=ORDER[1])
        for k in (3, 5, 7, 10):
            ms = f1["모델"].head(k).tolist()
            q = [singles[m][singles[m]["fold"] == 1] for m in ms]
            s = np.mean([x["s_rank"].values for x in q], axis=0)
            rows.append({"피처": sname, "후보": f"F1 상위 {k}개 순위 평균", "구성": " + ".join(ms),
                         **ranking(q[0]["label"].values.astype(int), s)})
        for m, p in singles.items():
            q = p[p["fold"] == 1]
            rows.append({"피처": sname, "후보": m, "구성": "", **ranking(q["label"].values.astype(int), q["s"].values)})
    return pd.DataFrame(rows).sort_values(ORDER[0], ascending=ORDER[1])


def stability(d, t, n=20):
    """상위 n개(규칙 제외): 2024만 학습 → 2025 전체 예측(문턱을 정하는 방식)의 헛경보@90%도 계산해, 두 검증 중 나쁜 쪽으로 순위.
    학습 자료가 적을 때 크게 나빠지는 후보를 거르기 위함 (1차 비교에서 L1 로지스틱이 67 → 129로 흔들림)."""
    lab = d[d["label"].notna()]
    tr, te = lab[lab["date"] < "2025-01-01"], lab[lab["date"].dt.year == 2025]
    rows = []
    for _, r in t[t["구분"] != "규칙"].head(n).iterrows():
        s = score_fn(r["모델"], SETS[r["피처"]])(tr, te)
        fa = ranking(te["label"].values.astype(int), s)["헛경보@90%"]
        rows.append({**r[["피처", "구분", "모델", "헛경보@90%", "헛경보@100%", "PR-AUC", "ROC-AUC"]].to_dict(),
                     "2024 학습→2025 헛경보@90%": fa, "나쁜 쪽": max(fa, r["헛경보@90%"])})
    return pd.DataFrame(rows).sort_values(["나쁜 쪽", "헛경보@100%", "PR-AUC"], ascending=[True, True, False]).reset_index(drop=True)


def main(only=None):
    d = jd.day_table()
    d = d[d["date"] < "2026-01-01"].reset_index(drop=True)   # 2026은 보지 않는다
    miss = d.loc[d["date"] >= "2025-01-01", SUKUB].isna().mean().max()
    print(f"표 {d['date'].min().date()} ~ {d['date'].max().date()}, 급락일 2024 {int(d.loc[d.date.dt.year == 2024, 'label'].sum())}"
          f" · 2025 {int(d.loc[d.date.dt.year == 2025, 'label'].sum())}, 2025 계통실적 최대 결측률 {miss:.3f}", flush=True)
    preds_all, rows = {}, []
    for sname, cols in SETS.items():
        if only and not sname.startswith(only):
            continue
        t0 = time.time()
        cache = CACHE / f"{sname.split()[0]}.pkl"
        if cache.exists():   # 같은 묶음 예측을 이미 계산했으면 다시 쓰기 (모델·피처를 바꾸면 파일을 지울 것)
            p = pd.read_pickle(cache)
        else:
            p = run_set(d, sname, cols)
            CACHE.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(p, cache)
        for k, v in p.items():
            preds_all[(sname, k)] = v
            kind = "앙상블" if k in ENSEMBLES else ("규칙" if BASES[k].kind == "rule" else "단일")
            rows.append({"피처": sname, "구분": kind, "모델": k, "급락일": int(v["label"].sum()),
                         **ranking(v["label"].values.astype(int), v["s"].values)})
        print(f"{sname} 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    t = pd.DataFrame(rows).sort_values(ORDER[0], ascending=ORDER[1]).reset_index(drop=True)
    t.insert(0, "순위", range(1, len(t) + 1))
    TAB.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(TAB / "J4_zoo_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.head(40).round(3).to_string(index=False), flush=True)
    if only:
        return

    # 상위 5개(규칙 제외): 매월 재학습도 같은 규칙으로 비교
    top5 = t[t["구분"] != "규칙"].head(5)
    mrows = []
    for _, r in top5.iterrows():
        f = score_fn(r["모델"], SETS[r["피처"]])
        for sch_name, sch in (("고정(F1·F2)", jd.folds), ("매월 재학습", jd.monthly)):
            ss, yy = [], []
            for tr, te in sch(d):
                ss.append(f(tr, te))
                yy.append(te["label"].values)
            mrows.append({"피처": r["피처"], "모델": r["모델"], "방식": sch_name, **ranking(np.concatenate(yy).astype(int), np.concatenate(ss))})
            print(f"  매월 비교 {r['피처']} · {r['모델']} · {sch_name}: 헛경보@90% {mrows[-1]['헛경보@90%']}", flush=True)
    mt = pd.DataFrame(mrows).sort_values(ORDER[0], ascending=ORDER[1])
    mt.round(4).to_csv(TAB / "J5_zoo_monthly_2025.csv", index=False, encoding="utf-8-sig")
    print(mt.round(3).to_string(index=False), flush=True)

    top = [(r["피처"], r["모델"]) for _, r in t[t["구분"] != "규칙"].head(10).iterrows()]
    bs = bootstrap(preds_all, top)
    bs.round(4).to_csv(TAB / "J6_zoo_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(bs.round(3).to_string(index=False), flush=True)
    nt = nested_topk(preds_all)
    nt.round(4).to_csv(TAB / "J7_zoo_nested_F2.csv", index=False, encoding="utf-8-sig")
    print(nt.head(15).round(3).to_string(index=False), flush=True)

    st = stability(d, t)
    st.round(4).to_csv(TAB / "J8_zoo_stability_2025.csv", index=False, encoding="utf-8-sig")
    print(st.round(3).to_string(index=False), flush=True)
    best = st.iloc[0]
    m = mt[(mt["피처"] == best["피처"]) & (mt["모델"] == best["모델"])]
    scheme = m.iloc[0]["방식"] if len(m) else "고정(F1·F2)"   # 매월 재학습 비교에 있으면 더 나은 쪽
    inc = t[(t["피처"] == INCUMBENT[0]) & (t["모델"] == INCUMBENT[1])].iloc[0]
    pick = {"피처": best["피처"], "피처 목록": SETS[best["피처"]], "모델": best["모델"], "재학습": scheme,
            "헛경보@90%": int(best["헛경보@90%"]), "2024 학습→2025 헛경보@90%": int(best["2024 학습→2025 헛경보@90%"]),
            "헛경보@100%": int(best["헛경보@100%"]), "PR-AUC": float(best["PR-AUC"]),
            "1차 선택(S4 랜덤포레스트 분류) 헛경보@90%": int(inc["헛경보@90%"]), "후보 수": int(len(t)),
            "규칙": "F1·F2 이어 붙인 2025 표본 밖 상위 20개 → 두 검증(F1·F2, 2024 학습→2025 전체) 중 나쁜 쪽 헛경보@90% → "
                    "F1·F2 @100% → PR-AUC (안정성 기준은 2026-10-25 1차 대규모 비교 결과를 본 뒤 추가, 사용자 확인)"}
    (TAB / "J_zoo_pick.json").write_text(json.dumps(pick, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in pick.items() if k != "피처 목록"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[a.index("--only") + 1] if "--only" in a else None)
