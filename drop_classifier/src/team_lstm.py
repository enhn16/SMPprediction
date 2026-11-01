"""팀 최종 합의 모델 (육지 급락, 1차 15시): LSTM + LightGBM(확장 변수) 두 점수의 평균 — 세부는 2025 검증으로만 선택.

결정: PLAN.md 결정 기록 2026-10-28 (사전 고정). 급락 = 모든 날 → 학습·문턱·채점 전체 날.
구성원 출발점 = ens8.py (LSTM: 시간별 SMP 흐름 + 표 피처, LightGBM: BASE+SOLAR + 확장 변수). 규제 고정, 시드 5개.
후보 (채점 전 고정)
  학습 목표  T1 분류(한낮 최저 30원 미만) / T2 회귀(한낮 최저 SMP, 0~150원 자름, 점수 = −예측) / T3 분류(50원 미만)
  LSTM 입력  168시간 / 336시간
  확장 변수  연료비 + SMP 7일 수준 / SMP 7일 수준만
  결합 척도  (분류만) 확률 평균 / 로짓 평균. 가중은 1:1 고정
선택 (2026-10-28 수정, 2026 미열람): 2023~2024 학습·2024 학습 → 2025 전체 날 헛경보@재현율80%의 나쁜 쪽(큰 값) 순위,
      2일 이내 동점 → PR-AUC 나쁜 쪽 높은 쪽 → 기본안(T1·168·연료비+수준·확률 평균)과 다른 축이 적은 쪽.
      (처음 규칙은 PR-AUC 나쁜 쪽 순위였음 — 주의 문턱 = 재현율 80%라 실전 성적에 가까운 지표로 바꿈, PLAN.md)
문턱·확률: 2023~2024 학습 → 2025 전체 날 예측. 주의 = 재현율 80%, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨
      (없으면 적중 3일 이상 중 정밀도 최고). 로지스틱 보정.
최종: 2023~2025 학습 → 2026-01-01 ~ 10-18 전체 날 한 번 채점 (성적 행: 전체 날·휴일·주말·평일).
실행: python -m drop_classifier.src.team_lstm select   → 22_team_lstm_select_2025_육지.csv, 22_team_lstm_pick.json (2026 안 봄)
      python -m drop_classifier.src.team_lstm repick   → 저장된 2025 표로 선택 규칙만 다시 적용
      python -m drop_classifier.src.team_lstm final    → 22_team_lstm_score_육지.csv, 22_team_lstm_daily_육지.csv
      python -m drop_classifier.src.team_lstm final --official → + drop_육지_15h.csv, drop_choice_육지.json, drop_score_육지.csv
"""
import json
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from common.config import SEED
from common.loaders import load_smp_actual
from drop_classifier.src.ens8 import COLS, LGBM, LSTMCls, data as ens8_data
from drop_classifier.src.final import threshold_for
from drop_classifier.src.fuel_level import FUEL, LEVEL
from drop_classifier.src.team_final import alarm_threshold
from drop_classifier.src.team_select import boot, tier_row
from drop_classifier.src.two_tier import OUT_TAB
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
REGION = "육지"
TR_SEL = {"2023~2024": ("2023-01-01", "2024-12-31"), "2024": ("2024-01-01", "2024-12-31")}
VALID = ("2025-01-01", "2025-12-31")
TR_FIN, TEST = ("2023-01-01", "2025-12-31"), ("2026-01-01", "2026-10-18")
SEEDS = (0, 1, 2, 3, 4)
CLIP = (0, 150)
TARGETS = {"T1": "분류 30원 미만", "T2": "회귀 한낮 최저", "T3": "분류 50원 미만"}
SEQ_LENS = (168, 336)
XSETS = {"연료비+수준": FUEL + LEVEL, "수준만": LEVEL}
BASE_CFG = {"target": "T1", "seq": 168, "x": "연료비+수준", "comb": "확률 평균"}
MID = [11, 12, 13, 14, 15]


# ---------------------------------------------------------------- 데이터
def add_seqs(df: pd.DataFrame, smp: pd.DataFrame | None = None) -> pd.DataFrame:
    """seq168·seq336 = 대상일 0시 직전 n시간 SMP (D-1 24시까지 = 판단 시각에 확정된 실적). 육지는 엑셀 = 공표값."""
    s = (load_smp_actual(REGION) if smp is None else smp).sort_values(["date", "hour"])
    vals = s["smp"].values.astype("float32")
    pos = {pd.Timestamp(d): i * 24 for i, d in enumerate(s["date"].values[::24])}
    last = pd.Timestamp(s["date"].max()) + pd.Timedelta(days=1)          # 내일 = 마지막 실적일 다음 날
    pos[last] = len(vals)
    for n in SEQ_LENS:
        df[f"seq{n}"] = [vals[pos[d] - n:pos[d]] if d in pos and pos[d] >= n else np.full(n, np.nan, "float32")
                         for d in df["date"]]
    return df


def data() -> pd.DataFrame:
    df = ens8_data(team=True).drop(columns=["seq"])
    df = add_seqs(df)
    df["label50"] = (df["min_smp_11_15"] < 50).astype(int)
    return df


def split(df, rng):
    return df[df["date"].between(*rng)]


# ---------------------------------------------------------------- 구성원 (반환: 분류 = 확률, 회귀 = 예상 한낮 최저 원)
def target_y(tr, target):
    if target == "T2":
        return tr["min_smp_11_15"].clip(*CLIP).values.astype("float32")
    return tr["label50" if target == "T3" else "label"].values.astype("float32")


def lstm(tr, te, target="T1", seq=168, seeds=SEEDS, extra=()):
    """ens8.lstm과 같은 구조·규제 (은닉 16, 드롭아웃 0.3, AdamW lr 3e-3·감쇠 1e-2, 마지막 1/7 조기 종료). 회귀는 MSE(표준화 목표).
    extra: 표 피처에 더할 열 (추가 자료 실험, team_lstm_extra.py)."""
    key = f"seq{seq}"
    cols = COLS + list(extra)
    sx = SimpleImputer(strategy="median").fit(tr[cols])
    sc = StandardScaler().fit(sx.transform(tr[cols]))
    tab = lambda d: torch.tensor(sc.transform(sx.transform(d[cols])), dtype=torch.float32)
    allseq = np.stack(tr[key])
    mu, sd = np.nanmean(allseq), np.nanstd(allseq)
    sq = lambda d: torch.nan_to_num(torch.tensor((np.stack(d[key]) - mu) / sd, dtype=torch.float32))
    yr = target_y(tr, target)
    reg = target == "T2"
    ym, ys = (yr.mean(), yr.std()) if reg else (0.0, 1.0)
    y = torch.tensor((yr - ym) / ys, dtype=torch.float32)
    Xs, Xt, Ts, Tt = sq(tr), tab(tr), sq(te), tab(te)
    n_val = max(30, len(tr) // 7)
    if reg:
        loss_fn = torch.nn.MSELoss()
    else:
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
            out = net(Ts, Tt).numpy()
        preds.append(out * ys + ym if reg else 1 / (1 + np.exp(-out)))
    return np.mean(preds, axis=0)


def lgbm(tr, te, target="T1", x="연료비+수준", extra=()):
    cols = COLS + XSETS[x] + list(extra)
    y = target_y(tr, target)
    if target == "T2":
        return LGBMRegressor(**{k: v for k, v in LGBM.items() if k != "class_weight"}).fit(tr[cols], y).predict(te[cols])
    return LGBMClassifier(**LGBM).fit(tr[cols], y.astype(int)).predict_proba(te[cols])[:, 1]


def combine(p_lstm, p_lgbm, target, comb):
    """위험 점수 (클수록 위험). 회귀 = −두 예상 한낮 최저의 평균, 분류 = 확률 평균 또는 로짓 평균."""
    if target == "T2":
        return -(p_lstm + p_lgbm) / 2
    if comb == "로짓 평균":
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return (lg(p_lstm) + lg(p_lgbm)) / 2
    return (p_lstm + p_lgbm) / 2


def candidates():
    for t, n, x in product(TARGETS, SEQ_LENS, XSETS):
        for c in (("확률 평균", "로짓 평균") if t != "T2" else ("평균 (원)",)):
            yield {"target": t, "seq": n, "x": x, "comb": c}


def n_diff(cfg):
    return sum(cfg[k] != BASE_CFG[k] for k in ("target", "seq", "x")) + (cfg["comb"] == "로짓 평균")


def name_of(cfg):
    return f"{cfg['target']} {TARGETS[cfg['target']]} · LSTM {cfg['seq']}시간 · 확장 {cfg['x']} · {cfg['comb']}"


# ---------------------------------------------------------------- ① 선택 (2025만)
def select():
    t0 = time.time()
    df = data()
    df = df[df["date"] <= VALID[1]].reset_index(drop=True)              # 2026 행 제거 (열지 않음)
    va = split(df, VALID)
    y, off = va["label"].values, va["offday"].values
    print(f"2025 전체 {len(y)}일 급락 {int(y.sum())} (휴일·주말 {int(y[off].sum())}, 평일 {int(y[~off].sum())})", flush=True)
    cache = {}
    for w, rng in TR_SEL.items():
        tr = split(df, rng)
        for t, n in product(TARGETS, SEQ_LENS):
            cache[("lstm", w, t, n)] = lstm(tr, va, t, n)
            print(f"  LSTM {w} {t} {n}h: {time.time() - t0:.0f}초", flush=True)
        for t, x in product(TARGETS, XSETS):
            cache[("lgbm", w, t, x)] = lgbm(tr, va, t, x)
    rows = []
    for cfg in candidates():
        r = {"후보": name_of(cfg), **cfg, "기본안과 다른 축": n_diff(cfg)}
        for w in TR_SEL:
            s = combine(cache[("lstm", w, cfg["target"], cfg["seq"])], cache[("lgbm", w, cfg["target"], cfg["x"])],
                        cfg["target"], cfg["comb"])
            a, o = drop_stats(y, s), drop_stats(y[off], s[off])
            r[f"PR-AUC {w}학습"] = a["PR-AUC"]
            r[f"헛경보@80% {w}학습"] = a["헛경보@재현율80%"]
            r[f"헛경보@90% {w}학습"] = a["헛경보@재현율90%"]
            r[f"평일 헛경보@80% {w}학습"] = int(((s >= threshold_for(y, s, .8)) & ~off & (y == 0)).sum())
            r[f"휴일·주말 PR-AUC {w}학습"] = o["PR-AUC"]
        r["PR-AUC 나쁜쪽"] = min(r[f"PR-AUC {w}학습"] for w in TR_SEL)
        rows.append(r)
    for w in TR_SEL:                                                     # 구성원 단독 (참고)
        for t, n in product(TARGETS, SEQ_LENS):
            s = cache[("lstm", w, t, n)] * (-1 if t == "T2" else 1)
            rows.append({"후보": f"(참고) LSTM 단독 {t} {n}시간", f"PR-AUC {w}학습": drop_stats(y, s)["PR-AUC"],
                         f"헛경보@80% {w}학습": drop_stats(y, s)["헛경보@재현율80%"]})
        for t, x in product(TARGETS, XSETS):
            s = cache[("lgbm", w, t, x)] * (-1 if t == "T2" else 1)
            rows.append({"후보": f"(참고) LightGBM 단독 {t} {x}", f"PR-AUC {w}학습": drop_stats(y, s)["PR-AUC"],
                         f"헛경보@80% {w}학습": drop_stats(y, s)["헛경보@재현율80%"]})
    res = pd.DataFrame(rows)
    res = res.groupby("후보", sort=False).first().reset_index()          # 참고 행 두 학습 기간 합치기
    res.round(4).to_csv(OUT_TAB / f"22_team_lstm_select_2025_{REGION}.csv", index=False, encoding="utf-8-sig")
    pick_from_table()
    print(f"({time.time() - t0:.0f}초)")


FA_TIE = 2


def pick_from_table():
    """22_team_lstm_select_2025 표에 선택 규칙 적용 (LSTM 재학습 없이 다시 고를 수 있게 분리)."""
    path = OUT_TAB / f"22_team_lstm_select_2025_{REGION}.csv"
    res = pd.read_csv(path, encoding="utf-8-sig").drop(columns=["선택"], errors="ignore")
    cand = res[res["target"].notna()].copy()
    cand["헛경보@80% 나쁜쪽"] = cand[[f"헛경보@80% {w}학습" for w in TR_SEL]].max(axis=1)
    best = cand["헛경보@80% 나쁜쪽"].min()
    tied = cand[cand["헛경보@80% 나쁜쪽"] <= best + FA_TIE]
    pick = tied.sort_values(["PR-AUC 나쁜쪽", "기본안과 다른 축"], ascending=[False, True]).iloc[0]
    res = res.merge(cand[["후보", "헛경보@80% 나쁜쪽"]], on="후보", how="left")
    res["선택"] = np.where(res["후보"] == pick["후보"], "선택", np.where(res["후보"].isin(tied["후보"]), "동점권", ""))
    res = pd.concat([res[res["target"].notna()].sort_values(["헛경보@80% 나쁜쪽", "PR-AUC 나쁜쪽"], ascending=[True, False]),
                     res[res["target"].isna()]])
    res.round(4).to_csv(path, index=False, encoding="utf-8-sig")
    cfg = {k: (int(pick[k]) if k == "seq" else pick[k]) for k in BASE_CFG}
    (OUT_TAB / "22_team_lstm_pick.json").write_text(json.dumps(
        {"선택": cfg, "이름": pick["후보"], "헛경보@80% 나쁜쪽": int(pick["헛경보@80% 나쁜쪽"]),
         "PR-AUC 나쁜쪽": float(pick["PR-AUC 나쁜쪽"]), "동점권": tied["후보"].tolist(),
         "규칙": "PLAN.md 2026-10-28 (선택 지표 변경: 헛경보@재현율80% 나쁜 쪽)"}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["후보", "헛경보@80% 나쁜쪽", "PR-AUC 나쁜쪽", "헛경보@80% 2023~2024학습", "헛경보@80% 2024학습",
            "평일 헛경보@80% 2023~2024학습", "PR-AUC 2023~2024학습", "PR-AUC 2024학습", "선택"]
    print(res[res["target"].notna()][show].round(3).to_string(index=False))
    print(f"선택: {pick['후보']}")


# ---------------------------------------------------------------- ② 최종 (2026 한 번 채점)
def score_fn(cfg, extra=()):
    def f(tr, te):
        p1 = lstm(tr, te, cfg["target"], cfg["seq"], extra=extra)
        p2 = lgbm(tr, te, cfg["target"], cfg["x"], extra=extra)
        return combine(p1, p2, cfg["target"], cfg["comb"]), p1, p2
    return f


def rows_for(period, scope, y, s, prob, warn, strong):
    st = drop_stats(y, s)
    base = {"구간": period, "범위": scope, "일수": len(y), "급락일": int(y.sum()), **st, "Brier": brier_score_loss(y, prob)}
    if y.sum() >= 3:                                                     # 부트스트랩은 급락 3일 이상일 때만 (team_select.boot 조건)
        bs = boot(y, s)
        for k in ("PR-AUC", "헛경보@재현율80%", "헛경보@재현율90%"):
            base[f"{k} 95%하한"], base[f"{k} 95%상한"] = bs[k].quantile(.025), bs[k].quantile(.975)
    return [{**base, "경보": "주의 이상", **tier_row(y, s >= warn)}, {**base, "경보": "경보", **tier_row(y, s >= strong)}]


def final(official=False, extra=(), tag="", df=None):
    """extra·tag·df: 추가 자료 실험(team_lstm_extra.py)용 — 결과는 22{tag}_* 파일에만 쓰고 공식 파일은 쓰지 않는다."""
    t0 = time.time()
    cfg = json.loads((OUT_TAB / "22_team_lstm_pick.json").read_text(encoding="utf-8"))["선택"]
    name = "LSTM + LightGBM(확장 변수) 평균 — " + name_of(cfg) + (f" + {tag.strip('_')}" if extra else "")
    fn = score_fn(cfg, extra)
    df = data() if df is None else df
    official = official and not extra
    va, te = split(df, VALID), split(df, TEST)
    sv, lv, gv = fn(split(df, TR_SEL["2023~2024"]), va)
    t_val = time.time() - t0
    st, lt, gt = fn(split(df, TR_FIN), te)
    yv = va["label"].values
    warn, (strong, how) = threshold_for(yv, sv, .8), alarm_threshold(yv, sv)
    cal = LogisticRegression(C=1e6).fit(sv.reshape(-1, 1), yv)
    to_prob = lambda s: cal.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    reg = cfg["target"] == "T2"
    unit = (lambda s: f"{-s:.1f}원") if reg else (lambda s: f"{s:.3f}")
    print(f"문턱: 주의 {unit(warn)}, 경보 {unit(strong)} ({how})", flush=True)

    rows = []
    for period, d, s in (("2025 검증", va, sv), (f"2026 시험 {TEST[0]} ~ {TEST[1]}", te, st)):
        y, off = d["label"].values, d["offday"].values
        for scope, m in (("전체 날 (공식)", np.ones(len(y), bool)), ("휴일·주말", off), ("평일", ~off)):
            rows += rows_for(period, scope, y[m], s[m], to_prob(s[m]), warn, strong)
    sc = pd.DataFrame(rows)
    sc.insert(0, "모델", name)
    sc.round(4).to_csv(OUT_TAB / f"22{tag}_team_lstm_score_{REGION}.csv", index=False, encoding="utf-8-sig")

    both = pd.concat([va.assign(s=sv, p_lstm=lv, p_lgbm=gv), te.assign(s=st, p_lstm=lt, p_lgbm=gt)])
    daily = pd.DataFrame({"base_date": (both["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
                          "date": both["date"].dt.strftime("%Y-%m-%d"),
                          "grade": np.where(both["s"] >= strong, "경보", np.where(both["s"] >= warn, "주의", "없음")),
                          "prob": to_prob(both["s"]).round(4),
                          "score": (-both["s"]).round(1) if reg else both["s"].round(4),
                          "label": both["label"].astype(int).astype(str).values})
    daily.assign(offday=both["offday"].values, lstm=both["p_lstm"].round(4).values,
                 lgbm=both["p_lgbm"].round(4).values).to_csv(
        OUT_TAB / f"22{tag}_team_lstm_daily_{REGION}.csv", index=False, encoding="utf-8-sig")

    if official:
        daily.to_csv(OUT_TAB / f"drop_{REGION}_15h.csv", index=False, encoding="utf-8-sig")
        sc.round(4).to_csv(OUT_TAB / f"drop_score_{REGION}.csv", index=False, encoding="utf-8-sig")
        choice = {
            "stage": "15h (전날 15시)", "모델": "LSTM + LightGBM(확장 변수) 두 점수의 평균 (팀 최종 합의 2026-10-28)",
            "세부 선택": cfg, "이름": name,
            "LSTM": {"입력": f"직전 {cfg['seq']}시간 시간별 SMP(학습 구간 평균·표준편차로 정규화) + 표 피처",
                     "표 피처": COLS, "결측": "표 피처 학습 구간 중앙값 → 표준화",
                     "구조": "LSTM 1층 은닉 16 → [은닉, 표 피처] → 16 ReLU → 드롭아웃 0.3 → 1",
                     "학습": "AdamW lr 3e-3·감쇠 1e-2, 배치 32, 최대 150에폭, 학습 구간 마지막 1/7로 조기 종료(15회), 시드 5개 평균",
                     "목표": TARGETS[cfg["target"]] + (" (0~150원 자름, MSE, 표준화)" if reg else " (BCE, 양성 가중)")},
            "LightGBM": {"피처": COLS + XSETS[cfg["x"]], "설정": LGBM if not reg else
                         {k: v for k, v in LGBM.items() if k != "class_weight"}, "결측": "빈칸 그대로"},
            "결합": cfg["comb"] + " (1:1)",
            "점수": ("score = 예상 한낮(11~15시) 최저 SMP(원) = 두 모델 예측 평균, 낮을수록 위험" if reg else
                   "score = 두 확률의 " + cfg["comb"] + ", 높을수록 위험. 분류 모델이라 '예상 한낮 최저'는 없음"),
            "문턱": ({"주의": float(-warn), "경보": float(-strong), "규칙": "반올림 전 score ≤ 문턱이면 해당 단계"} if reg else
                   {"주의": float(warn), "경보": float(strong), "규칙": "반올림 전 score ≥ 문턱이면 해당 단계"}),
            "확률식": {"형태": "prob = 1/(1+exp(-(a·r+b))), r = " + ("−score" if reg else "score"),
                    "a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])},
            "학습": {"문턱·확률": "2023-01-01 ~ 2024-12-31 학습 → 2025 전체 날 예측",
                   "최종": "2023-01-01 ~ 2025-12-31 (고정, 2026 예측에 사용)"},
            "문턱 규칙": "주의 = 2025 전체 날 급락 80% 포착, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨 "
                     "(없으면 적중 3일 이상 중 정밀도 최고, 같으면 가장 느슨)",
            "경보 문턱 적용": how,
            "선택 절차": "PLAN.md 결정 기록 2026-10-28: 후보 20개를 2023~2024·2024 학습 → 2025 전체 날 PR-AUC 나쁜 쪽으로 선택 "
                     "(22_team_lstm_select_2025_육지.csv)",
            "급락 정의": "11~15시 최저 SMP 30원 미만, 모든 날(평일 포함)",
            "이전 운영 모델": "Ridge+EBM+랜덤포레스트 → drop_*_육지*_v1_ridge_ebm_rf"}
        (OUT_TAB / f"drop_choice_{REGION}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")

    show = ["구간", "범위", "경보", "일수", "급락일", "PR-AUC", "PR-AUC 95%하한", "PR-AUC 95%상한", "ROC-AUC",
            "헛경보@재현율80%", "헛경보@재현율90%", "Brier", "적중", "놓침", "헛경보", "재현율", "정밀도"]
    pd.set_option("display.width", 250)
    print(sc[[c for c in show if c in sc]].round(3).to_string(index=False))
    print(f"2025 예측(1회 학습) {t_val:.0f}초, 총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    if "repick" in sys.argv:
        pick_from_table()
    elif "select" in sys.argv:
        select()
    elif "final" in sys.argv:
        final("--official" in sys.argv)
