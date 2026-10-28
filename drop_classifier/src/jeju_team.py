"""제주 급락 1차(15시) 팀 규칙 재선택 — 학습 2023-01부터, 휴일·주말로 선택·채점 (ROLES.md R2, 4.1).

규칙 (JEJU.md 결정 기록 2026-10-28, 2026 채점 전 고정)
  표: jeju_drop.day_table(start=2023-01-01). 학습 행 = 전체 날(평일 포함), 선택·문턱·채점 = 휴일·주말만
  2023 빈칸: Open-Meteo 제주·육지 일사(2024-01부터)만 남음 → 학습 구간 중앙값 + has_om(0/1) 열, LightGBM은 빈칸 그대로
  후보: 피처 3종(S4·S5·S5c, 각각 + has_om) × 모델 7개(로지스틱 L2·L1, 나이브 베이즈, KNN 분류, RF 분류, LightGBM 분류,
        KNN+로지스틱 L2+RF 확률 평균) = 21개. 설정은 jeju_zoo.BASES 그대로
  선택: 2025 휴일·주말 PR-AUC, 두 검증 중 나쁜 쪽
        (i) 2023-01 ~ 2024-12 학습 → 2025 전체  (ii) 2023-01 ~ 2025-06 학습 → 2025 하반기
        1위와 0.03 이내면 단순한 쪽(선형·나이브 베이즈 < KNN·RF·LightGBM < 3개 결합) → 헛경보@재현율80%((i)) 적은 쪽 → PR-AUC
  문턱 ((i) 2025 휴일·주말 예측): 주의 = 재현율 80%, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨
        (없으면 적중 3일 이상 중 정밀도 최고) — 육지 team_final.alarm_threshold와 같음. 확률 = 같은 예측에 로지스틱 보정
  최종: 2023-01 ~ 2025-12 고정 학습 → 2026-01-01 ~ 10-18 휴일·주말 한 번 채점 (team_scorecard 지표), 현재 모델과 나란히
출력: outputs/tables/J9_team_select_2025.csv, J9_team_pick.json / (--test) drop_제주_15h.csv, drop_choice_제주.json, drop_score_제주.csv
실행: python -m drop_classifier.src.jeju_team          (선택, 2025만)
      python -m drop_classifier.src.jeju_team --test   (2026 한 번 채점)
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drop_classifier.src import jeju_drop as jd
from drop_classifier.src import jeju_zoo as jz
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_final import alarm_threshold
from drop_classifier.src.team_select import score_rows
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
TAB = jd.TAB
REGION = "제주"
START = "2023-01-01"
VAL = {"(i) 2023~2024 → 2025": (("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")),
       "(ii) 2023~2025상 → 2025하": (("2023-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"))}
TR_FIN, TEST = ("2023-01-01", "2025-12-31"), ("2026-01-01", "2026-10-18")
RECALL_WARN = 0.8
TIE = 0.03
FLAG = ["has_om"]
FEATS = {k: v + FLAG for k, v in jz.SETS.items()}
SINGLE = {"로지스틱(L2)": 0, "로지스틱(L1)": 0, "나이브 베이즈": 0, "KNN(분류)": 1, "랜덤포레스트(분류)": 1, "LightGBM(분류)": 1}
ENS = {"KNN+로지스틱+RF (확률 평균)": ["KNN(분류)", "로지스틱(L2)", "랜덤포레스트(분류)"]}
TIER = {**SINGLE, **{k: 2 for k in ENS}}


def table():
    d = jd.day_table(start=START)
    d["has_om"] = d["jo_rad_mid"].notna().astype(float)
    return d


def fit_score(name, tr, te, cols, cache=None):
    """후보 이름 → 위험 점수 (모두 분류 확률, 높을수록 위험). cache: 같은 학습·예측의 단일 모델 점수 재사용."""
    cache = {} if cache is None else cache
    def one(m):
        if m not in cache:
            cache[m] = jz.BASES[m](tr, te, cols)
        return cache[m]
    return np.mean([one(m) for m in ENS[name]], axis=0) if name in ENS else one(name)


def rows_of(d, tr_rng, te_rng):
    tr = d[d["date"].between(*tr_rng) & d["label"].notna()]
    te = d[d["date"].between(*te_rng) & d["label"].notna()]
    return tr, te


def select():
    t0 = time.time()
    d = table()
    print(f"표 {d['date'].min():%Y-%m-%d} ~ {d['date'].max():%Y-%m-%d}, 2023 Open-Meteo 있음 "
          f"{d.loc[d['date'] < '2024-01-01', 'has_om'].mean():.0%}", flush=True)
    rows = []
    for fname, cols in FEATS.items():
        res = {}
        for vname, (tr_rng, te_rng) in VAL.items():
            tr, te = rows_of(d, tr_rng, te_rng)
            off = te["offday"].astype(bool).values
            y = te.loc[off, "label"].astype(int).values
            cache = {}
            for m in TIER:
                s = fit_score(m, tr, te, cols, cache)[off]
                res[(vname, m)] = drop_stats(y, s)
            print(f"  {fname} / {vname}: 학습 {len(tr)}일(급락 {int(tr['label'].sum())}), 휴일·주말 {len(y)}일(급락 {y.sum()}) "
                  f"{time.time() - t0:.0f}초", flush=True)
        vi, vii = VAL
        for m in TIER:
            a, b = res[(vi, m)], res[(vii, m)]
            rows.append({"피처": fname, "모델": m, "단계": TIER[m], "PR-AUC (i)": a["PR-AUC"], "PR-AUC (ii)": b["PR-AUC"],
                         "PR-AUC 나쁜쪽": min(a["PR-AUC"], b["PR-AUC"]), "헛경보@80% (i)": a["헛경보@재현율80%"],
                         "헛경보@90% (i)": a["헛경보@재현율90%"], "헛경보@80% (ii)": b["헛경보@재현율80%"],
                         "ROC-AUC (i)": a["ROC-AUC"]})
    sel = pd.DataFrame(rows).sort_values("PR-AUC 나쁜쪽", ascending=False).reset_index(drop=True)
    best = sel["PR-AUC 나쁜쪽"].iloc[0]
    sel["동점권"] = sel["PR-AUC 나쁜쪽"] >= best - TIE
    pick = sel[sel["동점권"]].sort_values(["단계", "헛경보@80% (i)", "PR-AUC 나쁜쪽"], ascending=[True, True, False]).iloc[0]
    sel["선택"] = (sel["피처"] == pick["피처"]) & (sel["모델"] == pick["모델"])
    sel.round(4).to_csv(TAB / "J9_team_select_2025.csv", index=False, encoding="utf-8-sig")
    (TAB / "J9_team_pick.json").write_text(json.dumps({"피처": pick["피처"], "모델": pick["모델"]}, ensure_ascii=False),
                                          encoding="utf-8")
    pd.set_option("display.width", 250)
    print(sel.round(3).to_string(index=False))
    print(f"선택: {pick['피처']} / {pick['모델']} (총 {time.time() - t0:.0f}초)")


def current_rows(test_days):
    """현재 운영 모델(S5 KNN, 2024~2025 학습, JD_choice_v1 문턱)의 같은 날 성적 — drop_prob_제주.csv 점수."""
    c = pd.read_csv(TAB / "drop_prob_제주.csv", parse_dates=["date"]).set_index("date")
    ch = json.loads((jd.TAB2 / "JD_choice_v1.json").read_text(encoding="utf-8"))
    t = test_days.join(c[["score", "prob"]], on="date")
    y, s = t["label"].astype(int).values, t["score"].values
    return score_rows(f"2026 시험 {TEST[0]} ~ {TEST[1]}", y, s, t["prob"].values, ch["주의 문턱"], ch["강한 경보 문턱"])


def test():
    t0 = time.time()
    pk = json.loads((TAB / "J9_team_pick.json").read_text(encoding="utf-8"))
    cols, name = FEATS[pk["피처"]], pk["모델"]
    d = table()
    tr_v, va = rows_of(d, *VAL["(i) 2023~2024 → 2025"])
    va = d[d["date"].between("2025-01-01", "2025-12-31")]
    sv = fit_score(name, tr_v, va, cols)
    tr_f = d[d["date"].between(*TR_FIN) & d["label"].notna()]
    te = d[d["date"] >= TEST[0]]
    st = fit_score(name, tr_f, te, cols)
    vo = (va["offday"].astype(bool) & va["label"].notna()).values
    yv = va.loc[vo, "label"].astype(int).values
    warn, (strong, how) = threshold_for(yv, sv[vo], RECALL_WARN), alarm_threshold(yv, sv[vo])
    if strong < warn:   # 경보는 주의보다 엄격해야 함 (주의 = 재현율 80%에서 이미 정밀도 50%를 넘으면 경보 = 주의)
        strong, how = warn, f"주의 문턱과 같음 (정밀도 50% 문턱 {strong:.4f}이 주의보다 느슨, 2025 주의 정밀도 50% 이상)"
    cal = LogisticRegression(C=1e6).fit(sv[vo].reshape(-1, 1), yv)
    to_prob = lambda s: cal.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    print(f"문턱: 주의 {warn:.4f}, 경보 {strong:.4f} ({how})", flush=True)

    to = (te["offday"].astype(bool) & te["date"].between(*TEST) & te["label"].notna()).values
    yt = te.loc[to, "label"].astype(int).values
    name_new = f"{pk['피처']} · {name} (2023~2025 학습)"
    sc = pd.DataFrame(score_rows("2025 검증", yv, sv[vo], to_prob(sv[vo]), warn, strong)
                      + score_rows(f"2026 시험 {TEST[0]} ~ {TEST[1]}", yt, st[to], to_prob(st[to]), warn, strong))
    sc.insert(0, "모델", name_new)
    cur = pd.DataFrame(current_rows(te.loc[to, ["date", "label"]]))
    cur.insert(0, "모델", "현재: S5 · KNN(분류) (2024~2025 학습, 비교용)")
    sc = pd.concat([sc, cur], ignore_index=True)
    sc.round(4).to_csv(TAB / f"drop_score_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 4.1 일별 (2025 = (i) 예측, 2026~ = 최종 모델). label = 휴일·주말만, 미확정 빈칸
    both = pd.concat([va.assign(s=sv), te.assign(s=st)])
    lab = both["label"]
    out = pd.DataFrame({"base_date": (both["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
                        "date": both["date"].dt.strftime("%Y-%m-%d"),
                        "grade": np.where(both["s"] >= strong, "경보", np.where(both["s"] >= warn, "주의", "없음")),
                        "prob": to_prob(both["s"]).round(4), "score": both["s"].round(4),
                        "label": np.where(both["offday"].astype(bool) & lab.notna(), lab.fillna(0).astype(int).astype(str), "")})
    out.to_csv(TAB / f"drop_{REGION}_15h.csv", index=False, encoding="utf-8-sig")

    choice = {"stage": "15h (전날 15시)", "계통": REGION, "모델": name, "구성": ENS.get(name, [name]), "피처 묶음": pk["피처"],
              "피처": cols, "점수": "score = 급락 분류 확률(모델 출력, 0~1), 높을수록 위험",
              "문턱": {"주의": float(warn), "경보": float(strong), "규칙": "score ≥ 문턱이면 해당 단계 (반올림 전 값)"},
              "확률식": {"형태": "prob = 1/(1+exp(-(a·score+b)))", "a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])},
              "학습": {"문턱·확률": "2023-01-01 ~ 2024-12-31 학습 → 2025 휴일·주말 예측",
                     "최종": "2023-01-01 ~ 2025-12-31 (고정, 2026 예측에 사용)", "학습 행": "전체 날(평일 포함)"},
              "문턱 규칙": "주의 = 2025 휴일·주말 급락 80% 포착, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨 "
                       "(없으면 적중 3일 이상 중 정밀도 최고, 같으면 가장 느슨)",
              "경보 문턱 적용": how,
              "선택 절차": "21개 후보(피처 3 × 모델 7) 2025 휴일·주말 PR-AUC 두 검증 중 나쁜 쪽, 0.03 이내면 단순한 쪽 (J9_team_select_2025.csv)",
              "2023 자료": "구역예보 제주 2023-01~11·5분 계통 실적 2023-03~12 보강(2026-10-27). Open-Meteo는 2024-01부터 → 학습 중앙값 + has_om",
              "급락 정의": "11~15시 최저 SMP 30원 미만, 휴일·주말만 채점"}
    (TAB / f"drop_choice_{REGION}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")

    show = ["모델", "구간", "경보", "일수", "급락일", "PR-AUC", "PR-AUC 95%하한", "PR-AUC 95%상한", "ROC-AUC", "헛경보@재현율80%",
            "헛경보@재현율90%", "Brier", "적중", "놓침", "헛경보", "재현율", "정밀도"]
    pd.set_option("display.width", 300)
    print(sc[show].round(3).to_string(index=False))
    a26 = out[out["date"].between(*TEST)]
    print("2026 등급 일수 (전체 날):", a26["grade"].value_counts().to_dict())
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    test() if "--test" in sys.argv[1:] else select()
