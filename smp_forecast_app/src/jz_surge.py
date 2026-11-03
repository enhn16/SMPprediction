"""제주 급등 1차(15시) — 육지 급등 v3(팀 합의 구성)를 제주에 (R2, 2026-11-03).

급등 정의: 제주 그날 최고 SMP ≥ 직전 28일 제주 일 최고 중앙값 × 1.25 (육지와 같은 식, 제주 SMP로).
  제주 급등일 2023 53 · 2024 55 · 2025 7 (2024 하반기부터 제주 가격이 육지와 거의 같아져 급감).
후보 (단순한 순, 규칙은 결과 보기 전 고정 — JEJU.md 11절)
  S1  육지 v3 점수 그대로 (surge_classifier.team_model3, 육지 SMP·수요로 학습) — 제주 모델 없음
  S2  v3 구성(C: 흐린 평일·주간전망 추가 + 휴무일 0)을 제주 SMP·제주 수요예측(jlfd)으로 학습. 날씨·주간전망은 육지와 같은 입력
  S2b S2에서 휴무일 0을 뺀 것 (B) — 제주 급등은 휴무일에도 있어서
선택: 2024~2025 매달 앞으로만 검증(학습 2023-01 ~ 전달) 제주 레이블 PR-AUC 최대, 0.03 이내면 단순한 쪽
문턱·확률: 같은 검증 점수로 v3 규칙 (0.01 격자, 주의 = 재현율 80% 최고선, 경보 = 정밀도 50%·적중 2일 / 없으면 정밀도 최고선), Platt
최종: 2023~2025 재학습 → 2026-01-01 ~ 10-18 한 번 채점
출력 (surge_classifier/outputs/tables, 제주 이름만 — 육지 파일은 건드리지 않음):
  surge_제주_15h.csv, surge_choice_제주_15h.json, surge_score_제주_15h.csv, surge_select_제주.csv
  모델: data/processed/surge_classifier/v3_models_제주.pkl (S2·S2b일 때)
실행: python -m smp_forecast_app.src.jz_surge
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common.loaders import load_smp_actual, load_smp_forecast_api
from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from surge_classifier.src import team_model3 as T3
from surge_classifier.src.label_days import OUT_DATA, OUT_TAB
from surge_classifier.src.models import score_rows

REGION = "제주"
TEST = T3.TEST
TIE = 0.03
MODEL_PKL = OUT_DATA / "v3_models_제주.pkl"
CANDS = {"S1 육지 v3 점수 그대로": None, "S2 v3(C) 제주 학습": FA.CANDS["C A+B"], "S2b v3(B) 제주 학습, 휴무일 포함": FA.CANDS["B 흐린 평일·주간전망 추가"]}
_LAND_HOURLY = T2.load_hourly


def jeju_hourly() -> pd.DataFrame:
    """T2.load_hourly와 같은 열 이름(smp_land·demand_fcst_land)에 제주 SMP·제주 수요예측(jlfd)."""
    s = load_smp_actual(REGION)[["date", "hour", "smp"]]
    f = load_smp_forecast_api()
    f = f[f["region"] == REGION][["date", "hour", "demand_fcst"]]
    h = s.merge(f, on=["date", "hour"], how="left").rename(columns={"hour": "hour_n", "smp": "smp_land",
                                                                    "demand_fcst": "demand_fcst_land"})
    h["date"] = h["date"].dt.strftime("%Y-%m-%d")
    return h.sort_values(["date", "hour_n"]).reset_index(drop=True)


def run(cfg, region):
    """(표, 2024~2025 앞으로만 점수, 최종 모델, 2026~ 점수)."""
    T2.load_hourly = jeju_hourly if region == REGION else _LAND_HOURLY
    FA.configure(cfg)
    try:
        d = T3.data()
        sv = FA.combine(T2.walk_forward(d), d, cfg)
        models = T2.final_models(d)
        te = d[(d["date_s"] >= "2026-01-01") & d[T2.FEATS[0]].notna()]
        st = FA.combine(T2.predict(models, te), d, cfg)
        feats = list(T2.FEATS)
    finally:
        FA.configure({})
        T2.load_hourly = _LAND_HOURLY
    return d, sv, models, st, feats


def main():
    dj = None
    res = {}
    for name, cfg in CANDS.items():
        if cfg is None:
            d, sv, models, st, feats = run(FA.CANDS["C A+B"], "육지")
        else:
            d, sv, models, st, feats = run(cfg, REGION)
            dj = d
        res[name] = dict(sv=pd.Series(sv.values, index=d.loc[sv.index, "date_s"].values),
                         st=pd.Series(st.values, index=d.loc[st.index, "date_s"].values), models=models, feats=feats, cfg=cfg)
    yj = dj.set_index("date_s")["y"]
    rows = []
    for i, (name, r) in enumerate(res.items()):
        s = r["sv"]
        y = yj.reindex(s.index)
        ok = y.notna().values
        yy, ss = y.values[ok].astype(int), s.values[ok]
        r["yv"], r["svv"], r["idx"] = yy, ss, s.index[ok]
        row = {"후보": name, "단순 순서": i, "검증 일수": len(yy), "급등일": int(yy.sum()), "PR-AUC 2024~2025": average_precision_score(yy, ss)}
        for yr in ("2024", "2025"):
            m = s.index[ok].str[:4] == yr
            row[f"PR-AUC {yr}"] = average_precision_score(yy[m], ss[m]) if yy[m].sum() else np.nan
            row[f"급등일 {yr}"] = int(yy[m].sum())
        off = dj.set_index("date_s")["is_off"].reindex(s.index[ok]).values == 1
        row["휴무일 급등일"] = int(yy[off].sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    best = t["PR-AUC 2024~2025"].max()
    pick = t[t["PR-AUC 2024~2025"] >= best - TIE].sort_values("단순 순서").iloc[0]["후보"]
    t["선택"] = np.where(t["후보"] == pick, "✓", "")
    t.round(4).to_csv(OUT_TAB / "surge_select_제주.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))

    r = res[pick]
    yv, sv = r["yv"], r["svv"]
    th = {"주의": T2.caution_line(yv, sv), "경보": T2.alert_line(yv, sv)}
    lr = LogisticRegression(C=1e6).fit(sv.reshape(-1, 1), yv)
    cal = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    if r["cfg"] is not None:
        MODEL_PKL.write_bytes(pickle.dumps(r["models"]))
    s = pd.concat([r["sv"][r["sv"].index >= "2025-01-01"], r["st"]])
    s = s[~s.index.duplicated(keep="last")]
    dd = pd.to_datetime(pd.Series(s.index))
    bt = pd.DataFrame({"base_date": (dd - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").values, "date": s.index, "stage": "15h",
                       "grade": T3.grade(s.values, th), "prob": T3.prob(s.values, cal).round(4), "score": s.values.round(4),
                       "label": yj.reindex(s.index).values})
    bt.to_csv(OUT_TAB / f"surge_{REGION}_15h.csv", index=False, encoding="utf-8-sig")

    sc = []
    m25 = r["idx"] >= "2025-01-01"
    parts = {"2024~2025 검증 (문턱·보정 구간)": (yv, sv), "2025 검증": (yv[m25], sv[m25])}
    x = bt[(bt["date"] >= TEST[0]) & (bt["date"] <= TEST[1])].dropna(subset=["label"])
    parts[f"2026 시험 ({TEST[0]} ~ {TEST[1]})"] = (x["label"].astype(int).values, x["score"].values)
    for nm, (y, s_) in parts.items():
        sc += score_rows(nm, np.asarray(y, int), np.asarray(s_, float), T3.prob(s_, cal), th["주의"], th["경보"])
    sc = pd.DataFrame(sc)
    sc.insert(0, "모델", pick)
    # 참고: 다른 후보의 2026 같은 날 (선택에 쓰지 않음)
    for name, rr in res.items():
        if name == pick:
            continue
        s2 = rr["st"].reindex(x["date"].values)
        if s2.notna().all() and x["label"].sum():
            sc = pd.concat([sc, pd.DataFrame([{"모델": f"{name} (참고)", "구간": parts and f"2026 시험 ({TEST[0]} ~ {TEST[1]})",
                                               "일수": len(x), "급등일": int(x["label"].sum()),
                                               "PR-AUC": average_precision_score(x["label"].astype(int), s2.values)}])], ignore_index=True)
    sc.round(4).to_csv(OUT_TAB / f"surge_score_{REGION}_15h.csv", index=False, encoding="utf-8-sig")
    choice = {"stage": "15h (전날 15시)", "계통": REGION, "모델": pick,
              "구성": "육지 급등 v3와 같음: 부호 고정 점수 + 스플라인 로지스틱 + LightGBM 그루터기(깊이 1) 확률 평균",
              "학습 자료": "육지 SMP·수요 (육지 v3 모델 그대로)" if r["cfg"] is None else "제주 SMP·제주 하루전 수요예측(jlfd), 날씨·주간전망은 육지 입력",
              "휴무일": "0 (경보 없음)" if (r["cfg"] or FA.CANDS["C A+B"]).get("offday") else "포함",
              "피처": r["feats"], "문턱": th, "확률식": {"형태": "1/(1+exp(-(a·score+b)))", **cal},
              "급등 정의": "제주 그날 최고 SMP ≥ 직전 28일 제주 일 최고 중앙값 × 1.25",
              "문턱 규칙": "2024~2025 매달 앞으로만 검증 점수(제주 레이블), 0.01 격자, 주의 = 재현율 80% 최고선, 경보 = 정밀도 50%·적중 2일 이상 최저선",
              "선택 절차": "후보 3개(S1·S2·S2b) 2024~2025 검증 PR-AUC 최대, 0.03 이내면 단순한 쪽 (surge_select_제주.csv)",
              "학습": {"검증": "2024-01 ~ 2025-12 매달: 2023-01-01 ~ 그 전달 말", "최종": "2023-01-01 ~ 2025-12-31"},
              "주의": "2025 제주 급등 7일뿐 — 2024 하반기 이후 제주 가격이 육지와 거의 같아져 급등 자체가 드묾. 문턱은 2024~2025 합쳐 정함"}
    (OUT_TAB / f"surge_choice_{REGION}_15h.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    print("선택:", pick, "문턱", th, "확률식", cal)
    print(sc[[c for c in ["모델", "구간", "일수", "급등일", "PR-AUC", "헛경보@재현율80%", "경보", "적중", "놓침", "헛경보", "재현율", "정밀도"]
              if c in sc.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
