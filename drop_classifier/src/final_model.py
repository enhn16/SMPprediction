"""최종 하루 전 모델 (육지, 판단 시각 D-1 15:00) — 결과표·과제2 연계 파일·결과 페이지용 데이터 생성.

최종 구성 (2026-10-19 결정, 모두 2025년 성적으로만 선택)
  - 피처: 기본+주간전망+보강 + 태양광 설비용량 (supply_level.py에서 선택)
  - 모델: Ridge(α=3)+EBM 회귀 평균, 예측 = 내일 한낮(11~15시) 최저 SMP
  - 문턱: 주의 = 2024 학습→2025 예측에서 2025 급락 80%를 잡는 값
          강한 경보 = 같은 예측에서 정밀도 50%·적중 3일 이상인 가장 느슨한 값
  - 시험: 2024~2025 학습 → 2026-01-01 ~ 최신일 한 번 채점
출력 (outputs/tables)
  13_final_test2026_육지.csv   구간·경보 단계·기준선 규칙별 성적
  13_final_daily_육지.csv      2026 일별 예측·등급
  13_final_web.json            결과 페이지 데이터 (A: 최종 모델, R1·R2: 규칙)
  13_final_choice.json         문턱
  drop_prob_육지.csv           과제2·3 연계 (2025: 2024 학습 예측, 2026~: 2024~2025 학습 예측)
실행: python -m drop_classifier.src.final_model
"""
import glob
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.config import ROOT
from drop_classifier.src.fair_select import fcst15, metrics
from drop_classifier.src.final import threshold_for
from drop_classifier.src.fuel_level import ABS, BASE
from drop_classifier.src.models import split
from drop_classifier.src.search import day_table
from drop_classifier.src.supply_level import SOLAR, supply_features
from drop_classifier.src.two_tier import OUT_TAB, strong_threshold

warnings.filterwarnings("ignore")
REGION = "육지"
COLS = BASE + SOLAR
NAME = "최종 모델 (Ridge+EBM)"


def baseline_weekly_d1() -> pd.DataFrame:
    """현업 주간 예측(15시)의 D+1 한낮(11~15시) 최저 예측값 (data/smp_edu_data, 읽기 전용)."""
    rows = []
    for f in glob.glob(str(ROOT / "data/smp_edu_data/outputs/predict/SMP주간예측_*.csv")):
        d = pd.read_csv(f, parse_dates=["기준일", "대상일"])
        d = d[(d["지평"] == "D+1") & d["거래시간"].between(11, 15)]
        rows.append(d.groupby("대상일")["예측"].min())
    b = pd.concat(rows).groupby(level=0).last()
    return b.rename("base").rename_axis("date").reset_index()


def main():
    df = day_table().merge(fcst15(), on="date", how="left")
    df = pd.concat([df.reset_index(drop=True), supply_features(df["date"])], axis=1)

    # 문턱 (2025 검증 예측)
    tr, va = split(df, ("2024-01-01", "2024-12-31")), split(df, ("2025-01-01", "2025-12-31")).copy()
    va["score"] = ABS(tr, va, COLS)
    yv = va["label"].values
    warn, strong = threshold_for(yv, va["score"].values, 0.8), strong_threshold(yv, va["score"].values)
    w_won, s_won = round(-warn, 1), round(-strong, 1)
    print(f"문턱: 주의 {w_won}원, 강한 경보 {s_won}원", flush=True)

    # 2026 시험
    tr2, te = split(df, ("2024-01-01", "2025-12-31")), split(df, ("2026-01-01", "2099-12-31")).copy()
    te["score"] = ABS(tr2, te, COLS)
    te["pred"] = -te["score"]
    te["alarm"] = (te["score"] >= warn).astype(int)
    te["strong"] = (te["score"] >= strong).astype(int)
    te["등급"] = np.where(te["strong"] == 1, "강한 경보", np.where(te["alarm"] == 1, "주의", "없음"))

    off = te["day_type"] != "평일"
    parts = {"2026 전체": te, "1~6월": te[te["date"] < "2026-07-01"], "7월~": te[te["date"] >= "2026-07-01"],
             "휴무일": te[off], "평일": te[~off]}
    rows = [{"방법": NAME, "경보": "주의 이상", "구간": k, **metrics(g["label"], g["alarm"], g["score"])}
            for k, g in parts.items()]
    rows += [{"방법": NAME, "경보": "강한 경보", "구간": k, **metrics(g["label"], g["strong"], g["score"])}
             for k, g in parts.items()]
    mid = te[(te["alarm"] == 1) & (te["strong"] == 0)]
    rows.append({"방법": NAME, "경보": "주의만", "구간": "2026 전체", "일수": len(mid),
                 "적중": int(mid["label"].sum()), "헛경보": int((mid["label"] == 0).sum())})
    rules = {"규칙: 휴무일이면 급락": off.astype(int),
             "규칙: 휴무일 + 맑음 예보": (off & (te["rf_clear_pm"] >= 0.5)).astype(int)}
    rows += [{"방법": k, "경보": "-", "구간": "2026 전체", **metrics(te["label"], v, v)} for k, v in rules.items()]
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT_TAB / f"13_final_test2026_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 현업 주간 예측 D+1과 비교
    m = te.merge(baseline_weekly_d1(), on="date")
    y = m["label"]
    cmp = {"일수": len(m), "급락": int(y.sum()), "기간": f"{m.date.min().date()} ~ {m.date.max().date()}"}
    for k, s in (("현업", -m["base"]), ("모델", m["score"])):
        thr = s[y == 1].min()
        cmp[k] = {"ROC": round(roc_auc_score(y, s), 3), "모두잡을때헛경보": int(((s >= thr) & (y == 0)).sum())}
    cmp["현업_30원미만판정"] = int(((m["base"] < 30) & (y == 1)).sum())
    cmp["모델_주의"] = {"적중": int(((m.alarm == 1) & (y == 1)).sum()), "헛경보": int(((m.alarm == 1) & (y == 0)).sum())}
    cmp["모델_강한"] = {"적중": int(((m.strong == 1) & (y == 1)).sum()), "헛경보": int(((m.strong == 1) & (y == 0)).sum())}

    te[["date", "day_type", "label", "min_smp_11_15", "pred", "alarm", "strong", "등급"]].round(2).to_csv(
        OUT_TAB / f"13_final_daily_{REGION}.csv", index=False, encoding="utf-8-sig")
    info = {"피처": "기본+주간전망+보강+태양광설비", "모델": "Ridge+EBM 회귀 평균", "주의 문턱": w_won,
            "강한 경보 문턱": s_won, "현업 비교": cmp}
    (OUT_TAB / "13_final_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    # 결과 페이지 데이터
    def days(alarm, pred):
        return [{"d": r.date.strftime("%m-%d"), "y": int(r.label), "smp": round(float(r.min_smp_11_15), 1),
                 "dt": r.day_type if r.day_type in ("평일", "주말") else "공휴일", "a": int(a), "p": round(float(p), 1)}
                for r, a, p in zip(te.itertuples(), alarm, pred)]
    web = {"A": {"name": NAME, "threshold": w_won, "strong": s_won, "is_reg": True,
                 "m": metrics(te["label"], te["alarm"], te["score"]), "days": days(te["alarm"], te["pred"]),
                 "strong_days": te["strong"].tolist()}}
    for k, (name, v) in zip(("R1", "R2"), rules.items()):
        web[k] = {"name": name, "threshold": None, "is_reg": False, "m": metrics(te["label"], v, v),
                  "days": days(v, np.zeros(len(v)))}
    for v in web.values():
        v["m"] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for kk, vv in v["m"].items()}
    (OUT_TAB / "13_final_web.json").write_text(json.dumps(web, ensure_ascii=False), encoding="utf-8")

    # 과제2·3 연계 파일
    both = pd.concat([va[["date", "score"]], te[["date", "score"]]])
    pred = -both["score"]
    pd.DataFrame({"date": both["date"].dt.strftime("%Y-%m-%d"), "계통": REGION,
                  "예측_한낮최저SMP": pred.round(1), "위험점수": both["score"].round(3),
                  "등급": np.where(pred <= s_won, "강한 경보", np.where(pred <= w_won, "주의", "없음")),
                  "판정": (pred <= w_won).astype(int)}).to_csv(
        OUT_TAB / f"drop_prob_{REGION}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(res[["방법", "경보", "구간", "일수", "급락일", "적중", "놓침", "헛경보", "재현율", "정밀도", "헛경보율", "ROC-AUC"]]
          .round(3).to_string(index=False))
    print(json.dumps(cmp, ensure_ascii=False))
    print("놓침:", te[(te.label == 1) & (te.alarm == 0)][["date", "day_type", "pred"]].round(1).to_string(index=False))
    print("주목일:", te[te.date.isin(pd.to_datetime(["2026-09-25", "2026-10-05"]))][["date", "pred", "등급"]].round(1)
          .to_string(index=False))
    print("예측 중앙값 급락/비급락:", te.groupby("label")["pred"].median().round(1).to_dict())


if __name__ == "__main__":
    main()
