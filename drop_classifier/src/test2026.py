"""발표용 최종 시험: 2024~2025 학습 → 2026 전체 시험 (육지, 하루 전 모델).

경보 문턱은 2026년을 보지 않고 정한다.
  1) 2024년으로 학습한 모델로 2025년을 예측 → 2025년 급락의 80%를 잡는 가장 엄격한 문턱
  2) 2024~2025년 전체로 다시 학습 → 2026-01-01 ~ 최신일 예측, 1)의 문턱을 그대로 적용
모델 점수는 '클수록 급락 위험'(회귀는 −예측 최저 SMP, 분류는 확률).

실행: python -m drop_classifier.src.test2026
저장: outputs/tables/08_test2026_summary_육지.csv, 08_test2026_daily_육지.csv
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common.config import ROOT
from drop_classifier.src.features import CALENDAR, DEMAND, FCST, LAG
from drop_classifier.src.final import threshold_for
from drop_classifier.src.models import split
from drop_classifier.src.search import CANDIDATES, DAY_SETS, day_table

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
TUNE = (("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"))
FINAL = (("2024-01-01", "2025-12-31"), ("2026-01-01", "2099-12-31"))
WITH_DEMAND = CALENDAR + DEMAND + LAG + FCST
VARIANTS = {
    "A": ("수요예측 안 넣음 · Ridge+EBM (최종)", DAY_SETS["기본+주간전망"], "Ridge+EBM 회귀 평균"),
    "A_ridge": ("수요예측 안 넣음 · Ridge", DAY_SETS["기본+주간전망"], "Ridge 회귀(α=3)"),
    "A_logit": ("수요예측 안 넣음 · 로지스틱", DAY_SETS["기본+주간전망"], "로지스틱(급락)"),
    "B_ens": ("수요예측 넣음 · Ridge+EBM", WITH_DEMAND, "Ridge+EBM 회귀 평균"),
    "B_ridge": ("수요예측 넣음 · Ridge", WITH_DEMAND, "Ridge 회귀(α=3)"),
    "B_logit": ("수요예측 넣음 · 로지스틱", WITH_DEMAND, "로지스틱(급락)"),
}


def metrics(y, a, s) -> dict:
    tp, fn = int(((a == 1) & (y == 1)).sum()), int(((a == 0) & (y == 1)).sum())
    fp, tn = int(((a == 1) & (y == 0)).sum()), int(((a == 0) & (y == 0)).sum())
    return {"일수": len(y), "급락일": tp + fn, "적중": tp, "놓침": fn, "헛경보": fp, "정상": tn,
            "재현율": tp / max(tp + fn, 1), "정밀도": tp / max(tp + fp, 1), "헛경보율": fp / max(fp + tn, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1),
            "ROC-AUC": roc_auc_score(y, s) if y.nunique() > 1 else np.nan,
            "PR-AUC": average_precision_score(y, s) if y.nunique() > 1 else np.nan}


def run(df, cols, model):
    fn = CANDIDATES[model]
    tr, te = split(df, TUNE[0]), split(df, TUNE[1])
    thr = threshold_for(te["label"].values, fn(tr, te, cols), 0.8)
    tr, te = split(df, FINAL[0]), split(df, FINAL[1]).copy()
    te["score"] = fn(tr, te, cols)
    te["alarm"] = (te["score"] >= thr).astype(int)
    return te, thr


def main():
    df = day_table()
    rows, web = [], {}
    for key, (name, cols, model) in VARIANTS.items():
        te, thr = run(df, cols, model)
        is_reg = "회귀" in model
        for part, g in (("2026 전체", te), ("1~6월", te[te["date"] < "2026-07-01"]), ("7월~", te[te["date"] >= "2026-07-01"]),
                        ("휴무일", te[te["day_type"] != "평일"]), ("평일", te[te["day_type"] == "평일"])):
            rows.append({"모델": name, "구간": part, "문턱": round(-thr, 1) if is_reg else round(thr, 3),
                         **metrics(g["label"], g["alarm"], g["score"])})
        web[key] = {"name": name, "threshold": round(-thr, 1) if is_reg else round(thr, 3), "is_reg": is_reg,
                    "m": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics(te["label"], te["alarm"], te["score"]).items()},
                    "days": [{"d": d.strftime("%m-%d"), "y": int(y), "smp": round(float(sm), 1), "dt": dt, "a": int(a),
                              "p": round(float(-s), 1) if is_reg else round(float(s), 3)}
                             for d, y, sm, dt, a, s in zip(te["date"], te["label"], te["min_smp_11_15"], te["day_type"], te["alarm"], te["score"])]}
        if key == "A":
            te.assign(pred=-te["score"])[["date", "day_type", "label", "min_smp_11_15", "pred", "alarm"]].to_csv(
                OUT_TAB / "08_test2026_daily_육지.csv", index=False, encoding="utf-8-sig")
        c = te[te["date"].isin(pd.to_datetime(["2026-09-25", "2026-10-05"]))]
        print(f"{name}: 문턱 {web[key]['threshold']}, 9/25 {'경보' if c['alarm'].iloc[0] else '없음'}, 10/5 {'경보' if c['alarm'].iloc[1] else '없음'}", flush=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT_TAB / "08_test2026_summary_육지.csv", index=False, encoding="utf-8-sig")
    (OUT_TAB / "08_test2026_web.json").write_text(json.dumps(web, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 260)
    print(res[res["구간"] == "2026 전체"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
