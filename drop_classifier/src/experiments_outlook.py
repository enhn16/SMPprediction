"""주간 전력수요전망 피처 효과 시험 (육지).

공개 시점 가정 두 가지
- 보수적: 등록 다음 날부터 사용 (판단일 전날까지 등록분)
- 당일: 등록 당일 판단 시각 전에 올라왔다고 가정 (판단일까지 등록분)
하루 전(판단일 = D-1), 일주일(판단일 = I) 각각 기존 선택 피처에 OUTLOOK을 더해 롤링 평가로 비교.

실행: python -m drop_classifier.src.experiments_outlook
"""
import warnings

import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from drop_classifier.src import outlook_features as of
from drop_classifier.src.features_weekly import W_CALENDAR, W_MID, W_STATE
from drop_classifier.src.search import CANDIDATES, DAY_SETS, day_table, evaluate

warnings.filterwarnings("ignore")
REGION = "육지"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
MODELS = ["Ridge 회귀(α=3)", "Ridge 회귀(α=10)", "Ridge+EBM 회귀 평균", "로지스틱(급락)"]


def attach(df: pd.DataFrame, decision: pd.Series, same_day: bool) -> pd.DataFrame:
    dec = decision + pd.Timedelta(days=1) if same_day else decision
    return pd.concat([df, of.attach(df["date"], dec)], axis=1)


def main():
    rows = []
    day = day_table()
    for rule, same in (("보수적", False), ("당일", True)):
        d = attach(day, day["date"] - pd.Timedelta(days=1), same)
        cov = d[of.OUTLOOK[0]].notna().mean()
        for m in MODELS:
            for name, cols in (("기본", DAY_SETS["기본"]), ("기본+주간전망", DAY_SETS["기본"] + of.OUTLOOK)):
                if rule == "당일" and name == "기본":
                    continue
                r = evaluate(d, cols, CANDIDATES[m])
                rows.append({"모델 구분": "하루 전", "공개 가정": rule, "전망 확보율": round(cov, 3), "피처": name, "모델": m, **r})
                print(f"[하루 전·{rule}] {name} / {m}: 헛경보@80% {r['헛경보@80%']}, @90% {r['헛경보@90%']}", flush=True)

    w = pd.read_csv(PROCESSED_DIR / "drop_classifier" / f"features_weekly_{REGION}.csv", parse_dates=["date", "issue_date"])
    w["label"] = w["label"].astype(int)
    for rule, same in (("보수적", False), ("당일", True)):
        wa = attach(w, w["issue_date"], same)
        for h in range(2, 8):
            d = wa[wa["lead"] == h].reset_index(drop=True)
            base = W_CALENDAR + W_STATE + (W_MID if h >= 5 else [])
            cov = d[of.OUTLOOK[0]].notna().mean()
            for m in MODELS[:2]:
                for name, cols in (("기존", base), ("기존+주간전망", base + of.OUTLOOK)):
                    if rule == "당일" and name == "기존":
                        continue
                    r = evaluate(d, cols, CANDIDATES[m])
                    rows.append({"모델 구분": f"{h}일 전", "공개 가정": rule, "전망 확보율": round(cov, 3), "피처": name, "모델": m, **r})
            print(f"[{h}일 전·{rule}] 완료 (전망 확보율 {cov:.2f})", flush=True)

    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT_TAB / f"06_outlook_experiments_{REGION}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    cols = ["모델 구분", "공개 가정", "전망 확보율", "피처", "모델", "헛경보@70%", "헛경보@80%", "헛경보@90%",
            "헛경보@90% 롤링1", "헛경보@90% 롤링2", "PR-AUC 평균"]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
