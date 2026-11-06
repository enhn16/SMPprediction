"""지금 운영 제주 급락 피처(S5, 63개)를 2024-01 ~ 2026-10-31로 뽑는다 — 운영 코드는 import만, 운영 파일은 쓰지 않음.

저장된 운영 피처 표(features_hourly_제주_h1.csv)는 10-29까지라, jz_features.build(1)을 메모리에서 새로 계산해 넘긴다.
5분 실적 피처(js_*)는 운영 로더 기준이라 운영 폴더에 없는 날(2026-10-29·30)은 비어 있다(모델에서 학습 중앙값으로 채움).
저장: data/processed/extended_backtest/jeju_s5_features.csv
실행(루트에서): python -m extended_backtest.src.jeju_s5
"""
import drop_classifier.src.jeju_drop as jd
import drop_classifier.src.jeju_zoo as jz
from smp_forecast_app.src import jz_features
from extended_backtest.src.jeju_models import S5F


def main():
    f = jz_features.build(1)
    d = jd.day_table(f=f, start="2024-01-01")
    cols = jz.SETS["S5 +계통실적"]
    d = d[d["date"] <= "2026-10-31"]
    d[["date"] + cols].to_csv(S5F, index=False)
    print(d.shape, d["date"].min().date(), d["date"].max().date())
    print("마지막 3일 결측 수:", d.tail(3).set_index("date")[cols].isna().sum(axis=1).to_dict())


if __name__ == "__main__":
    main()
