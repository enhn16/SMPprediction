"""과제2 전체 재실행 (약 1시간 30분). 새 실적·예보를 수집한 뒤 실행하면 웹앱 데이터가 갱신된다.

순서: 피처(시간별 지평 1~6, 일별) → 내일 1차(B)·2차(V) → 주간 지평 2~6(W) → 한 달 1차(M)·2차(M2v)
      → 담당자 모델 범위(A, 문서용) → 웹 데이터 → 그래프
최신 공표 SMP 수집: python -m common.collect.smp_forecast --recent 5
실행: python -m smp_forecast_app.src.run_all
"""
from smp_forecast_app.src import (drop_badge, export_web, features, features_daily, figures, interval_operator, model, model_monthly,
                                  model_monthly_v2, model_v2, model_weekly)


def main():
    print("===== 피처 (시간별 지평 1~6) =====", flush=True)
    features.main(leads=range(1, 7))
    for name, mod in (("피처 (일별)", features_daily), ("내일 1차", model), ("내일 2차", model_v2),
                      ("주간 지평 2~6", model_weekly), ("한 달 1차", model_monthly), ("한 달 2차", model_monthly_v2),
                      ("담당자 모델 범위", interval_operator), ("급락 배지 확률", drop_badge), ("웹 데이터", export_web), ("그래프", figures)):
        print(f"\n===== {name} =====", flush=True)
        mod.main()


if __name__ == "__main__":
    main()
