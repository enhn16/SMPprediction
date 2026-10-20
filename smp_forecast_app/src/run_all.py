"""과제2 전체 재실행: 피처(지평 1~6) → 자체 모델·범위(B) → 주간 지평 2~6 → 담당자 모델 범위(A) → 웹 데이터 → 그래프.

새 실적·예보를 수집한 뒤 이 스크립트 하나로 웹앱 데이터를 갱신한다. (약 40분)
실행: python -m smp_forecast_app.src.run_all
"""
from smp_forecast_app.src import export_web, features, figures, interval_operator, model, model_weekly


def main():
    print("===== 피처 (지평 1~6) =====", flush=True)
    features.main(leads=range(1, 7))
    for name, mod in (("자체 모델·범위", model), ("주간 지평 2~6", model_weekly), ("담당자 모델 범위", interval_operator),
                      ("웹 데이터", export_web), ("그래프", figures)):
        print(f"\n===== {name} =====", flush=True)
        mod.main()


if __name__ == "__main__":
    main()
