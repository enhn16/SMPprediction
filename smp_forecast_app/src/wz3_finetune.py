"""일주일 v3 (R3) C1 — Chronos-2 small LoRA 미세조정 (육지 SMP + 달력 공변량, 예측 길이 168시간).

학습 자료: 2022-01-01 ~ 마감일(포함)의 시간별 육지 SMP 한 줄 + 휴무일·요일유형·명절(미래에도 아는 공변량).
  미세조정 함수가 이 긴 시계열에서 과거 2,048시간 + 앞 168시간 구간을 무작위로 잘라 학습한다.
판 (2단계, 잔차 학습 이력이 표본 밖이 되도록)
  ft2023: 마감 2023-12-31 → 대상일 2024 예측 (잔차 모델 학습 이력)
  ft2024: 마감 2024-12-31 → 대상일 2025 예측 (선택)
  ft2025: 마감 2025-12-31 → 대상일 2026 예측 (마지막 한 번 채점)
설정 (WEEK_V3.md ④-0, 속도 측정 후 한 번 정함): LoRA, 학습률 5e-5, 500단계, 배치 32 — 한 단계 약 6.5초, 판당 약 55분
저장: data/processed/smp_forecast_app/wz3/ft/{판}/finetuned-ckpt
실행: python -m smp_forecast_app.src.wz3_finetune ft2023 ft2024 ft2025  (⑥: ft2023l ft2024l ft2023f ft2024f)   (속도 측정: --bench)
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.calendar import build_calendar
from common.config import SEED
from common.loaders import load_smp_actual
from smp_forecast_app.src.wz3_chronos import CTX, MODELS, W3

CUTS = {"ft2023": "2023-12-31", "ft2024": "2024-12-31", "ft2025": "2025-12-31",
        # ⑦ 분기 마감 판 (wz3_ftq)
        "fq2403": "2024-03-31", "fq2406": "2024-06-30", "fq2409": "2024-09-30",
        "fq2503": "2025-03-31", "fq2506": "2025-06-30", "fq2509": "2025-09-30"}
LR = 5e-5
# ⑥ 재시도 (WEEK_V3.md ⑥): 판 이름 끝 l = LoRA 1e-5(문서 권장), f = 전체 1e-6(문서 기본값). 끝 글자 없으면 ④ C1 설정
VARIANTS = {"l": ("lora", 1e-5), "f": ("full", 1e-6), "": ("full", LR)}   # ④ C1은 peft 미설치로 실제 전체 미세조정이었음
STEPS = 500
BATCH = 32


def train_input(cut):
    s = load_smp_actual("육지").sort_values(["date", "hour"])
    s = s[(s["date"] >= "2022-01-01") & (s["date"] <= cut)]
    cal = build_calendar(s["date"].min(), s["date"].max()).set_index("date")
    cov = {"offday": cal["is_offday"].astype(float),
           "daytype": cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}).astype(float),
           "myeongjeol": cal["is_myeongjeol"].astype(float)}
    past = {k: s["date"].map(v).values.astype("float32") for k, v in cov.items()}
    return [{"target": s["smp"].values.astype("float32"), "past_covariates": past,
             "future_covariates": {k: None for k in cov}}]


def finetune(name, steps=STEPS):
    from chronos import BaseChronosPipeline
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pipe = BaseChronosPipeline.from_pretrained(MODELS["c2s"], device_map="cpu", torch_dtype=torch.float32)
    t0 = time.time()
    out = W3 / "ft" / name
    mode, lr = VARIANTS[name[6:]]
    pipe.fit(train_input(CUTS[name[:6]]), prediction_length=168, finetune_mode=mode, learning_rate=lr,
             num_steps=steps, batch_size=BATCH, context_length=CTX, output_dir=out, seed=SEED,
             logging_steps=max(1, steps // 10), save_strategy="no", report_to=[])
    print(f"{name}: {steps}단계 {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        CUTS["bench"] = CUTS["ft2023"]
        finetune("bench", steps=10)
    else:
        for n in sys.argv[1:]:
            finetune(n)
