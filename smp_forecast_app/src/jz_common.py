"""제주 일주일 예측 비교(jz_*) 공용 — 기준일 I 15시 판단, 대상일 D = I+H (H=1~7), 시간별. 육지 wz_common과 같은 규칙.

예측 저장: data/processed/smp_forecast_app/jeju/jzoo/{모델}.csv.gz [date, hour, H, pred(, qlo, qhi)]
평가: 2025 상반기 맞춤 → 2025 하반기 비교 → 2025 전체 재적합 → 2026 한 번 채점 (사용자 확인 후)
재학습: 빠른 모델 매월 1일, 느린 모델 분기 첫날. 실적은 (재학습일 − H)일까지만.
학습 시작: 2023-01-01 (육지 Chronos 과거 예측이 2023부터라 육지 예측을 피처로 쓰는 모델과 맞춤)
"""
import pandas as pd

from common.config import PROCESSED_DIR
from smp_forecast_app.src.jz_features import FEATURES, JDIR, load as load_feats

OUT = JDIR / "jzoo"
HIST = JDIR / "jzoo_hist"
LAND = PROCESSED_DIR / "smp_forecast_app"
HS = range(1, 8)
START_EVAL = "2025-01-01"
TRAIN_START = "2023-01-01"
LAND_LAMBDA = 0.5   # 육지 최종 = Chronos-2 small + 0.5 × 잔차 보정(외부만) (WZ_choice.json)


def periods(df, freq):
    return pd.date_range(START_EVAL, df["date"].max(), freq=freq).union(pd.DatetimeIndex([pd.Timestamp(START_EVAL)]))


def rolling_fit(df, H, freq, fit_predict, train_start=TRAIN_START):
    """재학습 시점마다 학습(실적은 시점−H일까지) → 다음 시점 전까지 예측."""
    out = []
    ps = list(periods(df, freq)) + [pd.Timestamp("2100-01-01")]
    for a, b in zip(ps[:-1], ps[1:]):
        tr = df[(df["date"] >= train_start) & (df["date"] < a - pd.Timedelta(days=H - 1)) & df["y"].notna()]
        te = df[(df["date"] >= a) & (df["date"] < b)]
        if te.empty:
            continue
        r = fit_predict(tr, te)
        o = te[["date", "hour"]].copy()
        if isinstance(r, tuple):
            o["pred"], o["qlo"], o["qhi"] = r
        else:
            o["pred"] = r
        out.append(o)
    return pd.concat(out).assign(H=H)


def save(name, df, hist=False):
    d = HIST if hist else OUT
    d.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(d / f"{name}.csv.gz", index=False, encoding="utf-8")


def load_pred(name, hist=False):
    return pd.read_csv((HIST if hist else OUT) / f"{name}.csv.gz", parse_dates=["date"])


def land_pred() -> pd.DataFrame:
    """육지 일주일 예측 [date, hour, H, L_c2s, L_final, L_base] (대상일 2023~).
    L_c2s: 육지 Chronos-2 small (2023~2024 과거 예측 + 2025~ 백테스트), L_final: 육지 최종(2025~),
    L_base: L_final, 없으면 L_c2s (2023~2024 학습용). 백테스트 이후는 육지 운영 기록으로 채움."""
    c = pd.concat([pd.read_csv(LAND / "wzoo_hist" / "c2s.csv.gz", parse_dates=["date"]),
                   pd.read_csv(LAND / "wzoo" / "c2s.csv.gz", parse_dates=["date"])], ignore_index=True)
    c = c.drop_duplicates(["date", "hour", "H"], keep="last").rename(columns={"pred": "L_c2s", "qlo": "L_qlo", "qhi": "L_qhi"})
    r = pd.read_csv(LAND / "wzoo_resid" / "외부만.csv.gz", parse_dates=["date"])
    d = c.merge(r, on=["date", "hour", "H"], how="left")
    d["L_final"] = d["L_c2s"] + LAND_LAMBDA * d["rhat"]
    d["L_base"] = d["L_final"].fillna(d["L_c2s"])
    d = d[["date", "hour", "H", "L_c2s", "L_qlo", "L_qhi", "L_final", "L_base"]]
    # 백테스트 이후 날짜: 육지 운영 기록(ops/hourly_log.csv, 읽기만) — 백테스트에 있는 (날짜, H)는 백테스트 값 유지
    ol = LAND / "ops" / "hourly_log.csv"
    if ol.exists():
        o = pd.read_csv(ol, parse_dates=["date"])
        o = pd.DataFrame({"date": o["date"], "hour": o["hour"], "H": o["H"], "L_c2s": o["c2s"], "L_qlo": o["c2s_qlo"],
                          "L_qhi": o["c2s_qhi"], "L_final": o["pred"], "L_base": o["pred"]})
        key = pd.MultiIndex.from_frame(d[["date", "hour", "H"]])
        o = o[~pd.MultiIndex.from_frame(o[["date", "hour", "H"]]).isin(key)]
        d = pd.concat([d, o], ignore_index=True)
    return d


def feats(H, with_land=True):
    """제주 피처 표 + 육지 예측(같은 H)."""
    f = load_feats(H)
    if with_land:
        lp = land_pred()
        f = f.merge(lp[lp["H"] == H].drop(columns="H"), on=["date", "hour"], how="left")
    return f


def actual_table():
    df = load_feats(1)[["date", "hour", "y", "y_land", "offday", "s_d1"]]
    return df[df["date"] >= START_EVAL]
