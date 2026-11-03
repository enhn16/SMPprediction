"""일주일 v3 (R3) ⑥ K3 범위 채점 — 이미 채택·채점한 K3(점예측)에 P8과 같은 범위(A1)를 붙여 2026 포함률을 팀 성적표 지표로.
K3 구성·λ는 바꾸지 않는다 (같은 모델을 다른 지표로 한 번 더 보는 것, 고르기·조정 없음). 사용자 요청 2026-11-02.

출력: outputs/tables/WZ3_X4_K3_test2026_full.csv, data/processed/smp_forecast_app/wz3/WZ3_predictions_hourly_K3.csv
실행: python -m smp_forecast_app.src.wz3_k3range
"""
import json

import pandas as pd

from common.loaders import load_smp_actual
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.wz3_select import CHOICE, DATA, TAB, TEST, W3, final_predictions

K3 = ["mix|ftf|ext", "one", False]


def main():
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))   # 범위 설정은 P8 그대로, 점예측만 K3 (파일은 고치지 않음)
    ch["점예측"] = {"spec": K3}
    k3, info = final_predictions(ch)
    k3.assign(stage="15h").round(2).to_csv(W3 / "WZ3_predictions_hourly_K3.csv", index=False, encoding="utf-8-sig")
    print("K3 재맞춤:", info)
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    cur = range_adjust.apply(cur[cur["y"].notna()].reset_index(drop=True), "육지")
    p8 = pd.read_csv(W3 / "WZ3_predictions_hourly_P8.csv", parse_dates=["date"])[["date", "hour", "H", "pred", "lo", "hi"]]
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    lag7 = act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"})
    m = (cur.merge(k3, on=["date", "hour", "H"], suffixes=("_cur", "_k3"))
         .merge(p8.rename(columns={"pred": "pred_p8", "lo": "lo_p8", "hi": "hi_p8"}), on=["date", "hour", "H"]))
    m = m[m["date"].between(*TEST) & m["y_cur"].notna()].merge(lag7, on=["date", "hour"], how="left")
    m["y"], m["offday"] = m["y_cur"], m["offday_cur"].astype(bool)
    segs = hourly_segments(m)
    segs |= {"D+1": (m["H"] == 1).values, "D+2~7": (m["H"] >= 2).values}
    rows = []
    for seg, s in segs.items():
        g = m[s]
        for who, label in (("cur", "지금 모델"), ("p8", "v3 P8"), ("k3", "K3")):
            rows.append({"구분": seg, "모델": label,
                         **metrics(g["y"], g[f"pred_{who}"], g[f"lo_{who}"], g[f"hi_{who}"], g["lag7"])})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "WZ3_X4_K3_test2026_full.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(t[["구분", "모델", "표본수", "MAE", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
