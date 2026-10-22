"""일주일 예측 — 모델 비교 + 앙상블 + 80% 범위 선택 + 2026 채점.

공정한 선택 절차 (2026 채점 전 고정)
  1) 개별 모델: 2025 상반기(H1)·하반기(H2)·전체 MAE
  2) 앙상블 후보 ~20종: 순위·가중치는 2025 상반기로 정하고, 2025 하반기 MAE로 앙상블끼리 비교 → 1위 유형 선택
     (0.05원 이내 동률이면 구성 모델 수가 적은 쪽)
  3) 고른 유형을 2025 전체로 다시 맞춤(순위·가중치) → 2026 한 번 채점 (일주일은 1차 채점 이후라 '2차 시험' 표기)
  4) 범위: 고른 앙상블에 오차 분위수형·보정 분위수 회귀(LightGBM 분위수, Chronos-2 분위수) 적용,
     2025 하반기 성적으로 지평별 선택 (intervals.select), 오차 이력은 대상일 ≤ 기준일만 (gap=H)
출력: outputs/tables/WZ1_models.csv, WZ2_ensembles_2025H2.csv, WZ3_test2026.csv, WZ4_intervals.csv, WZ_choice.json,
      data/processed/smp_forecast_app/WZ_predictions_hourly.csv [base_date, H, date, hour, y, pred, lo, hi]
실행: python -m smp_forecast_app.src.wz_ens
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.intervals import add_keys, methods, report, score, select
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src.wz_common import HS, OUT, actual_table

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
NAMES = {"naive7": "기준: 지난주 같은 시간", "same": "기준: 최근 같은 유형일", "lgb": "LightGBM", "xgb": "XGBoost",
         "ridge": "Ridge", "cat": "CatBoost", "rf": "랜덤포레스트", "et": "엑스트라트리", "ebm": "EBM", "mlp": "MLP",
         "tabpfn": "TabPFN", "lstm": "LSTM", "gru": "GRU", "cnn": "1D-CNN", "transformer": "Transformer",
         "c2": "Chronos-2", "c2s": "Chronos-2 small", "bolt": "Chronos-Bolt small"}
FAMILY = {"부스팅": ["lgb", "xgb", "cat"], "트리": ["rf", "et"], "선형·GAM": ["ridge", "ebm"], "신경망(표)": ["mlp", "tabpfn"],
          "시계열 신경망": ["lstm", "gru", "cnn", "transformer"], "Chronos": ["c2", "c2s", "bolt"]}
BASELINES = ["naive7", "same"]


def load_all():
    act = actual_table()
    frames = {}
    for p in sorted(OUT.glob("*.csv.gz")):
        name = p.name.replace(".csv.gz", "")
        if name == "lgbq":
            continue
        frames[name] = pd.read_csv(p, parse_dates=["date"]).set_index(["date", "hour", "H"])["pred"]
    P = pd.DataFrame(frames)
    P = P.reset_index().merge(act, on=["date", "hour"], how="left")
    from common.calendar import build_calendar   # 미래 날짜의 휴무일은 달력에서 (사전 확정)
    cal = build_calendar(P["date"].min(), P["date"].max()).set_index("date")["is_offday"]
    P["offday"] = P["date"].map(cal).astype(int)
    P["base_date"] = P["date"] - pd.to_timedelta(P["H"], unit="D")
    return P


def mae(y, p):
    ok = y.notna() & p.notna()
    return float((y[ok] - p[ok]).abs().mean())


def ens_candidates(P, fit_mask, models, family=None, hs=HS):
    """fit_mask 구간으로 순위·가중치를 정한 앙상블 후보들 → {이름: (예측 Series, 구성 설명)}."""
    family = family or FAMILY   # 한 달 예측(mz_ens)에서 계열·지평 묶음을 바꿔 씀
    F = P[fit_mask]
    rank = sorted(models, key=lambda m: mae(F["y"], F[m]))
    out = {}

    def avg(ms, w=None):
        w = np.ones(len(ms)) / len(ms) if w is None else np.asarray(w) / np.sum(w)
        return (P[ms] * w).sum(axis=1, min_count=len(ms)), ms
    for k in (3, 5, 7):
        out[f"상위 {k}개 평균"] = avg(rank[:k])
    out["전부 평균"] = avg(models)
    out["상위 5개 중앙값"] = (P[rank[:5]].median(axis=1), rank[:5])
    for fam, ms in family.items():
        ms = [m for m in ms if m in models]
        if len(ms) >= 2:
            out[f"{fam} 평균"] = avg(ms)
    best_fam = [min([m for m in ms if m in models], key=lambda m: mae(F["y"], F[m])) for ms in family.values()
                if any(m in models for m in ms)]
    out["계열별 1위 평균"] = avg(best_fam)
    boost = [m for m in family.get("부스팅", []) if m in models]
    if "c2" in models and boost:
        b, _ = avg(boost)
        out["부스팅 70% + Chronos-2 30% (현업 방식)"] = (0.7 * b + 0.3 * P["c2"], boost + ["c2"])
    inv = np.array([1 / mae(F["y"], F[m]) for m in rank[:5]])
    out["상위 5개 오차역수 가중"] = avg(rank[:5], inv)
    ok = F[models + ["y"]].dropna()
    w, _ = nnls(ok[models].values, ok["y"].values)
    sel = [m for m, x in zip(models, w) if x > 1e-3]
    out["가중 평균(NNLS)"] = avg(sel, [x for x in w if x > 1e-3])
    # 지평별 NNLS / 지평별 최고 / 지평별 스태킹(Ridge)
    ph, pb, ps, used = pd.Series(np.nan, index=P.index), pd.Series(np.nan, index=P.index), pd.Series(np.nan, index=P.index), set()
    for H in hs:
        fh, mh = F[F["H"] == H], P["H"] == H
        okh = fh[models + ["y"]].dropna()
        wh, _ = nnls(okh[models].values, okh["y"].values)
        wh = wh / wh.sum()
        ph[mh] = (P.loc[mh, models] * wh).sum(axis=1, min_count=len(models))
        bm = min(models, key=lambda m: mae(fh["y"], fh[m]))
        pb[mh] = P.loc[mh, bm]
        used.add(bm)
        rg = Ridge(alpha=1.0).fit(okh[models].values, okh["y"].values)
        X = P.loc[mh, models].dropna()
        ps[X.index] = rg.predict(X.values)
    out["지평별 가중 평균(NNLS)"] = (ph, models)
    out["지평별 최고 모델"] = (pb, sorted(used))
    out["지평별 스태킹(Ridge)"] = (ps, models)
    # 탐욕적 전진 선택 (Caruana, 중복 허용 25회)
    chosen, cur = [], None
    okF = F[models + ["y"]].dropna()
    for _ in range(25):
        best_m, best_v = None, 1e9
        for m in models:
            trial = okF[chosen + [m]].mean(axis=1)
            v = (okF["y"] - trial).abs().mean()
            if v < best_v:
                best_m, best_v = m, v
        chosen.append(best_m)
    cnt = pd.Series(chosen).value_counts()
    out["탐욕적 전진 선택(Caruana)"] = avg(list(cnt.index), cnt.values)
    return out, rank


def main():
    P = load_all()
    models = [m for m in NAMES if m in P.columns and m not in BASELINES]
    y25 = P["date"].dt.year == 2025
    h1 = y25 & (P["date"] < "2025-07-01")
    h2 = y25 & (P["date"] >= "2025-07-01")
    t26 = (P["date"] >= "2026-01-01") & P["y"].notna()
    # 모든 모델이 예측한 행만 비교 (공정)
    full = P[models + BASELINES].notna().all(axis=1)
    rows = []
    for m in BASELINES + models:
        r = {"코드": m, "모델": NAMES[m], "2025 상반기": mae(P.loc[h1 & full, "y"], P.loc[h1 & full, m]),
             "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, m]), "2025 전체": mae(P.loc[y25 & full, "y"], P.loc[y25 & full, m])}
        for H in HS:
            mm = y25 & full & (P["H"] == H)
            r[f"2025 D+{H}"] = mae(P.loc[mm, "y"], P.loc[mm, m])
        rows.append(r)
    mt = pd.DataFrame(rows).sort_values("2025 전체")
    mt.round(3).to_csv(TAB / "WZ1_models.csv", index=False, encoding="utf-8-sig")
    print(mt[["모델", "2025 상반기", "2025 하반기", "2025 전체", "2025 D+1", "2025 D+7"]].round(2).to_string(index=False), flush=True)

    # 앙상블 선택: 상반기로 맞추고 하반기로 비교
    cands, rank = ens_candidates(P, h1 & full, models)
    erows = [{"앙상블": "단일 최고 (" + NAMES[rank[0]] + ")", "구성": NAMES[rank[0]], "구성 수": 1,
              "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, rank[0]])}]
    for name, (p, ms) in cands.items():
        erows.append({"앙상블": name, "구성": ", ".join(NAMES[m] for m in ms), "구성 수": len(ms),
                      "2025 하반기": mae(P.loc[h2 & full, "y"], p[h2 & full])})
    et = pd.DataFrame(erows).sort_values("2025 하반기")
    et.round(3).to_csv(TAB / "WZ2_ensembles_2025H2.csv", index=False, encoding="utf-8-sig")
    print(et.round(3).to_string(index=False), flush=True)
    top = et.iloc[0]["2025 하반기"]
    pick = et[et["2025 하반기"] <= top + 0.05].sort_values(["구성 수", "2025 하반기"]).iloc[0]["앙상블"]
    print(f"\n선택: {pick}\n", flush=True)

    # 2025 전체로 다시 맞춤 → 최종 예측
    cands_full, rank_full = ens_candidates(P, y25 & full, models)
    if pick.startswith("단일 최고"):
        P["pred"], members = P[rank_full[0]], [rank_full[0]]
    else:
        P["pred"], members = cands_full[pick][0], cands_full[pick][1]

    # 범위 (지평별): 오차 이력 = 2025 이후 이 앙상블 예측 (상반기 오차로 시작)
    q_lgb = pd.read_csv(OUT / "lgbq.csv.gz", parse_dates=["date"]) if (OUT / "lgbq.csv.gz").exists() else None
    q_c2 = pd.read_csv(OUT / "c2.csv.gz", parse_dates=["date"]) if (OUT / "c2.csv.gz").exists() else None
    irows, outs, ichoice = [], [], {}
    for H in HS:
        d = P[(P["H"] == H) & P["pred"].notna()].copy()
        d["offday"] = d["offday"].astype(bool)
        # 실적 있는 행 + 아직 실적 없는 미래 행(웹 '앞으로 7일'용). 미래 행은 오차 이력에 들어가지 않음
        allk = add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))
        allk = allk.assign(err=allk["err"].fillna(0), rerr=allk["rerr"].fillna(0))
        for qname, qdf in (("lgb", q_lgb), ("c2", q_c2)):
            if qdf is not None:
                qq = qdf[qdf["H"] == H][["date", "hour", "qlo", "qhi"]].rename(columns={"qlo": f"qlo_{qname}", "qhi": f"qhi_{qname}"})
                allk = allk.merge(qq, on=["date", "hour"], how="left")
        has_y = allk["y"].notna().values
        ev = has_y & (allk["date"] >= "2025-07-01").values
        cand = {}
        for k, f in methods(gap=H).items():
            if "최근1년" in k or "365일" in k:
                continue   # 이력이 2025-01부터라 1년 창은 불가
            cand["오차분위수·" + k] = f
        for qname, label in (("lgb", "LightGBM 분위수"), ("c2", "Chronos-2 분위수")):
            if f"qlo_{qname}" not in allk:
                continue
            kq = allk.assign(qlo=allk[f"qlo_{qname}"], qhi=allk[f"qhi_{qname}"])
            cand[f"보정분위수회귀({label})·갱신90일·시간대묶음×휴무일"] = (lambda kq: lambda dd, e: cqr(kq, e, window=90, key="k_band_off", gap=H))(kq)
            cand[f"{label} 그대로"] = (lambda kq: lambda dd, e: (kq.loc[e, "qlo"].clip(lower=0).values, kq.loc[e, "qhi"].values))(kq)
        part = []
        for name, fn in cand.items():
            try:
                lo, hi = fn(allk, ev)
            except Exception as ex:  # noqa: BLE001
                print(f"  범위 {name} 실패: {ex}")
                continue
            r = report(allk[ev].assign(lo=lo, hi=hi), name, "선택 2025 하반기")
            for x in r:
                x["H"] = H
            part += r
        sel = pd.DataFrame(part)
        ichoice[H] = select(sel)
        irows += part
        allmask = (allk["date"] >= "2025-07-01").values
        lo, hi = cand[ichoice[H]](allk, allmask)
        outs.append(allk[allmask].assign(lo=np.clip(lo, 0, None), hi=hi, H=H))
        r = sel[(sel["방법"] == ichoice[H]) & (sel["구간"] == "전체")].iloc[0]
        print(f"범위 D+{H}: {ichoice[H]} — 2025 하반기 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}", flush=True)
    pd.DataFrame(irows).round(4).to_csv(TAB / "WZ4_intervals.csv", index=False, encoding="utf-8-sig")
    W = pd.concat(outs)
    W["base_date"] = W["date"] - pd.to_timedelta(W["H"], unit="D")

    # 2026 채점 (한 번)
    te = W[(W["date"] >= "2026-01-01") & W["y"].notna()]
    trows = []
    for H in ["전체"] + list(HS):
        g = te if H == "전체" else te[te["H"] == H]
        r = {"H": H, **score(g["y"], g["lo"], g["hi"], g["pred"])}
        base = P[(P["date"] >= "2026-01-01") & P["y"].notna() & ((P["H"] == H) if H != "전체" else True)]
        r["LightGBM 단독 MAE"] = mae(base["y"], base["lgb"]) if "lgb" in base else np.nan
        r["단일 최고(2025) MAE"] = mae(base["y"], base[rank_full[0]])
        r["기준(지난주) MAE"] = mae(base["y"], base["naive7"])
        trows.append(r)
    test = pd.DataFrame(trows)
    test.round(4).to_csv(TAB / "WZ3_test2026.csv", index=False, encoding="utf-8-sig")
    keep = ["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]
    W[keep].round(2).to_csv(PROCESSED_DIR / "smp_forecast_app" / "WZ_predictions_hourly.csv", index=False, encoding="utf-8-sig")
    info = {"선택 앙상블": pick, "구성": [NAMES[m] for m in members], "단일 최고(2025)": NAMES[rank_full[0]],
            "범위": {str(k): v for k, v in ichoice.items()}, "선택 절차": "상반기로 맞춤 → 2025 하반기 비교 → 2025 전체로 재적합",
            "시험": "2026-01-01 ~ (일주일 예측 2차 시험)"}
    (TAB / "WZ_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(test.round(3).to_string(index=False))
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
