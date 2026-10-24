"""제주 일주일 예측 — 모델 비교 + 잔차 보정 + 앙상블 + 80% 범위 선택 (+ 사용자 확인 후 2026 한 번 채점).

공정한 선택 절차 (육지 wz_ens와 같음, 2026 채점 전 고정)
  1) 개별 모델: 2025 상반기(H1)·하반기(H2)·전체 MAE (모든 모델이 예측한 행만)
  2) 잔차 보정 후보 '기준 + λ × 잔차 예측' (jz_resid: 기준 3 × 피처 묶음 2 × 목적 2, λ ∈ {0.1, 0.25, 0.5, 0.75, 1}를 H1로 선택)
     → 개별 모델과 함께 앙상블 재료(pool)로 씀
  3) 앙상블 후보(wz_ens.ens_candidates ~20종)의 순위·가중치를 H1로 정하고 H2 MAE로 비교 → 1위
     (0.05원 이내 동률이면 구성 수가 적은 쪽; 잔차 보정 후보는 기준 + 보정 = 2개로 셈)
  4) 고른 유형을 2025 전체로 다시 맞춤(λ·순위·가중치) → 2025-07 ~ 최신 예측
  5) 범위: 지평별로 오차분위수형(하한 0원 자르기 없음)·보정 분위수 회귀(LightGBM·Chronos 분위수)·분위수 그대로를
     2025 하반기 성적으로 선택 (intervals.select: 포함률 75~85% 중 구간점수 최소)
  6) --score2026: 2026-01-01 ~ 최신 한 번 채점 (제주 1차 시험) — 사용자 확인 후에만 실행
출력: outputs/tables/JWZ1_models.csv, JWZ2_ensembles_2025H2.csv, JWZ4_intervals.csv, JWZ_choice.json,
      data/processed/smp_forecast_app/jeju/JWZ_predictions_hourly.csv [base_date, H, date, hour, y, pred, lo, hi, offday]
      (--score2026) JWZ3_test2026.csv
실행: python -m smp_forecast_app.src.jz_ens [--score2026]
"""
import json
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import ROOT
from smp_forecast_app.src.intervals import add_keys, methods, report, score, select
from smp_forecast_app.src.jz_common import HS, JDIR, OUT, actual_table
from smp_forecast_app.src.jz_resid import BASES, OBJ, RES, SETS
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src.wz_ens import ens_candidates, mae

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
NAMES = {"jnaive7": "기준: 지난주 같은 시간", "jsame": "기준: 최근 같은 유형일", "land": "육지 예측 그대로",
         "land_c2s": "육지 Chronos-2 small", "jc2": "Chronos-2", "jc2s": "Chronos-2 small", "jbolt": "Chronos-Bolt small",
         "jc2x": "Chronos-2 + 육지 공변량", "jc2sx": "Chronos-2 small + 육지 공변량",
         "jlgb": "LightGBM", "jlgbl1": "LightGBM(절대오차)", "jxgb": "XGBoost", "jridge": "Ridge", "jcat": "CatBoost",
         "jrf": "랜덤포레스트", "jet": "엑스트라트리", "jebm": "EBM", "jmlp": "MLP", "jtabpfn": "TabPFN",
         "jlstm": "LSTM", "jgru": "GRU", "jcnn": "1D-CNN", "jtransformer": "Transformer"}
BASE_NAMES = {"land": "육지 예측", "jc2sx": "Chronos-2 small+육지", "jc2s": "Chronos-2 small"}
BASELINES = ["jnaive7", "jsame"]
FAMILY = {"육지": ["land", "land_c2s"], "부스팅": ["jlgb", "jlgbl1", "jxgb", "jcat"], "트리": ["jrf", "jet"],
          "선형·GAM": ["jridge", "jebm"], "신경망(표)": ["jmlp", "jtabpfn"], "시계열 신경망": ["jlstm", "jgru", "jcnn", "jtransformer"],
          "Chronos": ["jc2", "jc2s", "jbolt", "jc2x", "jc2sx"], "잔차 보정": []}
SEG_MID = (11, 15)
LAMBDAS = (0.1, 0.25, 0.5, 0.75, 1.0)   # 육지(0.25~1.0)보다 작은 값 추가: 제주 잔차 보정이 작은 λ에서 최적 (2025 상반기로 선택)


def load_all():
    act = actual_table()
    frames = {}
    for p in sorted(OUT.glob("*.csv.gz")):
        name = p.name.replace(".csv.gz", "")
        if name == "jlgbq" or name not in NAMES:
            continue
        frames[name] = pd.read_csv(p, parse_dates=["date"]).set_index(["date", "hour", "H"])["pred"]
    P = pd.DataFrame(frames).reset_index().merge(act, on=["date", "hour"], how="left")
    cal = build_calendar(P["date"].min(), P["date"].max()).set_index("date")["is_offday"]
    P["offday"] = P["date"].map(cal).astype(int)
    P["base_date"] = P["date"] - pd.to_timedelta(P["H"], unit="D")
    for b in BASES:
        for k in SETS:
            for o in OBJ:
                f = RES / f"{b}_{k}_{o}.csv.gz"
                if f.exists():
                    r = pd.read_csv(f, parse_dates=["date"]).rename(columns={"rhat": f"rh_{b}_{k}_{o}"})
                    P = P.merge(r, on=["date", "hour", "H"], how="left")
    return P


def resid_series(P, fit_mask):
    """잔차 보정 후보: {열 이름: (예측, λ, 기준, 잔차열)} — λ는 fit_mask로 선택."""
    out = {}
    for b in BASES:
        if b not in P:
            continue
        for k in SETS:
            for o in OBJ:
                col = f"rh_{b}_{k}_{o}"
                if col not in P:
                    continue
                F = P[fit_mask]
                lam = min(LAMBDAS, key=lambda l: mae(F["y"], F[b] + l * F[col]))
                out[f"R_{b}_{k}_{o}"] = (P[b] + lam * P[col], lam, b, col)
    return out


def seg_mae(P, mask, col):
    g = P[mask]
    mid = g["offday"].astype(bool) & g["hour"].between(*SEG_MID)
    return {"휴무일 한낮": mae(g.loc[mid, "y"], g.loc[mid, col]), "급락 시간(30원 미만)": mae(g.loc[g["y"] < 30, "y"], g.loc[g["y"] < 30, col])}


def n_members(ms):
    return sum(2 if m.startswith("R_") else 1 for m in ms)


def label(m):
    if m.startswith("R_"):
        _, b, k, o = m.split("_")
        return f"{BASE_NAMES[b]} + 잔차 보정({k}·{o})"
    return NAMES.get(m, m)


def build(P, fit_mask):
    """fit_mask로 λ를 맞춘 잔차 보정 열을 P에 붙이고 (pool, 정보) 반환."""
    rs = resid_series(P, fit_mask)
    P = P.drop(columns=[c for c in P if c.startswith("R_")])
    for name, (s, lam, b, col) in rs.items():
        P[name] = s
    return P, {k: v[1] for k, v in rs.items()}


def main(score2026=False):
    P0 = load_all()
    y25 = P0["date"].dt.year == 2025
    h1 = y25 & (P0["date"] < "2025-07-01")
    h2 = y25 & (P0["date"] >= "2025-07-01")
    models = [m for m in NAMES if m in P0.columns and m not in BASELINES]

    # 1) 개별 모델 + 잔차 보정 (λ는 H1로)
    P, lam_h1 = build(P0, h1)
    rcols = sorted(lam_h1)
    pool = models + rcols
    full = P[pool + BASELINES].notna().all(axis=1)
    print(f"비교 행: 2025 상반기 {int((h1 & full).sum())} · 하반기 {int((h2 & full).sum())} (모든 후보가 예측한 행)", flush=True)
    rows = []
    for m in BASELINES + pool:
        r = {"코드": m, "모델": label(m) + (f" λ={lam_h1[m]}" if m in lam_h1 else ""),
             "2025 상반기": mae(P.loc[h1 & full, "y"], P.loc[h1 & full, m]),
             "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, m]),
             "2025 전체": mae(P.loc[y25 & full, "y"], P.loc[y25 & full, m])}
        r.update({f"하반기 {k}": v for k, v in seg_mae(P, h2 & full, m).items()})
        for H in HS:
            mm = h2 & full & (P["H"] == H)
            r[f"하반기 D+{H}"] = mae(P.loc[mm, "y"], P.loc[mm, m])
        rows.append(r)
    mt = pd.DataFrame(rows).sort_values("2025 하반기")
    TAB.mkdir(parents=True, exist_ok=True)
    mt.round(3).to_csv(TAB / "JWZ1_models.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(mt[["모델", "2025 상반기", "2025 하반기", "2025 전체", "하반기 휴무일 한낮", "하반기 급락 시간(30원 미만)", "하반기 D+1", "하반기 D+7"]]
          .round(2).to_string(index=False), flush=True)

    # 2) 앙상블: H1로 맞추고 H2로 비교
    fam = dict(FAMILY, **{"잔차 보정": rcols})
    cands, rank = ens_candidates(P, h1 & full, pool, family=fam)
    erows = [{"앙상블": "단일 최고 (" + label(rank[0]) + ")", "구성": label(rank[0]), "구성 수": n_members([rank[0]]),
              "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, rank[0]])}]
    for r in rcols:   # 잔차 보정 단독도 후보
        erows.append({"앙상블": label(r) + f" λ={lam_h1[r]}", "구성": label(r), "구성 수": 2,
                      "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, r]), "_key": r})
    for name, (p, ms) in cands.items():
        erows.append({"앙상블": name, "구성": ", ".join(label(m) for m in ms), "구성 수": n_members(ms),
                      "2025 하반기": mae(P.loc[h2 & full, "y"], p[h2 & full])})
    et = pd.DataFrame(erows)
    if "_key" not in et:
        et["_key"] = np.nan
    et = et.sort_values("2025 하반기")
    et.drop(columns="_key").round(3).to_csv(TAB / "JWZ2_ensembles_2025H2.csv", index=False, encoding="utf-8-sig")
    print(et[["앙상블", "구성 수", "2025 하반기"]].head(30).round(3).to_string(index=False), flush=True)
    top = et.iloc[0]["2025 하반기"]
    pr = et[et["2025 하반기"] <= top + 0.05].sort_values(["구성 수", "2025 하반기"]).iloc[0]
    pick, pick_key = pr["앙상블"], pr.get("_key")
    print(f"\n선택: {pick}\n", flush=True)

    # 3) 2025 전체로 다시 맞춤
    Pf, lam_full = build(P0, y25)
    fullf = Pf[pool + BASELINES].notna().all(axis=1)
    cands_f, rank_f = ens_candidates(Pf, y25 & fullf, pool, family=fam)
    if pick.startswith("단일 최고"):
        Pf["pred"], members = Pf[rank_f[0]], [rank_f[0]]
    elif isinstance(pick_key, str):
        Pf["pred"], members = Pf[pick_key], [pick_key]
    else:
        Pf["pred"], members = cands_f[pick][0], cands_f[pick][1]
    P = Pf

    # 4) 범위 (지평별, 하한 0원 자르기 없음)
    qsrc = {}
    for qname, label_q in (("jlgbq", "LightGBM 분위수"), ("jc2x", "Chronos-2+육지 분위수"), ("jc2sx", "Chronos-2 small+육지 분위수"),
                           ("jc2s", "Chronos-2 small 분위수")):
        if (OUT / f"{qname}.csv.gz").exists():
            qsrc[qname] = (label_q, pd.read_csv(OUT / f"{qname}.csv.gz", parse_dates=["date"]))
    irows, outs, ichoice = [], [], {}
    for H in HS:
        d = P[(P["H"] == H) & P["pred"].notna()].copy()
        d["offday"] = d["offday"].astype(bool)
        allk = add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))
        allk = allk.assign(err=allk["err"].fillna(0), rerr=allk["rerr"].fillna(0))
        for qn, (_, qdf) in qsrc.items():
            qq = qdf[qdf["H"] == H][["date", "hour", "qlo", "qhi"]].rename(columns={"qlo": f"qlo_{qn}", "qhi": f"qhi_{qn}"})
            allk = allk.merge(qq, on=["date", "hour"], how="left")
        ev = allk["y"].notna().values & (allk["date"] >= "2025-07-01").values & (allk["date"] < "2026-01-01").values
        cand = {}
        for k, f in methods(gap=H, clip0=False).items():
            if "최근1년" in k or "365일" in k:
                continue
            cand["오차분위수·" + k] = f
        for qn, (lq, _) in qsrc.items():
            kq = allk.assign(qlo=allk[f"qlo_{qn}"], qhi=allk[f"qhi_{qn}"])
            for w in (90, 180):
                cand[f"보정분위수회귀({lq})·갱신{w}일·시간대묶음×휴무일"] = (lambda kq, w: lambda dd, e: cqr(kq, e, window=w, key="k_band_off", gap=H, clip0=False))(kq, w)
            cand[f"{lq} 그대로"] = (lambda kq: lambda dd, e: (kq.loc[e, "qlo"].values, kq.loc[e, "qhi"].values))(kq)
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
        outs.append(allk[allmask].assign(lo=lo, hi=hi, H=H))
        r = sel[(sel["방법"] == ichoice[H]) & (sel["구간"] == "전체")].iloc[0]
        rm = sel[(sel["방법"] == ichoice[H]) & (sel["구간"] == "휴무일 한낮(11~15시)")]
        print(f"범위 D+{H}: {ichoice[H]} — 2025 하반기 포함률 {r['포함률']:.1%} (아래 {r['하한아래']:.1%}·위 {r['상한위']:.1%}), "
              f"폭 {r['평균폭']:.1f}, 휴무일 한낮 포함률 {rm['포함률'].iloc[0]:.1%}", flush=True)
    pd.DataFrame(irows).round(4).to_csv(TAB / "JWZ4_intervals.csv", index=False, encoding="utf-8-sig")
    W = pd.concat(outs)
    W["base_date"] = W["date"] - pd.to_timedelta(W["H"], unit="D")
    keep = ["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]
    JDIR.mkdir(parents=True, exist_ok=True)
    W[keep].round(2).to_csv(JDIR / "JWZ_predictions_hourly.csv", index=False, encoding="utf-8-sig")
    info = {"선택": pick, "구성": [label(m) for m in members], "구성 코드": members,
            "잔차 보정 λ(2025 전체)": {m: lam_full[m] for m in members if m in lam_full},
            "단일 최고(2025)": label(rank_f[0]), "범위": {str(k): v for k, v in ichoice.items()}, "범위 하한": "0원 자르기 없음 (제주 음수 가격)",
            "선택 절차": "2025 상반기로 맞춤 → 2025 하반기 비교 → 2025 전체로 재적합", "시험": "2026-01-01 ~ (제주 1차 시험)"}
    (TAB / "JWZ_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))

    if score2026:   # 사용자 확인 후 한 번
        te = W[(W["date"] >= "2026-01-01") & W["y"].notna()]
        trows = []
        for H in ["전체"] + list(HS):
            g = te if H == "전체" else te[te["H"] == H]
            base = P[(P["date"] >= "2026-01-01") & P["y"].notna() & ((P["H"] == H) if H != "전체" else True)]
            trows.append({"H": H, **score(g["y"], g["lo"], g["hi"], g["pred"]),
                          "육지 예측 그대로 MAE": mae(base["y"], base["land"]),
                          "단일 최고(2025) MAE": mae(base["y"], base[rank_f[0]]), "기준(지난주) MAE": mae(base["y"], base["jnaive7"])})
        test = pd.DataFrame(trows)
        test.round(4).to_csv(TAB / "JWZ3_test2026.csv", index=False, encoding="utf-8-sig")
        print(test.round(3).to_string(index=False))


if __name__ == "__main__":
    main("--score2026" in sys.argv[1:])
