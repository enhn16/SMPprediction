"""제주 급락 1차(15시) v2 — 급락 = 모든 날 (팀 규칙 3, 10/28 변경) + 육지 일주일 입력 = K3 (R2, 2026-11-03).

jeju_team(휴일·주말 v1_offday)과 다른 점만 (규칙은 결과 보기 전 고정, JEJU.md 11절)
  육지 입력: L_pred_midmin·L_c2s_midmin = K3 H=1 한낮(11~15시) 최저 (점예측 pred, 기본 Chronos c), wz3/ops/k3_parts.csv.gz.
            K3가 없는 날(2023, 대상일 2024-01-01)만 이전 육지 값. → 이전 육지 일주일 모델 없이 운영 가능
  후보: jeju_team과 같은 21개 (피처 S4·S5·S5c + has_om × 모델 7)
  선택: 2025 **전체 날** PR-AUC 두 검증 중 나쁜 쪽 ((i) 2023~2024 → 2025, (ii) 2023~2025상 → 2025하), 0.03 이내면 단순한 쪽
        → 헛경보@재현율80% → PR-AUC
  문턱 ((i) 2025 전체 날 예측): 주의 = 재현율 80% (급락 60일 중 48일), 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨
        (없으면 정밀도 최고), 경보가 주의보다 느슨하면 경보 = 주의. 확률 = 같은 예측(전체 날)에 로지스틱 보정
  최종: 2023-01 ~ 2025-12 고정 학습 → 2026-01-01 ~ 10-18 한 번 채점. 행 = 전체 날(공식) / 휴일·주말 / 평일, 지금 운영 모델과 나란히
출력 (ROLES.md 4.1): drop_제주_15h.csv (label = 모든 날), drop_choice_제주.json, drop_score_제주.csv, J10_team2_select_2025.csv
      이전 파일은 *_v1_offday로 보관 (처음 실행 때 한 번만 복사)
실행: python -m drop_classifier.src.jeju_team2 select | test
"""
import json
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common.config import PROCESSED_DIR
from drop_classifier.src import jeju_drop as jd
from drop_classifier.src import jeju_team as jt
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_final import alarm_threshold
from drop_classifier.src.team_select import score_rows
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
TAB = jd.TAB
REGION = "제주"
MID = list(range(11, 16))
K3 = PROCESSED_DIR / "smp_forecast_app" / "wz3" / "ops" / "k3_parts.csv.gz"
KEEP = ["drop_제주_15h.csv", "drop_choice_제주.json", "drop_score_제주.csv"]


def k3_mid() -> pd.DataFrame:
    """대상일 → [L_base, L_c2s] = K3 H=1 한낮 최저 (점예측, 기본 Chronos). 없는 날은 이전 육지 값."""
    k = pd.read_csv(K3, usecols=["date", "hour", "H", "pred", "c"], parse_dates=["date"])
    k = k[(k["H"] == 1) & k["hour"].isin(MID) & k["pred"].notna()].drop_duplicates(["date", "hour"], keep="last")
    k = k.groupby("date").agg(L_base=("pred", "min"), L_c2s=("c", "min"), n=("pred", "size"))
    k = k[k["n"] == 5][["L_base", "L_c2s"]]
    from smp_forecast_app.src.jz_common import land_pred
    lp = land_pred()
    old = lp[(lp["H"] == 1) & lp["hour"].isin(MID)].groupby("date")[["L_base", "L_c2s"]].min()
    return k.combine_first(old)


def table(**kw):
    d = jd.day_table(start=jt.START, land_mid=k3_mid(), **kw)
    d["has_om"] = d["jo_rad_mid"].notna().astype(float)
    return d


def select():
    t0 = time.time()
    d = table()
    rows = []
    for fname, cols in jt.FEATS.items():
        res = {}
        for vname, (tr_rng, te_rng) in jt.VAL.items():
            tr, te = jt.rows_of(d, tr_rng, te_rng)
            y = te["label"].astype(int).values
            cache = {}
            for m in jt.TIER:
                res[(vname, m)] = drop_stats(y, jt.fit_score(m, tr, te, cols, cache))
            print(f"  {fname} / {vname}: 학습 {len(tr)}일(급락 {int(tr['label'].sum())}), 전체 날 {len(y)}일(급락 {y.sum()}) "
                  f"{time.time() - t0:.0f}초", flush=True)
        vi, vii = jt.VAL
        for m in jt.TIER:
            a, b = res[(vi, m)], res[(vii, m)]
            rows.append({"피처": fname, "모델": m, "단계": jt.TIER[m], "PR-AUC (i)": a["PR-AUC"], "PR-AUC (ii)": b["PR-AUC"],
                         "PR-AUC 나쁜쪽": min(a["PR-AUC"], b["PR-AUC"]), "헛경보@80% (i)": a["헛경보@재현율80%"],
                         "헛경보@90% (i)": a["헛경보@재현율90%"], "헛경보@80% (ii)": b["헛경보@재현율80%"], "ROC-AUC (i)": a["ROC-AUC"]})
    sel = pd.DataFrame(rows).sort_values("PR-AUC 나쁜쪽", ascending=False).reset_index(drop=True)
    sel["동점권"] = sel["PR-AUC 나쁜쪽"] >= sel["PR-AUC 나쁜쪽"].iloc[0] - jt.TIE
    pick = sel[sel["동점권"]].sort_values(["단계", "헛경보@80% (i)", "PR-AUC 나쁜쪽"], ascending=[True, True, False]).iloc[0]
    sel["선택"] = (sel["피처"] == pick["피처"]) & (sel["모델"] == pick["모델"])
    sel.round(4).to_csv(TAB / "J10_team2_select_2025.csv", index=False, encoding="utf-8-sig")
    (TAB / "J10_team2_pick.json").write_text(json.dumps({"피처": pick["피처"], "모델": pick["모델"]}, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(sel.round(3).to_string(index=False))
    print(f"선택: {pick['피처']} / {pick['모델']} (총 {time.time() - t0:.0f}초)")


def scoped(prefix, t, s, p, warn, strong):
    out = []
    for scope, m in (("전체 날 (공식)", np.ones(len(t), bool)), ("휴일·주말", t["offday"].astype(bool).values),
                     ("평일", ~t["offday"].astype(bool).values)):
        y = t["label"].astype(int).values[m]
        for r in score_rows(f"{prefix}", y, s[m], p[m], warn, strong):
            out.append({"범위": scope, **r})
    return out


def test():
    t0 = time.time()
    for f in KEEP:                     # 이전(휴일·주말) 결과 보관 — 한 번만
        src, dst = TAB / f, TAB / f.replace(".", "_v1_offday.", 1)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    pk = json.loads((TAB / "J10_team2_pick.json").read_text(encoding="utf-8"))
    cols, name = jt.FEATS[pk["피처"]], pk["모델"]
    d = table()
    tr_v, _ = jt.rows_of(d, *jt.VAL["(i) 2023~2024 → 2025"])
    va = d[d["date"].between("2025-01-01", "2025-12-31")]
    sv = jt.fit_score(name, tr_v, va, cols)
    tr_f = d[d["date"].between(*jt.TR_FIN) & d["label"].notna()]
    te = d[d["date"] >= jt.TEST[0]]
    st = jt.fit_score(name, tr_f, te, cols)
    vo = va["label"].notna().values
    yv = va.loc[vo, "label"].astype(int).values
    warn, (strong, how) = threshold_for(yv, sv[vo], 0.8), alarm_threshold(yv, sv[vo])
    if strong < warn:
        strong, how = warn, f"주의 문턱과 같음 (정밀도 50% 문턱 {strong:.4f}이 주의보다 느슨)"
    cal = LogisticRegression(C=1e6).fit(sv[vo].reshape(-1, 1), yv)
    to_prob = lambda s: cal.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    print(f"문턱: 주의 {warn:.4f}, 경보 {strong:.4f} ({how})", flush=True)

    to = (te["date"].between(*jt.TEST) & te["label"].notna()).values
    T = f"2026 시험 {jt.TEST[0]} ~ {jt.TEST[1]}"
    rows = scoped("2025 검증", va[vo], sv[vo], to_prob(sv[vo]), warn, strong) + scoped(T, te[to], st[to], to_prob(st[to]), warn, strong)
    sc = pd.DataFrame(rows)
    sc.insert(0, "모델", f"새: {pk['피처']} · {name} (전체 날 문턱, 육지 입력 K3, 2023~2025 학습)")
    # 지금 운영 모델 (휴일·주말 문턱 v1_offday) — 같은 날
    cur = pd.read_csv(TAB / "drop_제주_15h_v1_offday.csv", parse_dates=["date"])
    chc = json.loads((TAB / "drop_choice_제주_v1_offday.json").read_text(encoding="utf-8"))
    c = te[to][["date", "label", "offday"]].merge(cur[["date", "score", "prob"]], on="date", how="left")
    cr = pd.DataFrame(scoped(T, c, c["score"].values, c["prob"].values, chc["문턱"]["주의"], chc["문턱"]["경보"]))
    cr.insert(0, "모델", "지금: S4 · KNN(분류) (휴일·주말 문턱, 이전 육지 입력)")
    sc = pd.concat([sc, cr], ignore_index=True)
    sc.round(4).to_csv(TAB / f"drop_score_{REGION}.csv", index=False, encoding="utf-8-sig")

    both = pd.concat([va.assign(s=sv), te.assign(s=st)])
    lab = both["label"]
    out = pd.DataFrame({"base_date": (both["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
                        "date": both["date"].dt.strftime("%Y-%m-%d"),
                        "grade": np.where(both["s"] >= strong, "경보", np.where(both["s"] >= warn, "주의", "없음")),
                        "prob": to_prob(both["s"]).round(4), "score": both["s"].round(4),
                        "label": np.where(lab.notna(), lab.fillna(0).astype(int).astype(str), "")})
    out.to_csv(TAB / f"drop_{REGION}_15h.csv", index=False, encoding="utf-8-sig")
    choice = {"stage": "15h (전날 15시)", "계통": REGION, "모델": name, "구성": jt.ENS.get(name, [name]), "피처 묶음": pk["피처"],
              "피처": cols, "점수": "score = 급락 분류 확률(모델 출력, 0~1), 높을수록 위험",
              "문턱": {"주의": float(warn), "경보": float(strong), "규칙": "score ≥ 문턱이면 해당 단계 (반올림 전 값)"},
              "확률식": {"형태": "prob = 1/(1+exp(-(a·score+b)))", "a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])},
              "학습": {"문턱·확률": "2023-01-01 ~ 2024-12-31 학습 → 2025 전체 날 예측",
                     "최종": "2023-01-01 ~ 2025-12-31 (고정, 2026 예측에 사용)", "학습 행": "전체 날(평일 포함)"},
              "육지 입력": "L_pred_midmin·L_c2s_midmin = 육지 일주일 K3 H=1 한낮 최저 (k3_parts.csv.gz, 2023은 이전 육지 값)",
              "문턱 규칙": "주의 = 2025 전체 날 급락 80% 포착, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨 "
                       "(없으면 적중 3일 이상 중 정밀도 최고), 경보가 주의보다 느슨하면 경보 = 주의",
              "경보 문턱 적용": how,
              "선택 절차": "21개 후보(피처 3 × 모델 7) 2025 전체 날 PR-AUC 두 검증 중 나쁜 쪽, 0.03 이내면 단순한 쪽 (J10_team2_select_2025.csv)",
              "2023 자료": "Open-Meteo는 2024-01부터 → 학습 중앙값 + has_om",
              "급락 정의": "11~15시 최저 SMP 30원 미만, 모든 날 (팀 규칙 3, 10/28)",
              "이전": "drop_choice_제주_v1_offday.json (휴일·주말 문턱, 이전 육지 입력)"}
    (TAB / f"drop_choice_{REGION}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")

    show = ["모델", "범위", "구간", "경보", "일수", "급락일", "PR-AUC", "ROC-AUC", "헛경보@재현율80%", "헛경보@재현율90%", "Brier",
            "적중", "놓침", "헛경보", "재현율", "정밀도"]
    pd.set_option("display.width", 300)
    print(sc[show].round(3).to_string(index=False))
    a26 = out[out["date"].between(*jt.TEST)]
    print("2026 등급 일수 (전체 날):", a26["grade"].value_counts().to_dict(), f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "select"]()
