"""급등 모델 세 가지를 같은 기간·같은 지표로 비교 (2026-10-28).

- 지금 지수식 (v1_index): 평일×(운량+흐림+주간전망), 학습 없음. 검증 점수는 models.index_score, 2026은 surge_육지_15h_v1_index.csv
- 우리 구현 v1 (team_model): 자체 설계 세 점수 평균, 해마다 앞으로만 검증 — 팀원 모델 재현 실패로 기록용
- v2 (team_model2): 팀원 spike_simple3 재구현, 매달 앞으로만 검증
지표: PR-AUC·배수·ROC-AUC·헛경보@재현율80%(문턱 없이 순위 비교) + 각 모델의 미리 정한 주의·경보 문턱 성적 + 상위 10일 적중.
실행: python -m surge_classifier.src.compare_versions → outputs/tables/06_compare_versions_육지.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from surge_classifier.src import team_model as T1
from surge_classifier.src.label_days import OUT_TAB
from surge_classifier.src.models import index_score, z_stats

PERIODS = {"검증 2024~2025": ("2024-01-01", "2025-12-31"), "2026 공식 ~10-18": ("2026-01-01", "2026-10-18"),
           "2026 팀원 기간 ~10-02": ("2026-01-01", "2026-10-02")}


def fa_at(y, s, r):
    o = np.argsort(-s, kind="stable")
    k = np.searchsorted(np.cumsum(y[o]), int(np.ceil(r * y.sum())))
    return int(((s >= s[o][k]) & (y == 0)).sum())


def scores() -> dict:
    """모델별 (날짜 → 점수, 레이블) Series 묶음과 문턱."""
    d = T1.data()
    out = {}
    # 지금 지수식
    v = d.loc["2024-01-01":"2025-12-31"]
    idx_val = pd.Series(index_score(v, "지수: 평일×(운량+흐림+주간전망)", z_stats(d.reset_index())), v.index)
    old = pd.read_csv(OUT_TAB / "surge_육지_15h_v1_index.csv", parse_dates=["date"]).set_index("date")
    idx = pd.concat([idx_val, old.loc["2026-01-01":, "score"]])
    th = json.loads((OUT_TAB / "surge_choice_육지_15h_v1_index.json").read_text(encoding="utf-8"))["문턱 (모델 내부 점수)"]
    out["지금 지수식"] = (idx, th)
    # 우리 구현 v1
    ch = json.loads((OUT_TAB / "04_team_choice_육지.json").read_text(encoding="utf-8"))
    parts = {k: {ch["척도"]: T1.oof(d, T1.FITS[k][0], next(c for c in T1.FITS[k][1] if c["name"] == ch["설정"][k]), ch["척도"])}
             for k in T1.FITS}
    v1_val = T1.ensemble(parts, ch["척도"], ch["가중"])
    v1_test = pd.read_csv(OUT_TAB / "04_team_daily_육지.csv", parse_dates=["date"]).set_index("date")["score"]
    out["우리 구현 v1"] = (pd.concat([v1_val, v1_test]), ch["문턱"])
    # v2 (팀원 방식)
    v2 = pd.read_csv(OUT_TAB / "05_team2_daily_육지.csv", parse_dates=["date"]).set_index("date")["avg"]
    th2 = json.loads((OUT_TAB / "05_team2_choice_육지.json").read_text(encoding="utf-8"))["문턱"]["avg"]
    out["v2 팀원 방식"] = (v2, th2)
    return out, d["label_25"]


def main():
    sc, lab = scores()
    rows = []
    for pname, (a, b) in PERIODS.items():
        for mname, (s, th) in sc.items():
            s = s.loc[a:b]
            y = lab.reindex(s.index)
            ok = y.notna()
            s, y = s[ok].values.astype(float), y[ok].values.astype(int)
            hit = lambda t: (int(((s >= t) & (y == 1)).sum()), int(((s >= t) & (y == 0)).sum()))
            (wt, wf), (st, sf) = hit(th["주의"]), hit(th["경보"])
            rows.append({"구간": pname, "모델": mname, "일수": len(y), "급등일": int(y.sum()),
                         "PR-AUC": average_precision_score(y, s), "PR-AUC 배수": average_precision_score(y, s) / y.mean(),
                         "ROC-AUC": roc_auc_score(y, s), "헛경보@재현율80%": fa_at(y, s, .8),
                         "상위 10일 적중": int(y[np.argsort(-s, kind="stable")[:10]].sum()),
                         "주의 적중": wt, "주의 헛경보": wf, "주의 정밀도": wt / max(wt + wf, 1),
                         "경보 적중": st, "경보 헛경보": sf, "경보 정밀도": st / max(st + sf, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "06_compare_versions_육지.csv", index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
