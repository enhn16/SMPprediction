"""일주일 v3 (R3) ⑦ 미세조정 주기 비교 — 1년(Y) vs 분기(Q). 규칙은 WEEK_V3.md ⑦ (결과 보기 전 고정).

판은 기준일 기준으로 고른다: 마감 C인 판은 기준일 I ≥ C (다음 판 마감 전날까지)에만 쓴다. 그 전 기준일은 제로샷.
  train    : 분기 마감 판 6개 학습 (wz3_finetune, 전체 1e-6·500단계)
  infer    : 판마다 맡은 기준일 범위만 추론 → wz3/chronos/ftq_{판}.csv.gz
  assemble : 대상일 2023~2025 Chronos 파일 c2s_fty_2023-01-01 (1년) · c2s_ftq_2023-01-01 (분기)
  eval     : 잔차 fty·ftq (wz3_resid) 뒤 mix|fty|ext vs mix|ftq|ext → outputs/tables/WZ3_Q1_ft_cycle.csv
실행: python -m smp_forecast_app.src.wz3_ftq train infer assemble eval
"""
import sys

import pandas as pd

from smp_forecast_app.src import wz3_resid as resid
from smp_forecast_app.src.wz3_chronos import CH, W3, run
from smp_forecast_app.src.wz3_finetune import finetune
from smp_forecast_app.src.wz3_select import TAB, accept, eval_cfg
from smp_forecast_app.src.wz_chronos import HIST

NEW = ["fq2403f", "fq2406f", "fq2409f", "fq2503f", "fq2506f", "fq2509f"]
END = "2025-12-31"      # 마지막 기준일 다음 날 (대상일 2025까지)
Y = [("ft2023f", "2023-12-31"), ("ft2024f", "2024-12-31")]
Q = [("ft2023f", "2023-12-31"), ("fq2403f", "2024-03-31"), ("fq2406f", "2024-06-30"), ("fq2409f", "2024-09-30"),
     ("ft2024f", "2024-12-31"), ("fq2503f", "2025-03-31"), ("fq2506f", "2025-06-30"), ("fq2509f", "2025-09-30")]
OUT = TAB / "WZ3_Q1_ft_cycle.csv"


def spans(plan):
    """[(판, 첫 기준일, 끝 기준일(미포함))]"""
    cuts = [c for _, c in plan] + [END]
    return [(n, pd.Timestamp(c), pd.Timestamp(cuts[i + 1])) for i, (n, c) in enumerate(plan)]


def train():
    for n in NEW:
        if (W3 / "ft" / n / "finetuned-ckpt").exists():
            print(f"{n}: 이미 있음 → 건너뜀")
            continue
        finetune(n)


def infer():
    need = {}
    for plan in (Y, Q):
        for n, a, b in spans(plan):
            lo, hi = need.get(n, (a, b))
            need[n] = (min(lo, a), max(hi, b))
    for n, (a, b) in need.items():
        p = CH / f"ftq_{n}.csv.gz"
        if p.exists():
            print(f"{p.name}: 이미 있음 → 건너뜀")
            continue
        bases = list(pd.date_range(a, b - pd.Timedelta(days=1)))
        out = run("c2s", "cal", "2000-01-01", "2100-01-01", bases_only=bases, ckpt=n)
        out.round(3).to_csv(p, index=False, encoding="utf-8")
        print(f"저장 {p.name} ({len(out):,}행, 기준일 {a.date()} ~ {(b - pd.Timedelta(days=1)).date()})", flush=True)


def assemble():
    hist = pd.read_csv(HIST / "c2s.csv.gz", parse_dates=["date"])
    base = lambda d: d["date"] - pd.to_timedelta(d["H"], unit="D")
    for tag, plan in (("fty", Y), ("ftq", Q)):
        sp = spans(plan)
        parts = [hist[base(hist) < sp[0][1]]]
        for n, a, b in sp:
            x = pd.read_csv(CH / f"ftq_{n}.csv.gz", parse_dates=["date"])
            parts.append(x[(base(x) >= a) & (base(x) < b)])
        out = pd.concat(parts, ignore_index=True)
        out = out[(out["date"] >= "2023-01-01") & (out["date"] <= END)].sort_values(["date", "hour", "H"])
        assert not out.duplicated(["date", "hour", "H"]).any()
        p = CH / f"c2s_{tag}_2023-01-01.csv.gz"
        out.to_csv(p, index=False, encoding="utf-8")
        print(f"저장 {p.name} ({len(out):,}행)")


def evaluate():
    for k in ("fty", "ftq"):
        resid.run(k)
    y, q = eval_cfg("Y 1년 미세조정", "mix|fty|ext"), eval_cfg("Q 분기 미세조정", "mix|ftq|ext")
    q |= {"Δ2024": q["2024 전체"] - y["2024 전체"], "Δ2025": q["2025 전체"] - y["2025 전체"], "분기 권장": accept(q, y)}
    t = pd.DataFrame([y, q])
    t.round(3).to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["후보", "λ", "2024 전체", "2025 전체", "Δ2024", "Δ2025", "2025 평일", "2025 휴무일", "분기 권장"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"train": train, "infer": infer, "assemble": assemble, "eval": evaluate}[step]()
