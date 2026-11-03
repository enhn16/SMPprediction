"""원본 데이터 로더. 원본 형식을 정리된 long 형식(날짜, 시간, 값)으로 바꿔 돌려준다.

시간(hour)은 원본 표기대로 1~24 (해당 시간대의 끝 시각, 예: 12 = 11:00~12:00).
"""
import numpy as np
import pandas as pd

from common.config import API_RAW_DIR, RAW_DIR

REGION_FILE = {"육지": "smpInLand", "제주": "smpJeju"}


def load_smp_excel(region: str) -> pd.DataFrame:
    """data/raw/smp{InLand,Jeju}_YYYY.xlsx → [date, hour, smp]."""
    frames = []
    for p in sorted(RAW_DIR.glob(f"{REGION_FILE[region]}_*.xlsx")):
        # 1행은 제목, 2행이 헤더
        df = pd.read_excel(p, header=1)
        df.columns = ["date"] + list(range(1, 25)) + ["max", "min", "mean"]
        frames.append(df.drop(columns=["max", "min", "mean"]))
    wide = pd.concat(frames, ignore_index=True)
    wide["date"] = pd.to_datetime(wide["date"].astype(str), format="%Y%m%d")
    out = wide.melt(id_vars="date", var_name="hour", value_name="smp")
    out["hour"] = out["hour"].astype(int)
    return out.sort_values(["date", "hour"]).reset_index(drop=True)


def load_smp_actual(region: str) -> pd.DataFrame:
    """[date, hour, smp, source]: 전력거래소 엑셀(공식 확정값) + 엑셀 이후 날짜는 API 공표값(하루전 확정 SMP).
    육지는 엑셀과 API가 42,048시간 전부 일치(data_inventory 6절)."""
    x = load_smp_excel(region).assign(source="엑셀")
    api = load_smp_forecast_api()
    api = api[(api["region"] == region) & (api["date"] > x["date"].max())]
    full = api.groupby("date")["hour"].transform("size") == 24
    api = api[full][["date", "hour", "smp"]].assign(source="API")
    return pd.concat([x, api], ignore_index=True).sort_values(["date", "hour"]).reset_index(drop=True)


def load_smp_forecast_api() -> pd.DataFrame:
    """data/raw/api/smp_forecast → [date, hour, region, smp, demand_fcst].

    demand_fcst는 해당 계통의 하루전 수요예측(MW): 육지=mlfd, 제주=jlfd.
    """
    files = sorted((API_RAW_DIR / "smp_forecast").glob("smp_forecast_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype={"date": str, "hour": str}) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["hour"] = df["hour"].astype(int)  # "1"과 "01"이 섞여 있음
    df["region"] = df["areaName"]
    df["demand_fcst"] = df["mlfd"].where(df["region"] == "육지", df["jlfd"])
    out = df[["date", "hour", "region", "smp", "demand_fcst"]]
    return out.drop_duplicates(["date", "hour", "region"]).sort_values(["region", "date", "hour"]).reset_index(drop=True)


def _read_csv_any(p) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {p}")


def _num(s: pd.Series) -> pd.Series:
    # 빈칸·공백은 발전량 0으로 처리 (원본에 숫자 뒤 공백, 빈 문자열 섞여 있음)
    return pd.to_numeric(s.str.strip().replace("", "0"), errors="coerce").fillna(0.0)


def load_renewables() -> pd.DataFrame:
    """data/raw 지역별 시간별 태양광·풍력 → [date, hour, region(육지/제주), solar, wind] (MWh, 시장 참여 설비만).

    파일 형식이 기간별로 다름:
    - ~2023-11: wide (지역=시도, 태양광·풍력 컬럼)
    - 2023-12~: long (연료원 컬럼). 태양광은 시도별, 풍력은 '육지'·'제주(도)' 합계 행만 있음
    """
    frames = []
    for p in sorted(RAW_DIR.glob("*지역별 시간별 태양광*.csv")):
        df = _read_csv_any(p)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "hour", df.columns[2]: "area"})
        df["area"] = df["area"].str.strip()
        df["region"] = df["area"].map(lambda a: "제주" if a in ("제주", "제주도") else "육지")
        if "연료원" in df.columns:
            df["value"] = _num(df[df.columns[4]])
            fuel = df["연료원"].str.strip()
            solar = df[(fuel == "태양광") & ~df["area"].isin(["육지", "제주"])]
            wind = df[fuel == "풍력"]
            part = pd.concat([
                solar.groupby(["date", "hour", "region"])["value"].sum().rename("solar"),
                wind.groupby(["date", "hour", "region"])["value"].sum().rename("wind"),
            ], axis=1)
        else:
            solar_col = next(c for c in df.columns if "태양광" in c)
            wind_col = next(c for c in df.columns if "풍력" in c)
            df["solar"], df["wind"] = _num(df[solar_col]), _num(df[wind_col])
            part = df.groupby(["date", "hour", "region"])[["solar", "wind"]].sum()
        frames.append(part.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["hour"] = out["hour"].astype(int)
    out = out.fillna({"solar": 0.0, "wind": 0.0})
    return out.drop_duplicates(["date", "hour", "region"], keep="last").sort_values(["region", "date", "hour"]).reset_index(drop=True)


def load_asos() -> pd.DataFrame:
    """data/raw/api/asos → [date, hour, stn_id, stn_name, ta, icsr, ss, cloud, rn, ws].

    ASOS 시각 tm(정시)은 직전 1시간 관측이므로 SMP 표기(1~24, 끝점)와 맞추기 위해
    00시는 전날 24시로 바꾼다. 강수(rn) 빈칸은 강수 없음으로 보고 0 처리.
    """
    files = sorted((API_RAW_DIR / "asos").glob("asos_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype=str) for p in files], ignore_index=True)
    tm = pd.to_datetime(df["tm"])
    df["hour"] = tm.dt.hour.replace(0, 24)
    df["date"] = (tm - pd.to_timedelta((tm.dt.hour == 0).astype(int), unit="D")).dt.normalize()
    out = pd.DataFrame({
        "date": df["date"], "hour": df["hour"],
        "stn_id": df["stnId"].astype(int), "stn_name": df["stnNm"],
        "ta": pd.to_numeric(df["ta"], errors="coerce"),
        "icsr": pd.to_numeric(df["icsr"], errors="coerce"),     # 일사량 MJ/m²
        "ss": pd.to_numeric(df["ss"], errors="coerce"),         # 일조 시간(hr)
        "cloud": pd.to_numeric(df["dc10Tca"], errors="coerce"), # 전운량(10분위)
        "rn": pd.to_numeric(df["rn"], errors="coerce").fillna(0.0),
        "ws": pd.to_numeric(df["ws"], errors="coerce"),
    })
    return out.sort_values(["stn_id", "date", "hour"]).reset_index(drop=True)


def load_short_forecast(stations: list[int] | None = None) -> pd.DataFrame:
    """data/raw/api/short_forecast → [date, hour, stn_id, base_date, SKY, POP, PTY, TMP, WSD, REH].

    D일 예보는 D-1 17시 발표분. fcstTime HH00을 hour=HH로 둔다 (정시 예보, SMP의 HH시 구간 끝과 같은 시각).
    SKY: 1 맑음, 3 구름많음, 4 흐림 / PTY: 0 없음, 1 비, 2 비·눈, 3 눈, 4 소나기
    """
    files = sorted((API_RAW_DIR / "short_forecast").glob("fcst_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype=str) for p in files], ignore_index=True)
    df["stn_id"] = df["stn_id"].astype(int)
    if stations is not None:
        df = df[df["stn_id"].isin(stations)]
    df = df[df["category"].isin(["SKY", "POP", "PTY", "TMP", "WSD", "REH"])]
    df["value"] = pd.to_numeric(df["fcstValue"], errors="coerce")
    df["date"] = pd.to_datetime(df["fcstDate"], format="%Y%m%d")
    df["hour"] = df["fcstTime"].str[:2].astype(int)
    df["base_date"] = pd.to_datetime(df["baseDate"], format="%Y%m%d")
    wide = df.pivot_table(index=["date", "hour", "stn_id", "base_date"], columns="category", values="value").reset_index()
    wide.columns.name = None
    return wide.sort_values(["stn_id", "date", "hour"]).reset_index(drop=True)


def load_mid_forecast() -> pd.DataFrame:
    """data/raw/FCT_MEDM_*.csv (기상청 중기 육상예보) → [issue_date, issue_hour, region, target_date, lead, ampm, sky, rain, pop].

    sky: 맑음 1, 구름많음* 3, 흐림·흐리고* 4 (단기예보 SKY 코드와 같은 척도) / rain: 비·눈·소나기 포함 여부
    '전라북도'는 2024년 중 '전북자치도'로 이름이 바뀌어 '전라북도'로 통일.
    """
    files = sorted(RAW_DIR.glob("FCT_MEDM_*.csv")) + sorted((API_RAW_DIR / "mid_fcst").glob("mid_*.csv"))  # 과거 파일 + 매일 수집분
    df = pd.concat([_read_csv_any(p) for p in files], ignore_index=True)
    df.columns = ["issue", "region", "target", "wf", "pop"]
    out = pd.DataFrame({
        "issue_date": pd.to_datetime(df["issue"].str[:10]),
        "issue_hour": df["issue"].str[11:13].astype(int),
        "region": df["region"].replace({"전북자치도": "전라북도"}),
        "target_date": pd.to_datetime(df["target"].str[:10]),
        "ampm": df["target"].str[-2:],
        "sky": np.select([df["wf"].str.startswith("맑음"), df["wf"].str.startswith("구름많")], [1, 3], 4),
        "rain": df["wf"].str.contains("비|눈|소나기").astype(int),
        "pop": pd.to_numeric(df["pop"], errors="coerce"),
    })
    out["lead"] = (out["target_date"] - out["issue_date"]).dt.days
    return out.drop_duplicates().reset_index(drop=True)


def load_kma_regional_fcst() -> pd.DataFrame:
    """data/raw/api/kma_afs (기상청 단기예보 구역별 육상예보) → [reg, tm_fc, tm_ef, ta, pop, sky, rain, wf].

    tm_fc 발표시각(05·11·17시), tm_ef 발효시각(D 00:00 = 오전, D 12:00 = 오후 구간).
    sky: DB01 맑음 1, DB02 구름조금 2, DB03 구름많음 3, DB04 흐림 4 / ta -99 = 결측 / rain: 강수유무 > 0
    """
    rows = []
    for p in sorted((API_RAW_DIR / "kma_afs").glob("afs_*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 16:
                continue
            rows.append({"reg": parts[0], "tm_fc": parts[1], "tm_ef": parts[2], "ta": parts[12], "pop": parts[13],
                         "sky": parts[14], "rain": parts[15], "wf": " ".join(parts[16:]).strip('"')})
    df = pd.DataFrame(rows)
    df["tm_fc"] = pd.to_datetime(df["tm_fc"], format="%Y%m%d%H%M")
    df["tm_ef"] = pd.to_datetime(df["tm_ef"], format="%Y%m%d%H%M")
    df["ta"] = pd.to_numeric(df["ta"], errors="coerce").replace(-99, np.nan)
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
    df["sky"] = df["sky"].str[-1].astype(int)
    df["rain"] = (pd.to_numeric(df["rain"], errors="coerce") > 0).astype(int)
    return df.drop_duplicates(["reg", "tm_fc", "tm_ef"]).reset_index(drop=True)


def load_gen_by_fuel() -> pd.DataFrame:
    """data/raw/api/gen_by_fuel → [date, hour, fuel, mwh] (정산 실적, 육지+제주 합계로 보임)."""
    files = sorted((API_RAW_DIR / "gen_by_fuel").glob("gen_by_fuel_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype=str) for p in files], ignore_index=True)
    return pd.DataFrame({
        "date": pd.to_datetime(df["tradeYmd"], format="%Y%m%d"),
        "hour": df["tradeNo"].astype(int),
        "fuel": df["fuelTpCd"],
        "mwh": pd.to_numeric(df["amgo"], errors="coerce"),
    })


def load_holidays() -> pd.DataFrame:
    """data/raw/api/holidays → [date, name] (공휴일·대체·임시·선거일 포함)."""
    files = sorted((API_RAW_DIR / "holidays").glob("holidays_*.csv"))
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df[df["isHoliday"] == "Y"]
    df["date"] = pd.to_datetime(df["locdate"].astype(str), format="%Y%m%d")
    # 같은 날 공휴일이 겹치면(예: 2025-05-05 어린이날·부처님오신날) 이름을 합침
    out = df.groupby("date")["dateName"].agg(lambda s: "·".join(dict.fromkeys(s))).reset_index()
    return out.rename(columns={"dateName": "name"})


FUEL_COST_COLS = {"c11": "nuclear", "c12": "coal", "c13": "anthracite", "c14": "oil", "c15": "lng"}


def load_fuel_cost() -> pd.DataFrame:
    """data/raw/api/epsis_fuel_cost → [month, fc_nuclear, fc_coal, fc_anthracite, fc_oil, fc_lng, usable_from].

    연료비단가(원/kWh). EPSIS는 매월 초 전월 통계치를 갱신하므로, 누수 방지를 위해
    M월 값은 M+2월 1일부터 쓸 수 있다고 본다(usable_from).
    """
    import re
    rows = {}
    for p in sorted((API_RAW_DIR / "epsis_fuel_cost").glob("fuc_*.txt")):
        text = p.read_text(encoding="utf-8")
        # 블록: c값 정의 → push({"Period":"YYYY/MM"...})
        for vals, period in re.findall(r"((?:c\d+ = textFormmat\(\"[^\"]*\",count\);\s*)+)\s*gridData\.push\(\{\"Period\":\"(\d{4}/\d{2})\"", text):
            v = dict(re.findall(r"(c\d+) = textFormmat\(\"([^\"]*)\"", vals))
            rows[period] = {f"fc_{n}": pd.to_numeric(v.get(c), errors="coerce") for c, n in FUEL_COST_COLS.items()}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.to_datetime(out.index, format="%Y/%m")
    out = out.sort_index().rename_axis("month").reset_index()
    out["usable_from"] = out["month"] + pd.DateOffset(months=2)
    return out


GAS_BACKFILL_UNTIL = "2026-11"             # 이 달까지는 2026-11-03 일괄 수집 → first_seen이 실제 게시 시각이 아님
GAS_LATE = {"2021-12", "2023-12"}         # 그달 시작 뒤 게시된 것으로 보이는 달 (연도 파일 수정 시각 기준)


def load_gas_tariff() -> pd.DataFrame:
    """data/raw/api/kogas_gas_tariff → [month, won_gj, won_kwh, usable_from].

    가스공사 발전용 천연가스 요금(일반발전 합계, 원/GJ). won_kwh = 원/GJ × 7.2 MJ/kWh ÷ 1000 (EPSIS LNG 연료비단가와 같은 단위 근사).
    usable_from(이 날부터 판단에 써도 됨): 일괄 수집분은 전달 28일(늦게 게시된 달은 그달 1일),
    이후 매일 수집분은 처음 보인 날의 다음 날.
    """
    t = pd.read_csv(API_RAW_DIR / "kogas_gas_tariff" / "first_seen.csv", dtype={"month": str})
    month = pd.to_datetime(t["month"], format="%Y-%m")
    rule = (month - pd.DateOffset(months=1)) + pd.DateOffset(days=27)
    rule = rule.where(~t["month"].isin(GAS_LATE), month)
    seen = pd.to_datetime(t["first_seen"]).dt.normalize() + pd.Timedelta(days=1)
    out = pd.DataFrame({"month": month, "won_gj": t["won_gj"].astype(float)})
    out["won_kwh"] = out["won_gj"] * 7.2 / 1000
    out["usable_from"] = rule.where(t["month"] <= GAS_BACKFILL_UNTIL, seen)
    return out.sort_values("month").reset_index(drop=True)


# 원전 호기별 설비용량(MW). 합계 26,050MW = 시장참여 설비용량 파일의 원자력 합계(2024-01 ~ 2026-02)와 일치
NUCLEAR_MW = {
    "고리2호기": 650, "고리3호기": 950, "고리4호기": 950, "신고리1호기": 1000, "신고리2호기": 1000,
    "새울1호기": 1400, "새울2호기": 1400,
    "한빛1호기": 950, "한빛2호기": 950, "한빛3호기": 1000, "한빛4호기": 1000, "한빛5호기": 1000, "한빛6호기": 1000,
    "한울1호기": 950, "한울2호기": 950, "한울3호기": 1000, "한울4호기": 1000, "한울5호기": 1000, "한울6호기": 1000,
    "월성2호기": 700, "월성3호기": 700, "월성4호기": 700, "신월성1호기": 1000, "신월성2호기": 1000,
    "신한울1호기": 1400, "신한울2호기": 1400,
}


def load_nuclear_available(start="2023-01-01", end="2027-12-31") -> pd.DataFrame:
    """한수원 계획예방정비 현황(~2025, 실적)·계획(2026, 2026-06-09 기준) → [date, nuc_total_mw, nuc_off_mw, nuc_avail_mw].

    정비 일정은 사전에 공개되는 계획이지만, 2025년 이전은 실제 종료일 기준이라 연장분은 약간 앞서 아는 셈이다.
    총용량: 2026-03부터 새울3호기 +1,400MW (시장참여 설비용량 파일 기준).
    """
    files = {"현황": "한국수력원자력(주)_원전 호기별 계획예방정비 현황_20260609.csv",
             "계획": "한국수력원자력(주)_가동원전 호기별 계획예방정비 계획_20260609.csv"}
    o = pd.concat([pd.read_csv(RAW_DIR / f, encoding="cp949")[["호기", "차수", "시작일", "종료일"]] for f in files.values()])
    o = o.drop_duplicates(["호기", "차수"], keep="last")
    o["시작일"], o["종료일"] = pd.to_datetime(o["시작일"]), pd.to_datetime(o["종료일"])
    days = pd.date_range(start, end)
    off = pd.Series(0.0, index=days)
    for _, r in o.iterrows():
        if r["호기"] in NUCLEAR_MW and pd.notna(r["시작일"]):
            e = r["종료일"] if pd.notna(r["종료일"]) else pd.Timestamp(end)
            off[(days >= r["시작일"]) & (days < e)] += NUCLEAR_MW[r["호기"]]
    total = pd.Series(float(sum(NUCLEAR_MW.values())), index=days)
    total[days >= "2026-03-01"] += 1400
    return pd.DataFrame({"date": days, "nuc_total_mw": total.values, "nuc_off_mw": off.values,
                         "nuc_avail_mw": (total - off).values})


def load_market_capacity() -> pd.DataFrame:
    """전력거래소 시장참여 설비용량(전력시장·PPA) 연료원별 월별 → 지역 '소계' 행 [month, cap_nuclear, cap_coal, cap_lng, cap_solar, cap_wind, cap_total, usable_from].

    단위 MW. 공표 시점 미확인 → M월 값은 M+2월 1일부터 사용.
    """
    c = pd.read_csv(RAW_DIR / "HOME_전력거래_시장참여설비용량(전력시장·PPA)_연료원별.csv", encoding="cp949").iloc[1:]
    c.columns = ["기간", "지역", "원자력", "유연탄", "무연탄", "석탄계", "유류", "LNG", "양수", "연료전지", "석탄가스화",
                 "태양", "풍력", "수력", "해양", "바이오", "폐기물", "신재생계", "기타", "계"]
    c = c[c["지역"] == "소계"]
    out = pd.DataFrame({"month": pd.to_datetime(c["기간"], format="%Y/%m")})
    for k, col in {"cap_nuclear": "원자력", "cap_coal": "석탄계", "cap_lng": "LNG", "cap_solar": "태양",
                   "cap_wind": "풍력", "cap_total": "계"}.items():
        out[k] = pd.to_numeric(c[col], errors="coerce").values
    # 매일 수집분(common.collect.epsis_capacity, 같은 값)이 있으면 최신 파일로 덮어씀
    files = sorted((API_RAW_DIR / "epsis_capacity").glob("gcp_*.txt"))
    if files:
        import re
        rows = []
        text = files[-1].read_text(encoding="utf-8")
        for chunk in text.split("gridData.push")[:-1]:
            c = dict(re.findall(r'(c\d+) = textFormmat\("([^"]*)"', chunk))
            yr = re.findall(r'year = "([0-9/]+)";\s*\}', chunk)
            if yr and "c8" in c:
                rows.append({"month": pd.to_datetime(yr[-1], format="%Y/%m"),
                             **{k: pd.to_numeric(c.get(v), errors="coerce") for k, v in
                                {"cap_nuclear": "c1", "cap_coal": "c23", "cap_lng": "c5", "cap_solar": "c93",
                                 "cap_wind": "c94", "cap_total": "c8"}.items()}})
        if rows:
            api = pd.DataFrame(rows)
            out = pd.concat([out[~out["month"].isin(api["month"])], api])
    out = out.sort_values("month").reset_index(drop=True)
    out["usable_from"] = out["month"] + pd.DateOffset(months=2)
    return out


def load_land_curtailment() -> pd.DataFrame:
    """육지 재생에너지 출력제어 시행일 → [date, curt_solar, curt_wind, curt_hours].

    출처: 사용자 제공 공공데이터 CSV(태양광·풍력, ~2026-06-13) + 전력거래소 '비중앙 출력제어 정보' 게시판 실적 제목
    (data/raw/api/kpx_curtailment, 2024-04 ~ 2026-07-12). 게시판 실적은 약 1개월 늦게 올라오므로 최근 구간은 누락 가능.
    """
    def read(name):
        d = pd.read_csv(RAW_DIR / name, encoding="cp949")
        hrs = [c for c in d.columns if c.endswith("시")]
        return pd.DataFrame({"date": pd.to_datetime(d["일자"]), "hours": (d[hrs] == "출력제어").sum(axis=1)})
    s = read("육지 태양광 출력제어횟수_26년상반기.csv").assign(curt_solar=1)
    w = read("육지 풍력 출력제어횟수_26년 상반기.csv").assign(curt_wind=1)
    out = pd.merge(s, w, on="date", how="outer", suffixes=("_s", "_w"))
    out["curt_hours"] = out[["hours_s", "hours_w"]].max(axis=1)
    b = pd.read_csv(API_RAW_DIR / "kpx_curtailment" / "board_land_20261019.csv", parse_dates=["target"])
    extra = b.loc[(b["kind"] == "실적") & ~b["target"].isin(out["date"]), ["target"]].rename(columns={"target": "date"})
    out = pd.concat([out, extra.assign(curt_solar=1)], ignore_index=True)
    return out[["date", "curt_solar", "curt_wind", "curt_hours"]].fillna(0).sort_values("date").reset_index(drop=True)


def load_kpx_power_source() -> pd.DataFrame:
    """전력거래소 발전원별 5분 실적 엑셀 → [ts, date, solar_btm, solar_ppa, solar_mkt, pump, hydro, gas, wind, renew,
    oil, coal_dom, coal_bit, nuclear] (MW). BTM·PPA 추정은 2024-11-23부터(그 전 0). '-'는 결측."""
    cols = ["date", "time", "solar_btm", "solar_ppa", "solar_mkt", "pump", "hydro", "gas", "wind", "renew", "ess",
            "renew_x", "new_etc", "oil", "coal_dom", "coal_bit", "coal", "nuclear"]
    frames = []
    for p in sorted((API_RAW_DIR / "kpx_power_source").glob("ps_*.xlsx")):
        d = pd.read_excel(p, header=None, skiprows=2, names=cols)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["ts"] = pd.to_datetime(d["date"].astype(str) + " " + d["time"].astype(str), errors="coerce")
    d = d.dropna(subset=["ts"]).drop_duplicates("ts")
    num = [c for c in cols[2:]]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d["date"] = d["ts"].dt.normalize()
    return d[["ts", "date", "solar_btm", "solar_ppa", "solar_mkt", "pump", "hydro", "gas", "wind", "renew",
              "oil", "coal_dom", "coal_bit", "nuclear"]].sort_values("ts").reset_index(drop=True)


def load_openmeteo_prev2() -> pd.DataFrame:
    """Open-Meteo 2일 전 실행 예보 → [date, model, rad_mid, cloud_mid]: 7개 도시 평균, 11~15시(시작 시각 11~14시) 일사 합(Wh/m²)·운량 평균."""
    import json
    rows = []
    for p in sorted((API_RAW_DIR / "openmeteo").glob("prev2_*.json")):
        model = p.name.split("_")[1] + ("_" + p.name.split("_")[2] if p.name.split("_")[2] in ("seamless", "ifs025") else "")
        model = model.replace("_seamless", "").replace("_ifs025", "")
        data = json.loads(p.read_text(encoding="utf-8"))
        for loc in (data if isinstance(data, list) else [data]):
            h = pd.DataFrame(loc["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h = h[h["time"].dt.hour.between(11, 14)]
            g = h.groupby(h["time"].dt.normalize()).agg(rad=("shortwave_radiation_previous_day2", "sum"),
                                                       cloud=("cloud_cover_previous_day2", "mean"),
                                                       n=("shortwave_radiation_previous_day2", "count"))
            g.loc[g["n"] < 4, ["rad", "cloud"]] = np.nan
            rows.append(g.assign(model=model))
    d = pd.concat(rows).rename_axis("date").reset_index()
    return d.groupby(["date", "model"]).agg(rad_mid=("rad", "mean"), cloud_mid=("cloud", "mean")).reset_index()


def load_kma_afs_jeju() -> pd.DataFrame:
    """data/raw/api/kma_afs의 제주 구역(JEJU_REGIONS) 단기 육상예보 → [reg, tm_fc, tm_ef, wd1, wd_tr, wd2, ta, pop, sky, rain, wf].

    육지용 load_kma_regional_fcst()와 열 위치는 같고 풍향(wd1·wd2: 16방위, wd_tr: 풍향 경향 코드)을 더 읽는다.
    원문에 두 레코드가 한 줄로 붙은 경우(월 경계)가 있어 레코드 시작(구역코드 + 발표시각)으로 다시 나눈다.
    """
    import re
    from common.collect.kma_regional_fcst import JEJU_REGIONS
    start = re.compile(r"(?=\b(?:%s) \d{12} \d{12})" % "|".join(JEJU_REGIONS))
    rows = []
    for reg in JEJU_REGIONS:
        for p in sorted((API_RAW_DIR / "kma_afs").glob(f"afs_{reg}_*.txt")):
            for rec in start.split(p.read_text(encoding="utf-8")):
                parts = rec.split()
                if len(parts) < 17:
                    continue
                rows.append({"reg": parts[0], "tm_fc": parts[1], "tm_ef": parts[2], "wd1": parts[9], "wd_tr": parts[10],
                             "wd2": parts[11], "ta": parts[12], "pop": parts[13], "sky": parts[14], "rain": parts[15],
                             "wf": " ".join(parts[16:]).strip('"')})
    df = pd.DataFrame(rows)
    df["tm_fc"] = pd.to_datetime(df["tm_fc"], format="%Y%m%d%H%M")
    df["tm_ef"] = pd.to_datetime(df["tm_ef"], format="%Y%m%d%H%M")
    df["ta"] = pd.to_numeric(df["ta"], errors="coerce").replace(-99, np.nan)
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
    df["sky"] = df["sky"].str[-1].astype(int)
    df["rain"] = (pd.to_numeric(df["rain"], errors="coerce") > 0).astype(int)
    return df.drop_duplicates(["reg", "tm_fc", "tm_ef"]).sort_values(["reg", "tm_fc", "tm_ef"]).reset_index(drop=True)


def load_openmeteo_prev2_jeju() -> pd.DataFrame:
    """data/raw/api/openmeteo_jeju (제주 4지점, 2일 전 실행 예보) → 시간별 [time, model, point, rad, cloud, ws10, ws100].

    time은 시작 시각(Asia/Seoul): HH시 값 = HH:00 ~ HH+1:00 구간(일사 Wh/m²) 또는 그 시각(운량 %, 풍속 km/h).
    육지 load_openmeteo_prev2()와 폴더가 달라 서로 섞이지 않는다. JMA는 100m 풍속 없음(NaN).
    """
    import json
    from common.collect.openmeteo_prevruns import JEJU_CITIES
    names = list(JEJU_CITIES)
    rows = []
    for p in sorted((API_RAW_DIR / "openmeteo_jeju").glob("prev2_*.json")):
        model = p.name.split("_")[1]
        data = json.loads(p.read_text(encoding="utf-8"))
        for name, loc in zip(names, data if isinstance(data, list) else [data]):
            h = pd.DataFrame(loc["hourly"]).rename(columns={
                "shortwave_radiation_previous_day2": "rad", "cloud_cover_previous_day2": "cloud",
                "wind_speed_10m_previous_day2": "ws10", "wind_speed_100m_previous_day2": "ws100"})
            h["time"] = pd.to_datetime(h["time"])
            rows.append(h.assign(model=model, point=name, file_end=p.stem.split("_")[-1]))
    d = pd.concat(rows, ignore_index=True)
    # 기간이 겹치는 파일은 나중(끝 날짜가 늦은) 파일 값을 쓴다
    d = d.sort_values("file_end").drop_duplicates(["time", "model", "point"], keep="last").drop(columns="file_end")
    return d[["time", "model", "point", "rad", "cloud", "ws10", "ws100"]].sort_values(["model", "point", "time"]).reset_index(drop=True)


def load_openmeteo_prevx_jeju() -> pd.DataFrame:
    """data/raw/api/openmeteo_jeju/prevx_*.json (제주 4지점, k=3~7일 전 실행 예보) → 시간별 [time, model, point, k, rad, cloud, ws10, ws100].

    H일 뒤 예측(판단일 D-H 15시)에는 k = H+1일 전 실행분만 쓴다 (2일 전 실행분을 H=1에만 쓰는 규칙과 같음).
    """
    import json
    from common.collect.openmeteo_prevruns import JEJU_CITIES
    names = list(JEJU_CITIES)
    rows = []
    for p in sorted((API_RAW_DIR / "openmeteo_jeju").glob("prevx_*.json")):
        model = p.name.split("_")[1]
        data = json.loads(p.read_text(encoding="utf-8"))
        for name, loc in zip(names, data if isinstance(data, list) else [data]):
            h = pd.DataFrame(loc["hourly"])
            t = pd.to_datetime(h["time"])
            for k in range(3, 8):
                part = pd.DataFrame({"time": t, "rad": h[f"shortwave_radiation_previous_day{k}"],
                                     "cloud": h[f"cloud_cover_previous_day{k}"], "ws10": h[f"wind_speed_10m_previous_day{k}"],
                                     "ws100": h[f"wind_speed_100m_previous_day{k}"]})
                if part[["rad", "ws10"]].notna().any().any():
                    rows.append(part.assign(model=model, point=name, k=k, file_end=p.stem.split("_")[-1]))
    d = pd.concat(rows, ignore_index=True)
    d = d.sort_values("file_end").drop_duplicates(["time", "model", "point", "k"], keep="last").drop(columns="file_end")
    return d[["time", "model", "point", "k", "rad", "cloud", "ws10", "ws100"]].sort_values(["k", "model", "point", "time"]).reset_index(drop=True)


def load_kpx_jeju_sukub() -> pd.DataFrame:
    """data/raw/api/kpx_jeju_sukub (제주계통운영정보 5분 실적) → [time, supply, demand, renew, solar, wind] (MW).

    time = 5분 시각(KST). demand = 발전단 수요(currPwrTot). 송전단 수요(currNtPwrTot)는 2026-06 이전 0이라 읽지 않는다.
    과거분은 사용자 내려받기 파일 data/raw/chejusukub*.csv (cp949, 2024-01 ~ 2026-10-23 09:15)와
    제주계통운영정보_*.csv (같은 형식, 2023년)를 함께 읽고, 겹치는 시각은 파일 값을 쓴다.
    (2024-10-01 ~ 2025-06-24는 API 수요가 파일보다 30~60MW 낮음. 파일 쪽이 하루전 수요예측 대비 비율이 경계에서 튀지 않아 일관됨)
    공급능력·수요 0 이하(2026년 41개 등)는 결측 처리.
    """
    files = sorted((API_RAW_DIR / "kpx_jeju_sukub").glob("jeju5m_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype={"baseDatetime": str}) for p in files], ignore_index=True)
    out = pd.DataFrame({"time": pd.to_datetime(df["baseDatetime"], format="%Y%m%d%H%M%S"),
                        "supply": pd.to_numeric(df["suppAbility"], errors="coerce"),
                        "demand": pd.to_numeric(df["currPwrTot"], errors="coerce"),
                        "renew": pd.to_numeric(df["renewPwrTot"], errors="coerce"),
                        "solar": pd.to_numeric(df["renewPwrSolar"], errors="coerce"),
                        "wind": pd.to_numeric(df["renewPwrWind"], errors="coerce")})
    paths = sorted(RAW_DIR.glob("chejusukub*.csv")) + sorted(RAW_DIR.glob("제주계통운영정보_*.csv"))
    dl = [pd.read_csv(p, encoding="cp949", dtype=str) for p in paths]
    if dl:
        d = pd.concat(dl, ignore_index=True)
        d.columns = ["time", "supply", "demand", "renew", "solar", "wind"]
        d["time"] = pd.to_datetime(d["time"], format="%Y%m%d%H%M%S")
        d[d.columns[1:]] = d[d.columns[1:]].apply(pd.to_numeric, errors="coerce")
        out = pd.concat([d, out], ignore_index=True)   # 파일 먼저 → 겹치면 파일 값 유지
    out = out.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    out.loc[out["supply"] <= 0, "supply"] = np.nan
    out.loc[out["demand"] <= 0, ["demand", "renew", "solar", "wind"]] = np.nan
    return out
