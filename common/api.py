"""공공데이터포털(apis.data.go.kr) 호출 공통 함수."""
import time

import requests

from common.config import get_api_key


class ApiError(RuntimeError):
    pass


class QuotaExceeded(ApiError):
    """일일 호출 한도 초과 (재시도해도 소용없으므로 즉시 중단)."""


def fetch_items(url: str, params: dict, key_param: str = "serviceKey",
                retries: int = 3, timeout: int = 60) -> tuple[list[dict], int]:
    """한 페이지를 호출해 (item 목록, totalCount)를 돌려준다.

    data.go.kr 응답은 item이 1개면 dict, 0개면 빈 문자열로 오므로 항상 list로 맞춘다.
    """
    q = {key_param: get_api_key(), **params}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=q, timeout=timeout)
            if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in r.text:
                raise QuotaExceeded("일일 호출 한도 초과")
            r.raise_for_status()
            resp = r.json()["response"]
            code = resp["header"]["resultCode"]
            if code == "22":
                raise QuotaExceeded("일일 호출 한도 초과")
            if code not in ("00", "0"):
                # 03 = 데이터 없음
                if code == "03":
                    return [], 0
                raise ApiError(f"{code} {resp['header'].get('resultMsg')}")
            body = resp.get("body") or {}
            items = body.get("items") or {}
            item = items.get("item", []) if isinstance(items, dict) else []
            if isinstance(item, dict):
                item = [item]
            return item, int(body.get("totalCount") or 0)
        except QuotaExceeded:
            raise
        except (requests.RequestException, ValueError, KeyError, ApiError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise ApiError(f"호출 실패 ({url}, {params}): {last_err}")


def fetch_all(url: str, params: dict, num_rows: int = 1000, **kw) -> list[dict]:
    """totalCount만큼 페이지를 넘기며 전부 가져온다."""
    out, page = [], 1
    while True:
        items, total = fetch_items(url, {**params, "pageNo": page, "numOfRows": num_rows}, **kw)
        out.extend(items)
        if len(out) >= total or not items:
            return out
        page += 1
