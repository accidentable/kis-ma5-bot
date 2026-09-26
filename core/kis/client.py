"""
core/kis/client.py — 한투 REST 호출 공통 래퍼

책임:
  - 유량제한(토큰버킷). 실전 조회는 초당 20건이지만 여유를 둔다.
  - 토큰 만료(EGW00123) 자동 감지 후 재발급 → 1회 재시도
  - 일시적 오류(5xx, 타임아웃) 지수 백오프 재시도
  - rt_cd != '0' 을 예외로 승격해 호출부가 성공만 다루게 한다
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Optional

import requests

import config
from core.kis import auth

logger = logging.getLogger(__name__)


class KisError(RuntimeError):
    """한투 API 가 rt_cd != '0' 으로 응답한 경우."""

    def __init__(self, msg_cd: str, msg: str, tr_id: str, body: Any = None):
        super().__init__(f"[{tr_id}] {msg_cd}: {msg}")
        self.msg_cd = msg_cd
        self.msg = msg
        self.tr_id = tr_id
        self.body = body


# 토큰 만료 계열 오류코드
_TOKEN_ERRORS = {"EGW00123", "EGW00121", "EGW00105"}
# 유량 초과
_RATE_ERRORS = {"EGW00133", "EGW00201"}


class _RateLimiter:
    """
    초당 N건으로 제한하는 토큰버킷.

    한투의 실제 유량 한도는 공지된 값보다 빡빡하게 걸릴 때가 있어서,
    EGW00201(초당 거래건수 초과)을 만나면 스스로 속도를 낮추고 그 상태를 유지한다.
    고정 backoff 만으로는 한도를 다시 치기만 반복하게 된다.
    """

    _MIN_RATE = 0.5   # 이 아래로는 안 내린다
    _MAX_INTERVAL = 1.0 / _MIN_RATE

    def __init__(self, per_sec: float):
        self._base_interval = 1.0 / max(per_sec, 0.1)
        self._interval = self._base_interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = now + self._interval

    def slow_down(self) -> float:
        """유량 오류를 만났을 때 호출. 간격을 1.8배로 늘린다."""
        with self._lock:
            self._interval = min(self._interval * 1.8, self._MAX_INTERVAL)
            return 1.0 / self._interval

    @property
    def rate(self) -> float:
        return 1.0 / self._interval


_limiter = _RateLimiter(config.KIS_RATE_LIMIT_PER_SEC)
_session = requests.Session()
_order_lock = threading.Lock()
_last_order_at = 0.0


def _throttle_order() -> None:
    """주문 API 는 초당 1건. 주문 간 최소 간격을 강제한다."""
    global _last_order_at
    with _order_lock:
        gap = time.monotonic() - _last_order_at
        if gap < config.KIS_ORDER_INTERVAL_SEC:
            time.sleep(config.KIS_ORDER_INTERVAL_SEC - gap)
        _last_order_at = time.monotonic()


def request(
    method: str,
    path: str,
    tr_id: str,
    *,
    params: Optional[dict] = None,
    body: Optional[dict] = None,
    tr_cont: str = "",
    use_hashkey: bool = False,
    is_order: bool = False,
    max_retries: int = 3,
) -> dict:
    """
    한투 API 호출. 성공 시 응답 JSON(dict) 전체를 반환한다.
    응답 헤더의 tr_cont/연속조회 키는 '_headers' 로 끼워 넣는다.
    """
    url = f"{config.KIS_BASE_URL}{path}"
    attempt = 0
    token_retried = False
    # 유량 초과는 일반 오류와 따로 센다. 속도를 낮추면 해결되는 문제라
    # 일반 재시도 횟수로 묶어버리면 조금만 붐벼도 스캔이 통째로 실패한다.
    rate_retries = 0
    max_rate_retries = 8

    while True:
        attempt += 1
        if is_order:
            _throttle_order()
        else:
            _limiter.acquire()

        hashkey = auth.get_hashkey(body) if (use_hashkey and body) else None
        headers = auth.build_headers(tr_id, hashkey=hashkey, tr_cont=tr_cont)

        try:
            if method.upper() == "GET":
                resp = _session.get(url, headers=headers, params=params, timeout=15)
            else:
                resp = _session.post(url, headers=headers, json=body, timeout=15)
        except requests.RequestException as e:
            if attempt <= max_retries:
                backoff = min(2 ** attempt, 8) + random.random()
                logger.warning("네트워크 오류(%s), %.1f초 후 재시도 %d/%d", e, backoff, attempt, max_retries)
                time.sleep(backoff)
                continue
            raise

        if resp.status_code >= 500:
            # 한투는 5xx 로 주면서 본문에 실제 원인(msg_cd)을 담는다.
            # 특히 유량 초과(EGW00201)가 500 으로 오기 때문에,
            # 상태코드만 보고 일반 재시도로 넘기면 원인을 영영 못 본다.
            body_hint = resp.text[:300].replace("\n", " ")
            body_cd = ""
            try:
                body_cd = str(resp.json().get("msg_cd", ""))
            except ValueError:
                pass

            if body_cd in _RATE_ERRORS:
                if rate_retries < max_rate_retries:
                    rate_retries += 1
                    new_rate = _limiter.slow_down()
                    logger.warning(
                        "유량 초과(%s) [%s] — 속도를 초당 %.1f건으로 낮추고 재시도 %d/%d",
                        body_cd, tr_id, new_rate, rate_retries, max_rate_retries,
                    )
                    time.sleep(1.0 + random.random())
                    continue
                raise KisError(body_cd, "초당 거래건수 초과 (재시도 소진)", tr_id)

            if attempt <= max_retries:
                backoff = min(2 ** attempt, 8) + random.random()
                logger.warning(
                    "HTTP %d [%s] %.1f초 후 재시도 %d/%d — 응답: %s",
                    resp.status_code, tr_id, backoff, attempt, max_retries, body_hint,
                )
                time.sleep(backoff)
                continue
            raise RuntimeError(f"[{tr_id}] HTTP {resp.status_code}: {body_hint}")

        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"[{tr_id}] JSON 파싱 실패 HTTP {resp.status_code}: {resp.text[:300]}")

        msg_cd = str(data.get("msg_cd", ""))

        # 토큰 만료 → 강제 재발급 후 딱 한 번 재시도
        if msg_cd in _TOKEN_ERRORS and not token_retried:
            logger.info("토큰 만료(%s) 감지 — 재발급", msg_cd)
            auth.get_token(force=True)
            token_retried = True
            continue

        # 유량 초과가 200 으로 오는 경우도 있다. 여기서도 속도를 낮춘다.
        if msg_cd in _RATE_ERRORS:
            if rate_retries < max_rate_retries:
                rate_retries += 1
                new_rate = _limiter.slow_down()
                logger.warning(
                    "유량 초과(%s) [%s] — 속도를 초당 %.1f건으로 낮추고 재시도 %d/%d",
                    msg_cd, tr_id, new_rate, rate_retries, max_rate_retries,
                )
                time.sleep(1.0 + random.random())
                continue
            raise KisError(msg_cd, "초당 거래건수 초과 (재시도 소진)", tr_id)

        if str(data.get("rt_cd", "0")) != "0":
            raise KisError(msg_cd, str(data.get("msg1", "")).strip(), tr_id, data)

        data["_headers"] = {
            "tr_cont": resp.headers.get("tr_cont", ""),
        }
        return data


def get(path: str, tr_id: str, params: dict, **kw) -> dict:
    return request("GET", path, tr_id, params=params, **kw)


def post(path: str, tr_id: str, body: dict, **kw) -> dict:
    return request("POST", path, tr_id, body=body, **kw)


def paginate(
    path: str,
    tr_id: str,
    params: dict,
    *,
    output_key: str = "output",
    max_pages: int = 20,
) -> list[dict]:
    """
    연속조회(tr_cont F/M) 를 끝까지 따라가며 결과를 모은다.
    CTX_AREA_FK100/NK100 를 쓰는 계좌성 조회에 사용.
    """
    rows: list[dict] = []
    tr_cont = ""
    cur = dict(params)

    for _ in range(max_pages):
        data = get(path, tr_id, cur, tr_cont=tr_cont)
        chunk = data.get(output_key) or []
        if isinstance(chunk, dict):
            chunk = [chunk]
        rows.extend(chunk)

        next_cont = data["_headers"]["tr_cont"]
        if next_cont not in ("F", "M"):
            break
        tr_cont = "N"

        # 연속조회 키 이름이 API마다 다르다.
        # 계좌성 조회는 CTX_AREA_FK100/NK100, 휴장일 조회 등은 CTX_AREA_FK/NK 를 쓴다.
        # 여기서 키를 안 넘기면 서버가 같은 요청으로 보고 SYDB0050(자료 변경됨) 을 낸다.
        advanced = False
        for fk, nk in (("CTX_AREA_FK100", "CTX_AREA_NK100"), ("CTX_AREA_FK", "CTX_AREA_NK")):
            if fk in cur:
                cur[fk] = str(data.get(fk.lower(), "")).strip()
                cur[nk] = str(data.get(nk.lower(), "")).strip()
                advanced = True

        # 다음 키를 못 받았으면 같은 페이지를 무한히 다시 부르게 되므로 멈춘다
        if not advanced or not any(cur.get(k) for k in ("CTX_AREA_NK100", "CTX_AREA_NK")):
            break

    return rows
