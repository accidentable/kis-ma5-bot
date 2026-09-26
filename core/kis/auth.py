"""
core/kis/auth.py — 접근토큰 발급 / 캐싱

한투 토큰 규칙:
  - 유효기간 24시간
  - 재발급 요청은 1분에 1회로 제한된다. 여러 프로세스(람다)가 각자 발급하면
    EGW00133(초당 거래건수 초과) 이나 발급 제한에 걸리므로 반드시 캐시를 공유한다.

캐시 백엔드:
  file  로컬 JSON  — 단일 프로세스/로컬 테스트용
  ssm   SSM Parameter Store (SecureString) — 람다 간 공유용
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

_TOKEN_URL = f"{config.KIS_BASE_URL}/oauth2/tokenP"
_REVOKE_URL = f"{config.KIS_BASE_URL}/oauth2/revokeP"
_HASHKEY_URL = f"{config.KIS_BASE_URL}/uapi/hashkey"

# 만료 1시간 전에 미리 갱신
_REFRESH_MARGIN_SEC = 3600

_lock = threading.Lock()
_mem: dict[str, float | str] = {"token": "", "expires_at": 0.0}


# ── 캐시 백엔드 ───────────────────────────────────────────────
def _file_path() -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return os.path.join(config.DATA_DIR, "kis_token.json")


def _cache_read() -> tuple[str, float]:
    if config.TOKEN_CACHE == "ssm":
        try:
            import boto3
            ssm = boto3.client("ssm", region_name=config.AWS_REGION)
            raw = ssm.get_parameter(Name=config.SSM_TOKEN_PATH, WithDecryption=True)["Parameter"]["Value"]
            data = json.loads(raw)
            return data.get("token", ""), float(data.get("expires_at", 0))
        except Exception as e:
            logger.warning("SSM 토큰 캐시 읽기 실패: %s", e)
            return "", 0.0
    try:
        with open(_file_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data.get("token", ""), float(data.get("expires_at", 0))
    except (OSError, ValueError, KeyError):
        return "", 0.0


def _cache_write(token: str, expires_at: float) -> None:
    payload = json.dumps({"token": token, "expires_at": expires_at})
    if config.TOKEN_CACHE == "ssm":
        try:
            import boto3
            ssm = boto3.client("ssm", region_name=config.AWS_REGION)
            ssm.put_parameter(
                Name=config.SSM_TOKEN_PATH, Value=payload,
                Type="SecureString", Overwrite=True,
            )
            return
        except Exception as e:
            logger.error("SSM 토큰 캐시 쓰기 실패: %s", e)
            return
    try:
        with open(_file_path(), "w", encoding="utf-8") as f:
            f.write(payload)
    except OSError as e:
        logger.error("토큰 파일 쓰기 실패: %s", e)


# ── 발급 ─────────────────────────────────────────────────────
def _issue() -> tuple[str, float]:
    body = {
        "grant_type": "client_credentials",
        "appkey": config.KIS_APP_KEY,
        "appsecret": config.KIS_APP_SECRET,
    }
    resp = requests.post(_TOKEN_URL, json=body, timeout=15)

    if resp.status_code != 200:
        # 한투는 발급 제한/키 오류를 본문에 담아 403 으로 준다.
        raise RuntimeError(f"토큰 발급 실패 HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    token = data.get("access_token", "")
    if not token:
        raise RuntimeError(f"토큰 발급 응답에 access_token 없음: {data}")

    # expires_in(초) 우선, 없으면 24시간
    expires_in = int(data.get("expires_in", 86400) or 86400)
    expires_at = time.time() + expires_in
    logger.info("KIS 토큰 발급 완료 (유효 %d초)", expires_in)
    return token, expires_at


def get_token(force: bool = False) -> str:
    """유효한 access_token 을 반환한다. 메모리 → 캐시 → 신규발급 순."""
    now = time.time()

    with _lock:
        if not force:
            tok = str(_mem["token"])
            if tok and float(_mem["expires_at"]) - now > _REFRESH_MARGIN_SEC:
                return tok

            tok, exp = _cache_read()
            if tok and exp - now > _REFRESH_MARGIN_SEC:
                _mem["token"], _mem["expires_at"] = tok, exp
                return tok

        token, expires_at = _issue()
        _mem["token"], _mem["expires_at"] = token, expires_at
        _cache_write(token, expires_at)
        return token


def build_headers(tr_id: str, hashkey: Optional[str] = None, tr_cont: str = "") -> dict[str, str]:
    """한투 REST 공통 헤더."""
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {get_token()}",
        "appkey": config.KIS_APP_KEY,
        "appsecret": config.KIS_APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",  # 개인
    }
    if tr_cont:
        headers["tr_cont"] = tr_cont
    if hashkey:
        headers["hashkey"] = hashkey
    return headers


def get_hashkey(body: dict) -> str:
    """주문 body 의 무결성 해시. POST 주문 시 헤더에 넣는다."""
    headers = {
        "content-type": "application/json; charset=utf-8",
        "appkey": config.KIS_APP_KEY,
        "appsecret": config.KIS_APP_SECRET,
    }
    resp = requests.post(_HASHKEY_URL, json=body, headers=headers, timeout=10)
    resp.raise_for_status()
    h = resp.json().get("HASH", "")
    if not h:
        raise RuntimeError(f"hashkey 응답 이상: {resp.text[:200]}")
    return h
