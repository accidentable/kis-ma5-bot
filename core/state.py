"""
core/state.py — 봇 상태 저장소

포지션·매매이력·일시정지 플래그를 담는다. 한투 잔고 조회로 보유 수량은 알 수 있지만
"언제 몇 거래일째 들고 있는지", "진입 당시 5일선이 얼마였는지" 는 봇만 아는 정보라
별도로 남겨야 한다.

백엔드
  local      JSON 파일 — 로컬/단일 프로세스
  dynamodb   단일 아이템(state) — 람다 간 공유
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

_DEFAULT: dict[str, Any] = {
    "positions": [],          # 보유 포지션
    "history": [],            # 청산된 거래
    "paused": False,          # 자동매매 일시정지
    "last_entry_date": "",
    "daily_candidates": {},   # {date, items[]} — 09:05 1차 통과 종목 (장중 재진입용 캐시)
    "traded_today": {},       # {date, tickers[]} — 당일 매수·매도한 종목 (재진입 금지)
    "updated_at": "",
}

_lock = threading.Lock()
_DDB_KEY = {"pk": "state"}


def _file_path() -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return os.path.join(config.DATA_DIR, "state.json")


def _read() -> dict:
    if config.STATE_BACKEND == "dynamodb":
        try:
            import boto3
            table = boto3.resource("dynamodb", region_name=config.AWS_REGION).Table(config.DYNAMODB_TABLE)
            item = table.get_item(Key=_DDB_KEY).get("Item")
            if item and "payload" in item:
                return json.loads(item["payload"])
        except Exception as e:
            logger.error("DynamoDB 상태 읽기 실패: %s", e)
        return json.loads(json.dumps(_DEFAULT))

    try:
        with open(_file_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return json.loads(json.dumps(_DEFAULT))


def _write(data: dict) -> None:
    data["updated_at"] = date.today().isoformat()
    payload = json.dumps(data, ensure_ascii=False)

    if config.STATE_BACKEND == "dynamodb":
        try:
            import boto3
            table = boto3.resource("dynamodb", region_name=config.AWS_REGION).Table(config.DYNAMODB_TABLE)
            table.put_item(Item={**_DDB_KEY, "payload": payload})
            return
        except Exception as e:
            logger.error("DynamoDB 상태 저장 실패: %s", e)
            return

    try:
        path = _file_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)  # 쓰다 죽어도 이전 상태가 남도록
    except OSError as e:
        logger.error("상태 파일 저장 실패: %s", e)


def load() -> dict:
    with _lock:
        data = _read()
        for k, v in _DEFAULT.items():
            data.setdefault(k, json.loads(json.dumps(v)))
        return data


def save(data: dict) -> None:
    with _lock:
        _write(data)


# ══════════════════════════════════════════════════════════════
# 포지션
# ══════════════════════════════════════════════════════════════
def get_positions() -> list[dict]:
    return load()["positions"]


def get_position(ticker: str) -> Optional[dict]:
    return next((p for p in get_positions() if p["ticker"] == ticker), None)


def add_position(pos: dict) -> None:
    data = load()
    data["positions"] = [p for p in data["positions"] if p["ticker"] != pos["ticker"]]
    data["positions"].append(pos)
    data["last_entry_date"] = pos.get("entry_date", date.today().isoformat())
    save(data)
    logger.info("포지션 기록: %s %d주 @%s", pos["ticker"], pos["qty"], pos["entry_price"])


def update_position(ticker: str, **fields) -> Optional[dict]:
    data = load()
    for p in data["positions"]:
        if p["ticker"] == ticker:
            p.update(fields)
            save(data)
            return p
    return None


def remove_position(ticker: str) -> bool:
    """포지션을 이력에 남기지 않고 지운다. 한 주도 안 채워진 분할 주문을 정리할 때 쓴다."""
    data = load()
    before = len(data["positions"])
    data["positions"] = [p for p in data["positions"] if p["ticker"] != ticker]
    if len(data["positions"]) == before:
        return False
    save(data)
    logger.info("포지션 제거 (미체결): %s", ticker)
    return True


def close_position(ticker: str, exit_price: float, reason: str, qty: Optional[int] = None) -> Optional[dict]:
    """포지션을 이력으로 옮긴다."""
    data = load()
    pos = next((p for p in data["positions"] if p["ticker"] == ticker), None)
    if pos is None:
        return None

    entry = float(pos.get("entry_price", 0) or 0)
    sold = qty if qty is not None else int(pos.get("qty", 0))
    pnl_pct = (exit_price / entry - 1) * 100 if entry > 0 else 0.0

    record = {
        **pos,
        "exit_price": exit_price,
        "exit_date": date.today().isoformat(),
        "exit_reason": reason,
        "sold_qty": sold,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_amount": round((exit_price - entry) * sold),
    }

    data["positions"] = [p for p in data["positions"] if p["ticker"] != ticker]
    data["history"].append(record)
    data["history"] = data["history"][-200:]
    save(data)

    logger.info("포지션 청산: %s %+.2f%% (%s)", ticker, pnl_pct, reason)
    return record


# ══════════════════════════════════════════════════════════════
# 운영 플래그
# ══════════════════════════════════════════════════════════════
def is_paused() -> bool:
    return bool(load().get("paused", False))


def set_paused(value: bool) -> None:
    data = load()
    data["paused"] = value
    save(data)
    logger.info("자동매매 %s", "일시정지" if value else "재개")


def get_history(limit: int = 20) -> list[dict]:
    return load()["history"][-limit:]


# ══════════════════════════════════════════════════════════════
# 당일 캐시 — 장중 재진입용
# ══════════════════════════════════════════════════════════════
def _today() -> str:
    return date.today().isoformat()


def get_daily_candidates() -> Optional[list[dict]]:
    """
    오늘 09:05 스캔에서 1차 통과한 종목 목록(직렬화된 Candidate).
    날짜가 다르면 None — 어제 캐시로 오늘 판정하면 5일선 기준가가 틀어진다.
    """
    cache = load().get("daily_candidates") or {}
    if cache.get("date") != _today():
        return None
    return list(cache.get("items") or [])


def set_daily_candidates(items: list[dict]) -> None:
    data = load()
    data["daily_candidates"] = {"date": _today(), "items": items}
    save(data)
    logger.info("당일 후보 캐시 저장: %d종목", len(items))


def get_traded_today() -> set[str]:
    """오늘 매수 또는 매도한 종목. 날짜가 바뀌면 빈 집합."""
    t = load().get("traded_today") or {}
    if t.get("date") != _today():
        return set()
    return set(t.get("tickers") or [])


def mark_traded_today(ticker: str) -> None:
    data = load()
    t = data.get("traded_today") or {}
    if t.get("date") != _today():
        t = {"date": _today(), "tickers": []}
    if ticker not in t["tickers"]:
        t["tickers"].append(ticker)
    data["traded_today"] = t
    save(data)
