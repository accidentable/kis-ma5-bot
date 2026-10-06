"""
core/notify.py — 텔레그램 알림

python-telegram-bot 을 쓰지 않고 Bot API 를 requests 로 직접 호출한다.
람다 패키지를 가볍게 유지하고, 알림 실패가 매매 로직을 중단시키지 않게 하기 위함이다.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"


def _call(method: str, payload: dict) -> Optional[dict]:
    if not config.TELEGRAM_BOT_TOKEN:
        logger.debug("텔레그램 토큰 없음 — 전송 생략")
        return None
    try:
        resp = requests.post(
            _API.format(token=config.TELEGRAM_BOT_TOKEN, method=method),
            json=payload, timeout=10,
        )
        data = resp.json()
        if not data.get("ok"):
            logger.warning("텔레그램 %s 실패: %s", method, data.get("description"))
        return data
    except requests.RequestException as e:
        # 알림 실패로 매매가 멈추면 안 된다.
        logger.warning("텔레그램 %s 예외: %s", method, e)
        return None


def send(text: str, chat_id: Optional[int] = None) -> None:
    """허용된 chat_id 전체(또는 지정한 하나)에게 메시지를 보낸다."""
    targets = [chat_id] if chat_id is not None else config.TELEGRAM_ALLOWED_CHAT_IDS
    if config.BOT_LABEL:
        text = f"[{config.BOT_LABEL}] {text}"
    if not targets:
        logger.info("[알림 대상 없음] %s", text.replace("\n", " / "))
        return
    for cid in targets:
        _call("sendMessage", {
            "chat_id": cid,
            "text": text,
            "disable_web_page_preview": True,
        })


def send_error(context: str, err: BaseException) -> None:
    send(f"⚠️ {context}\n{type(err).__name__}: {err}")


def is_allowed(chat_id: int) -> bool:
    return chat_id in config.TELEGRAM_ALLOWED_CHAT_IDS


def set_commands() -> None:
    """텔레그램 명령어 메뉴 등록 (최초 1회)."""
    _call("setMyCommands", {"commands": [
        {"command": "status", "description": "보유 포지션 + 손익"},
        {"command": "scan", "description": "지금 기준 시그널 스캔 (주문 안 함)"},
        {"command": "history", "description": "최근 매매 이력"},
        {"command": "config", "description": "현재 설정 확인"},
        {"command": "pause", "description": "자동매매 일시정지"},
        {"command": "resume", "description": "자동매매 재개"},
        {"command": "target", "description": "목표 수익률 보기/바꾸기 (/target 70, /target off)"},
        {"command": "deposit", "description": "입출금 반영 (/deposit 20만, /deposit -10만)"},
        {"command": "buy", "description": "수동 매수 (/buy 종목 가격 수량)"},
        {"command": "sell", "description": "수동 매도 (/sell 종목 가격 [수량])"},
        {"command": "bal", "description": "잔고 · 예수금"},
        {"command": "orders", "description": "미체결 주문"},
        {"command": "cancel", "description": "미체결 취소 (/cancel 주문번호|all)"},
        {"command": "fills", "description": "오늘 체결"},
        {"command": "progress", "description": "대회 조건 진행"},
        {"command": "close", "description": "자동매매 포지션 강제 청산 (/close 종목코드)"},
    ]})
