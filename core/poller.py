"""
core/poller.py — 텔레그램 롱폴링

VM 상주 실행용. 공개 엔드포인트(API Gateway) 없이 봇 명령을 받는다.
서버가 텔레그램 쪽으로 나가는 연결만 쓰므로 인바운드 포트를 열 필요가 없고,
웹훅을 안 쓰니 getUpdates 충돌(409)도 생기지 않는다.
"""
from __future__ import annotations

import logging
import threading
import time

import requests

import config
from core import commands

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"
_LONG_POLL_SEC = 30


def _url(method: str) -> str:
    return _API.format(token=config.TELEGRAM_BOT_TOKEN, method=method)


def ensure_no_webhook() -> None:
    """
    롱폴링과 웹훅은 동시에 못 쓴다. 예전 배포에서 남은 웹훅이 있으면 해제한다.
    """
    try:
        info = requests.get(_url("getWebhookInfo"), timeout=10).json()
        hook = (info.get("result") or {}).get("url", "")
        if hook:
            logger.warning("기존 웹훅 발견(%s) — 롱폴링을 위해 해제한다", hook)
            requests.post(_url("deleteWebhook"), timeout=10)
    except requests.RequestException as e:
        logger.warning("웹훅 상태 확인 실패: %s", e)


def poll_once(offset: int) -> int:
    """
    한 번 폴링하고 처리한 뒤 다음 offset 을 반환한다.
    getUpdates 는 timeout 만큼 서버가 붙잡고 있다가 응답한다(롱폴링).
    """
    from core import notify

    try:
        resp = requests.get(
            _url("getUpdates"),
            params={"offset": offset, "timeout": _LONG_POLL_SEC, "allowed_updates": '["message"]'},
            timeout=_LONG_POLL_SEC + 15,
        )
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("폴링 실패: %s", e)
        time.sleep(5)
        return offset

    if not data.get("ok"):
        logger.error("getUpdates 오류: %s", data.get("description"))
        # 웹훅이 다시 걸린 경우 등 — 한 번 정리하고 계속
        if "webhook" in str(data.get("description", "")).lower():
            ensure_no_webhook()
        time.sleep(5)
        return offset

    for update in data.get("result", []):
        offset = max(offset, int(update["update_id"]) + 1)

        msg = update.get("message") or update.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = (msg.get("chat") or {}).get("id")
        if not text or chat_id is None:
            continue

        chat_id = int(chat_id)
        if not notify.is_allowed(chat_id):
            logger.warning("허용되지 않은 chat_id 접근: %s (%r)", chat_id, text[:40])
            continue

        logger.info("명령 수신: %s", text)
        try:
            reply = commands.handle(text, chat_id)
        except Exception as e:
            logger.exception("명령 처리 실패: %s", text)
            reply = f"⚠️ 오류: {type(e).__name__}: {e}"

        if reply:
            notify.send(reply, chat_id=chat_id)

    return offset


def run(stop: threading.Event | None = None) -> None:
    """롱폴링 루프. stop 이벤트가 세팅되면 종료한다."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.info("텔레그램 토큰 없음 — 폴링 비활성")
        return

    ensure_no_webhook()

    # 재시작 직후 밀려있던 과거 명령을 재실행하지 않도록 오프셋을 끝으로 밀어둔다
    offset = 0
    try:
        data = requests.get(_url("getUpdates"), params={"timeout": 0}, timeout=15).json()
        for u in data.get("result", []):
            offset = max(offset, int(u["update_id"]) + 1)
        if offset:
            logger.info("밀린 업데이트 %d건 건너뜀", len(data.get("result", [])))
    except requests.RequestException:
        pass

    logger.info("텔레그램 롱폴링 시작")
    while not (stop and stop.is_set()):
        try:
            offset = poll_once(offset)
        except Exception:
            logger.exception("폴링 루프 예외 — 5초 후 계속")
            time.sleep(5)
    logger.info("텔레그램 롱폴링 종료")


def start_thread() -> tuple[threading.Thread, threading.Event]:
    """데몬 스레드로 폴링을 띄운다."""
    stop = threading.Event()
    t = threading.Thread(target=run, args=(stop,), name="telegram-poller", daemon=True)
    t.start()
    return t, stop
