"""
handlers/aws.py — AWS Lambda 진입점

하나의 배포 패키지에 핸들러 4개를 둔다. EventBridge Scheduler 가 시간 작업을,
API Gateway 가 텔레그램 웹훅을 부른다.

  handlers.aws.entry_handler     09:05 KST  진입
  handlers.aws.monitor_handler   장중 10분  청산 감시
  handlers.aws.close_handler     15:15 KST  마감 정리
  handlers.aws.webhook_handler   텔레그램 명령
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


def _ok(payload) -> dict:
    return {"statusCode": 200, "body": json.dumps(payload, ensure_ascii=False, default=str)}


def _guard(name: str, fn, event):
    """작업 실패를 텔레그램으로 알리고, 람다는 성공으로 끝낸다(무한 재시도 방지)."""
    from core import notify
    try:
        result = fn(force=bool((event or {}).get("force")))
        logger.info("%s 완료: %s", name, result)
        return _ok({"job": name, "result": result})
    except Exception as e:
        logger.exception("%s 실패", name)
        notify.send_error(f"{name} 작업 실패", e)
        return _ok({"job": name, "error": f"{type(e).__name__}: {e}"})


def _closebet() -> bool:
    import config
    return config.STRATEGY == "closebet"


def _near_high() -> bool:
    import config
    return config.STRATEGY == "near_high"


def prep_handler(event, context):
    if _closebet():
        from jobs import closebet
        return _guard("시가 매도", closebet.sell_open, event)
    if _near_high():
        from jobs import rotation
        return _guard("순위 계산", rotation.prep, event)
    from jobs import prep
    return _guard("준비", prep.run, event)


def premarket_handler(event, context):
    if _near_high():
        return _ok({"job": "프리마켓", "skipped": "near_high 는 프리마켓 진입 없음"})
    from jobs import prep
    return _guard("프리마켓 분할진입", prep.premarket_entry, event)


def entry_handler(event, context):
    if _closebet():
        from jobs import closebet
        return _guard("시가 미체결 점검", closebet.check_open, event)
    if _near_high():
        from jobs import rotation
        return _guard("교체 매매", rotation.entry, event)
    from jobs import entry
    return _guard("진입", entry.run, event)


def monitor_handler(event, context):
    if _closebet():
        return _ok({"job": "감시", "skipped": "closebet 은 장중 감시 없음"})
    if _near_high():
        from jobs import rotation
        return _guard("감시", rotation.monitor, event)
    from jobs import monitor
    return _guard("감시", monitor.run, event)


def close_handler(event, context):
    if _closebet():
        from jobs import closebet
        return _guard("종가 매수", closebet.buy_close, event)
    if _near_high():
        from jobs import rotation
        return _guard("마감", rotation.close, event)
    from jobs import close
    return _guard("마감", close.run, event)


def panic_scan_handler(event, context):
    """시장 급락 판정 (PANIC_SCAN_TIME, 장 마감 뒤). PANIC_ENABLED=false 면 아무것도 안 한다."""
    from jobs import panic
    return _guard("급락 판정", panic.scan, event)


def webhook_handler(event, context):
    """
    텔레그램 웹훅. API Gateway(HTTP API) payload v2.0 기준.
    어떤 경우에도 200 을 돌려줘야 텔레그램이 재전송을 멈춘다.
    """
    from core import commands, notify

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        update = json.loads(body)
    except Exception as e:
        logger.warning("웹훅 페이로드 파싱 실패: %s", e)
        return _ok({"ok": True})

    message = update.get("message") or update.get("edited_message") or {}
    text = (message.get("text") or "").strip()
    chat_id = (message.get("chat") or {}).get("id")

    if not text or chat_id is None:
        return _ok({"ok": True})

    if not notify.is_allowed(int(chat_id)):
        logger.warning("허용되지 않은 chat_id 접근: %s", chat_id)
        return _ok({"ok": True})

    try:
        reply = commands.handle(text, int(chat_id))
    except Exception as e:
        logger.exception("명령 처리 실패")
        reply = f"⚠️ 오류: {type(e).__name__}: {e}"

    if reply:
        notify.send(reply, chat_id=int(chat_id))
    return _ok({"ok": True})


# 로컬에서 핸들러를 직접 호출해볼 때: python -m handlers.aws entry
if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "entry"
    handler = {
        "entry": entry_handler,
        "monitor": monitor_handler,
        "close": close_handler,
    }.get(name)
    if handler is None:
        print(f"알 수 없는 핸들러: {name}")
        sys.exit(1)
    print(handler({"force": "--force" in sys.argv}, None))
