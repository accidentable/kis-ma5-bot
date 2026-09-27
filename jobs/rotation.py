"""
jobs/rotation.py — 52주 신고가 근접 로테이션의 하루 (STRATEGY=near_high)

  prep()     NH_PREP_TIME(08:20)  거래일 카운트를 올리고, 교체일이거나 빈 슬롯이 있으면 순위를 계산해 캐시한다
  entry()    NH_ENTRY_TIME(09:05) 교체일이면 순위 밖 보유 종목을 팔고, 빈 슬롯을 순위대로 채운다
  monitor()  10분마다            체결 동기화, (설정 시) 손절, 교체가 밀렸으면 실행, 빈 슬롯 재시도 (NH_BUY_CUTOFF 까지)
  close()    15:15               미체결 정리 + 마감 리포트. 보유 기간 만료 같은 청산은 없다 (교체일에만 판다)

상태는 state.json 의 "rotation" 에 둔다.
  cycle_start   마지막 교체일 (없으면 아직 한 번도 안 샀다 → 다음 진입이 곧 교체)
  day_idx       교체일 이후 지난 거래일 수. NH_HOLD_DAYS 이상이면 교체일
  counted       day_idx 를 마지막으로 올린 날짜 (하루 한 번만 올린다)
  ranking       {date, items[], stats} 그날 순위표
  rebalanced    교체를 실행한 날짜 (같은 날 두 번 교체하지 않는다)
  slot_budget   교체일에 정한 슬롯당 금액 (순자산 / NH_SLOTS)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime

import config
from core import nearhigh, notify, state, trader, universe
from core.kis import quotes, trading
from core.kis.tick import offset_ticks

logger = logging.getLogger(__name__)

lock = threading.Lock()


def _now_hhmm() -> str:
    return datetime.now().strftime("%H%M")


def _today() -> str:
    return date.today().isoformat()


# ══════════════════════════════════════════════════════════════
# 상태
# ══════════════════════════════════════════════════════════════
def get_rot() -> dict:
    return dict(state.load().get("rotation") or {})


def set_rot(**fields) -> dict:
    data = state.load()
    rot = dict(data.get("rotation") or {})
    rot.update(fields)
    data["rotation"] = rot
    state.save(data)
    return rot


def bump_day() -> dict:
    """거래일 아침마다 한 번 day_idx 를 올린다."""
    rot = get_rot()
    today = _today()
    if rot.get("counted") == today:
        return rot
    idx = int(rot.get("day_idx", 0) or 0)
    if rot.get("cycle_start") and rot.get("cycle_start") != today:
        idx += 1
    return set_rot(day_idx=idx, counted=today)


def is_rebalance_due(rot: dict | None = None) -> bool:
    rot = rot if rot is not None else get_rot()
    if rot.get("rebalanced") == _today():
        return False
    if not rot.get("cycle_start"):
        return True
    return int(rot.get("day_idx", 0) or 0) >= config.NH_HOLD_DAYS


def days_left(rot: dict | None = None) -> int:
    rot = rot if rot is not None else get_rot()
    if not rot.get("cycle_start"):
        return 0
    return max(config.NH_HOLD_DAYS - int(rot.get("day_idx", 0) or 0), 0)


# ══════════════════════════════════════════════════════════════
# 순위
# ══════════════════════════════════════════════════════════════
def _drop_today(candles: list[dict]) -> list[dict]:
    today = date.today().strftime("%Y%m%d")
    return [c for c in candles if c["date"] != today]


def build_ranking() -> tuple[list[nearhigh.Pick], dict]:
    """시총 상위 N 의 일봉을 받아 순위를 매기고 오늘 캐시에 저장한다 (3~5분)."""
    stocks = universe.get_large_universe(config.NH_UNIVERSE_TOP)
    picks, rejects, errors = [], {}, 0
    need = config.NH_HIGH_LOOKBACK + 5
    for s in stocks:
        try:
            candles = _drop_today(quotes.get_daily_history(s["ticker"], bars=need))
            p = nearhigh.evaluate(s["ticker"], s["name"], candles, market=s.get("market", ""),
                                  marcap=float(s.get("marcap", 0) or 0))
        except Exception as e:
            errors += 1
            logger.warning("%s(%s) 일봉 조회 실패: %s", s["name"], s["ticker"], e)
            continue
        picks.append(p)
        if not p.eligible:
            key = p.reject.split(" ")[0]
            rejects[key] = rejects.get(key, 0) + 1
    ranked = nearhigh.rank(picks)
    top = ranked[: config.NH_CANDIDATES]
    stats = {"universe": len(stocks), "eligible": len(ranked), "errors": errors, "rejects": rejects}
    set_rot(ranking={"date": _today(), "items": [nearhigh.to_dict(p) for p in top], "stats": stats})
    logger.info("순위 계산: 유니버스 %d → 적격 %d (오류 %d) %s", len(stocks), len(ranked), errors, rejects)
    return top, stats


def load_ranking(build_if_missing: bool = True) -> tuple[list[nearhigh.Pick], dict]:
    r = get_rot().get("ranking") or {}
    if r.get("date") == _today() and r.get("items") is not None:
        return [nearhigh.from_dict(d) for d in r["items"]], dict(r.get("stats") or {})
    if not build_if_missing:
        return [], {}
    return build_ranking()


def format_ranking(ranked: list[nearhigh.Pick], stats: dict, limit: int = 10) -> str:
    head = (f"📡 52주 신고가 근접 순위 {_today()}\n"
            f"시총 상위 {stats.get('universe', '-')} → 적격 {stats.get('eligible', '-')}"
            + (f" (오류 {stats['errors']})" if stats.get("errors") else ""))
    if not ranked:
        return head + "\n\n적격 종목 없음"
    lines = [head, ""]
    for i, p in enumerate(ranked[:limit], 1):
        lines.append(f"{i}. {nearhigh.describe(p)}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 매매
# ══════════════════════════════════════════════════════════════
def _slot_budget() -> float:
    rot = get_rot()
    b = float(rot.get("slot_budget", 0) or 0)
    if b > 0:
        return b
    bal = trading.get_balance()
    b = max(bal["net_asset"], bal["cash"]) / config.NH_SLOTS
    set_rot(slot_budget=b)
    return b


def _buy(p: nearhigh.Pick, budget: float, source: str) -> tuple[dict | None, str]:
    """슬롯 하나를 산다. 현재가 +ENTRY_LIMIT_TICKS 지정가, 수량은 슬롯 금액과 미수없는 매수가능 중 작은 쪽."""
    try:
        q = quotes.get_price(p.ticker)
    except Exception as e:
        logger.warning("%s 현재가 조회 실패: %s", p.ticker, e)
        return None, "시세실패"
    if q["is_halted"] or q["is_watch"]:
        return None, "정지/경고"
    price = q["price"]
    if price <= 0:
        return None, "시세실패"
    limit = offset_ticks(price, config.ENTRY_LIMIT_TICKS)
    info = trading.get_buyable(p.ticker, limit)
    qty = min(int(info["qty_no_margin"]), int(budget // limit))
    if qty <= 0:
        return None, "자금부족"
    r = trading.buy(p.ticker, qty, limit)
    if not r["success"]:
        notify.send(f"⚠️ {p.name}({p.ticker}) 매수 주문 거부: {r['message']}")
        return None, "주문거부"
    pos = {
        "ticker": p.ticker, "name": p.name, "qty": qty,
        "entry_price": limit,                 # 체결 확인 후 sync_fills 가 실제 평단으로 바꾼다
        "entry_date": _today(),
        "entry_order_no": r["order_no"], "entry_org_no": r["org_no"],
        "strategy": "near_high", "source": source,
        "signal": {"score": round(p.score, 4), "high_n": p.high_n, "prev_close": p.prev_close,
                   "mom": round(p.mom, 4), "marcap": p.marcap},
        "breakout_price": 0, "take_profit_price": 0, "stop_price": 0, "hold_days": 1, "atr": 0,
        "filled": False, "dry_run": r.get("dry_run", False),
    }
    state.add_position(pos)
    state.mark_traded_today(p.ticker)
    notify.send(
        f"🟢 매수 주문 [{source}]\n{p.name}({p.ticker}) {qty:,}주 @{limit:,}원 (약 {qty * limit:,}원)\n"
        f"{nearhigh.describe(p)}"
        + ("\n(DRY_RUN — 실제 주문 아님)" if r.get("dry_run") else "")
    )
    return pos, ""


def fill_slots(ranked: list[nearhigh.Pick], source: str) -> dict:
    """빈 슬롯을 순위대로 채운다. 비싸서 못 사는 종목은 건너뛴다."""
    held = {p["ticker"] for p in state.get_positions()}
    sold_today = state.get_traded_today() - held      # 오늘 판 종목은 되사지 않는다 (비용만 낸다)
    open_slots = config.NH_SLOTS - len(held)
    if open_slots <= 0:
        return {"entered": [], "skipped": [], "reason": "보유한도"}
    if state.is_paused():
        return {"entered": [], "skipped": [], "reason": "일시정지"}
    budget = _slot_budget()
    entered, skipped = [], []
    for p in ranked:
        if len(entered) >= open_slots:
            break
        if p.ticker in held or p.ticker in sold_today:
            continue
        pos, reason = _buy(p, budget, source)
        if pos:
            entered.append(pos)
            held.add(p.ticker)
        else:
            skipped.append(f"{p.name}({p.ticker}) — {reason}")
    if skipped and not entered:
        logger.info("빈 슬롯 매수 실패: %s", skipped)
    return {"entered": entered, "skipped": skipped, "reason": ""}


def rebalance(force: bool = False) -> dict:
    """교체일이면: 순위 밖 보유 종목 매도 → 빈 슬롯 매수. 교체일이 아니면 아무것도 안 한다."""
    rot = get_rot()
    if not force and not is_rebalance_due(rot):
        return {"skipped": "교체일 아님", "days_left": days_left(rot)}
    if state.is_paused():
        return {"skipped": "일시정지"}
    ranked, stats = load_ranking()
    if not ranked:
        set_rot(rebalanced=_today())     # 10분마다 같은 경고를 반복하지 않게. 교체 주기는 안 바꿔서 내일 다시 본다
        notify.send(format_ranking(ranked, stats) + "\n\n⚠️ 교체일인데 살 종목이 없다 — 보유 유지, 내일 다시 본다")
        return {"skipped": "후보없음"}

    # 슬롯 금액으로 1주도 못 사는 종목은 순위에서 뺀다. 안 빼면 비싼 1·2위가 '유지 목록'을 차지해
    # 멀쩡한 보유 종목을 팔고도 그 자리를 못 채운다. (16년 백테스트에서 이 필터 유무의 차이는 없었다)
    bal = trading.get_balance()
    budget = max(bal["net_asset"], bal["cash"]) / config.NH_SLOTS
    affordable = [p for p in ranked if 0 < p.prev_close <= budget]
    if not affordable:
        set_rot(rebalanced=_today())
        notify.send(f"⚠️ 교체일인데 슬롯 금액({budget:,.0f}원)으로 살 수 있는 후보가 없다 — 보유 유지")
        return {"skipped": "후보없음(가격)"}
    # 살 수 있는 순위 상위 NH_SLOTS 가 유지 목록. 그 밖의 보유 종목은 판다.
    keep = {p.ticker for p in affordable[: config.NH_SLOTS]}
    sold = []
    for pos in state.get_positions():
        if pos["ticker"] in keep:
            continue
        rec = trader.exit_position(pos, f"{config.NH_HOLD_DAYS}거래일 교체 (순위 밖)")
        if rec:
            sold.append(rec)
    if sold:
        _wait_sold({r["ticker"] for r in sold})

    bal = trading.get_balance()
    set_rot(cycle_start=_today(), day_idx=0, counted=_today(), rebalanced=_today(),
            slot_budget=max(bal["net_asset"], bal["cash"]) / config.NH_SLOTS)
    result = fill_slots(affordable, source="교체일 매수")
    kept = [p for p in state.get_positions() if p["ticker"] in keep and p["ticker"] not in
            {e["ticker"] for e in result["entered"]}]
    notify.send(
        f"🔄 교체일 {_today()} — 다음 교체 {config.NH_HOLD_DAYS}거래일 뒤\n"
        f"매도 {len(sold)} | 유지 {len(kept)} | 신규 {len(result['entered'])}"
        + (f"\n못 산 후보: " + "; ".join(result["skipped"][:5]) if result["skipped"] else "")
    )
    return {"sold": sold, "kept": [p["ticker"] for p in kept], **result}


def _wait_sold(tickers: set[str], timeout: float = 90.0) -> bool:
    """판 종목이 잔고에서 빠질 때까지 잠깐 기다린다. 매도 대금이 매수가능금액에 잡혀야 슬롯을 온전히 채운다."""
    if config.DRY_RUN:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            held = {h["ticker"] for h in trading.get_balance()["holdings"]}
        except Exception as e:
            logger.warning("매도 체결 확인 실패: %s", e)
            held = tickers
        if not (held & tickers):
            return True
        time.sleep(5)
    logger.warning("매도 체결 대기 시간 초과: %s — 가능한 금액만큼 산다 (남은 슬롯은 10분마다 재시도)", tickers & held)
    return False


def _check_stops() -> list[dict]:
    if config.NH_STOP_LOSS_PCT <= 0:
        return []
    closed = []
    for pos in state.get_positions():
        try:
            price = quotes.get_price(pos["ticker"])["price"]
        except Exception as e:
            logger.error("%s 현재가 조회 실패: %s", pos["ticker"], e)
            continue
        entry = float(pos.get("entry_price", 0) or 0)
        if entry > 0 and price > 0 and price <= entry * (1 - config.NH_STOP_LOSS_PCT / 100):
            rec = trader.exit_position(pos, f"손절 {(price / entry - 1) * 100:+.2f}%", price=price)
            if rec:
                closed.append(rec)
    return closed


# ══════════════════════════════════════════════════════════════
# 스케줄 작업
# ══════════════════════════════════════════════════════════════
def prep(force: bool = False) -> dict:
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}
    rot = bump_day()
    trader.sync_fills()
    due = is_rebalance_due(rot)
    need_rank = due or (config.NH_REFILL and len(state.get_positions()) < config.NH_SLOTS)
    msg = [f"🌅 개장 전 준비 {_today()}"]
    if rot.get("cycle_start"):
        msg.append(f"교체 주기 {int(rot.get('day_idx', 0))}/{config.NH_HOLD_DAYS}거래일"
                   + (" — 오늘 교체" if due else f" (교체까지 {days_left(rot)}거래일)"))
    else:
        msg.append("첫 매수일 — 오늘 순위대로 산다")
    ranked, stats = ([], {})
    if need_rank:
        ranked, stats = build_ranking()
        msg.append("")
        msg.append(format_ranking(ranked, stats, limit=5))
    held = ", ".join(f"{p.get('name', '')}({p['ticker']})" for p in state.get_positions()) or "없음"
    msg.append(f"\n보유: {held}")
    notify.send("\n".join(msg))
    return {"due": due, "ranked": [p.ticker for p in ranked], "stats": stats}


def entry(force: bool = False) -> dict:
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}
    if not lock.acquire(blocking=False):
        return {"skipped": "잠금"}
    try:
        bump_day()
        trader.sync_fills()
        if is_rebalance_due():
            return rebalance()
        if config.NH_REFILL and len(state.get_positions()) < config.NH_SLOTS:
            ranked, _ = load_ranking()
            return fill_slots(ranked, source="빈 슬롯 매수")
        return {"skipped": "교체일 아님", "days_left": days_left()}
    finally:
        lock.release()


def monitor(force: bool = False) -> dict:
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}
    trader.sync_fills()
    closed = _check_stops()
    now = _now_hhmm()
    out: dict = {"now": now, "closed": closed}
    if now < config.NH_ENTRY_TIME or now >= config.NH_BUY_CUTOFF:
        return out
    if not lock.acquire(blocking=False):
        return {**out, "skipped": "잠금"}
    try:
        bump_day()
        if is_rebalance_due():
            # 09:05 작업이 안 돌았으면(재시작 등) 여기서 교체한다
            out["rebalance"] = rebalance()
        elif len(state.get_positions()) < config.NH_SLOTS and (
                config.NH_REFILL or get_rot().get("rebalanced") == _today()):
            ranked, _ = load_ranking()
            out["refill"] = fill_slots(ranked, source="빈 슬롯 재시도")
    finally:
        lock.release()
    return out


def _report(closed: list[dict], cancelled: list[dict]) -> str:
    rot = get_rot()
    lines = [f"📕 마감 리포트 {_today()} — 52주 신고가 근접",
             f"교체 주기 {int(rot.get('day_idx', 0) or 0)}/{config.NH_HOLD_DAYS}거래일 (시작 {rot.get('cycle_start', '-')})"]
    try:
        bal = trading.get_balance()
        lines.append(f"순자산 {bal['net_asset']:,.0f}원 | 예수금 {bal['cash']:,.0f}원 | 평가손익 {bal['pnl_amount']:+,.0f}원")
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")
    positions = state.get_positions()
    lines.append("\n[보유]" if positions else "\n[보유] 없음")
    for p in positions:
        try:
            price = quotes.get_price(p["ticker"])["price"]
            entry_px = float(p.get("entry_price", 0) or 0)
            pnl = (price / entry_px - 1) * 100 if entry_px else 0
            lines.append(f"· {p.get('name', '')}({p['ticker']}) {int(p.get('qty', 0)):,}주 "
                         f"{entry_px:,.0f} → {price:,.0f} {pnl:+.2f}% | 진입 {p.get('entry_date', '')}")
        except Exception:
            lines.append(f"· {p.get('name', '')}({p['ticker']}) 시세 조회 실패")
    if closed:
        lines.append("\n[오늘 청산]")
        for r in closed:
            lines.append(f"· {r.get('name', '')}({r['ticker']}) {r['pnl_pct']:+.2f}% ({r['pnl_amount']:+,}원) — {r['exit_reason']}")
    if cancelled:
        lines.append(f"\n미체결 {len(cancelled)}건 취소")
    return "\n".join(lines)


def close(force: bool = False, send_report: bool = True) -> dict:
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}
    cancelled = trader.cancel_all_pending()
    trader.sync_fills()
    today_closed = [h for h in state.get_history(limit=20) if h.get("exit_date") == _today()]
    if send_report:
        notify.send(_report(today_closed, cancelled))
    return {"cancelled": cancelled, "closed": today_closed}


def status_text() -> str:
    rot = get_rot()
    lines = [f"전략: 52주 신고가 근접 | 교체 주기 {int(rot.get('day_idx', 0) or 0)}/{config.NH_HOLD_DAYS}거래일"
             + (f" (다음 교체까지 {days_left(rot)}거래일)" if rot.get("cycle_start") else " (아직 첫 매수 전)")]
    try:
        bal = trading.get_balance()
        lines.append(f"💰 순자산 {bal['net_asset']:,.0f}원 | 예수금 {bal['cash']:,.0f}원 | 평가손익 {bal['pnl_amount']:+,.0f}원")
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")
    positions = state.get_positions()
    if not positions:
        lines.append("\n보유 없음")
    for p in positions:
        try:
            price = quotes.get_price(p["ticker"])["price"]
        except Exception:
            price = 0
        entry_px = float(p.get("entry_price", 0) or 0)
        pnl = (price / entry_px - 1) * 100 if (entry_px and price) else 0
        lines.append(f"📌 {p.get('name', '')}({p['ticker']}) {int(p.get('qty', 0)):,}주 | {entry_px:,.0f} → {price:,.0f} {pnl:+.2f}%")
    if state.is_paused():
        lines.append("\n⏸ 자동매매 일시정지 중")
    return "\n".join(lines)
