"""
jobs/panic.py — 패닉 모드: 시장 급락일 다음 날 과매도주로 계좌를 갈아탔다가 PANIC_HOLD_DAYS 뒤 평소 전략으로 돌아온다.
STRATEGY=near_high 에 얹는 선택 기능 (PANIC_ENABLED=true 일 때만 동작). 계산 규칙은 core/panic.py.

  scan()      PANIC_SCAN_TIME(15:35)  장 마감 뒤 시총 상위 일봉을 받아 급락일인지 판정, 급락일이면 내일 살 후보를 저장
  enter()     다음 날 NH_ENTRY_TIME   평소 보유를 전부 팔고 후보 PANIC_SLOTS 개를 같은 금액씩 산다
              (rotation.entry / monitor 가 부른다. 못 산 슬롯은 NH_BUY_CUTOFF 까지 다음 후보로 재시도)
  bump()      매일 아침 (rotation.prep)  보유 거래일 수를 올린다 (산 날 = 1일째)
  maybe_exit() 15:15 마감 작업 (rotation.close)  PANIC_HOLD_DAYS 일째면 패닉 종목을 전부 팔고, 다음 날 평소 전략 재정렬

상태는 state.json 의 "panic": status(idle|pending|active), signal_date, drop, n, candidates[], entry_date, day_idx, counted
"""
from __future__ import annotations

import logging
from datetime import date

import config
from core import notify, state, trader, universe
from core import panic as P
from core.kis import quotes, trading
from core.kis.tick import offset_ticks

logger = logging.getLogger(__name__)


def _today() -> str:
    return date.today().isoformat()


def get() -> dict:
    return dict(state.load().get("panic") or {"status": "idle"})


def put(**fields) -> dict:
    data = state.load()
    p = dict(data.get("panic") or {"status": "idle"})
    p.update(fields)
    data["panic"] = p
    state.save(data)
    return p


def is_active() -> bool:
    return config.PANIC_ENABLED and get().get("status") == "active"


def is_pending() -> bool:
    return config.PANIC_ENABLED and get().get("status") == "pending"


def blocks_base() -> bool:
    """패닉 대기·보유 중엔 평소 전략이 사고팔지 않는다."""
    return is_active() or is_pending()


def panic_positions() -> list[dict]:
    return [p for p in state.get_positions() if p.get("strategy") == "panic"]


# ══════════════════════════════════════════════════════════════
# 판정 (장 마감 뒤)
# ══════════════════════════════════════════════════════════════
def scan(force: bool = False) -> dict:
    if not config.PANIC_ENABLED and not force:
        return {"skipped": "PANIC_ENABLED=false"}
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}
    if get().get("status") == "active":
        return {"skipped": "패닉 보유 중 (새 신호 안 받음)"}
    today = date.today().strftime("%Y%m%d")
    stocks = universe.get_large_universe(config.PANIC_UNIVERSE_TOP)
    snaps, errors = [], 0
    for i, s in enumerate(stocks, 1):
        try:
            cs = quotes.get_daily_history(s["ticker"], bars=30)
        except Exception as e:
            errors += 1
            logger.warning("%s 일봉 조회 실패: %s", s["ticker"], e)
            continue
        sn = P.snapshot(s["ticker"], s["name"], s.get("market", ""), i, cs, today)
        if sn:
            snaps.append(sn)
    avg, n = P.market_drop(snaps)
    out = {"date": _today(), "avg": avg, "n": n, "universe": len(stocks), "snaps": len(snaps), "errors": errors}
    if not P.is_panic(avg, n):
        if get().get("status") == "pending":
            put(status="idle")          # 오늘 새로 판정했으니 전날 대기는 버린다 (엔트리가 안 돈 경우)
        logger.info("급락일 아님: 평균 %+.2f%% (%d종목)", avg * 100, n)
        return {**out, "panic": False}
    cands = P.pick(snaps, config.PANIC_CANDIDATES)
    put(status="pending" if cands else "idle", signal_date=_today(), drop=avg, n=n,
        candidates=[c.to_dict() for c in cands], entry_date="", day_idx=0, counted="")
    lines = [f"🚨 시장 급락일 {_today()} — 거래대금 {config.PANIC_MKT_MIN_VALUE / 1e8:,.0f}억↑ {n}종목 평균 {avg * 100:+.2f}%",
             f"내일 {config.NH_ENTRY_TIME[:2]}:{config.NH_ENTRY_TIME[2:]} 보유를 모두 팔고 아래 {config.PANIC_SLOTS}종목을 산다 "
             f"({config.PANIC_HOLD_DAYS}거래일 보유 뒤 평소 전략으로 복귀)", ""]
    lines += [f"{k}. {P.describe(c)}" for k, c in enumerate(cands[: config.PANIC_SLOTS + 3], 1)]
    if not cands:
        lines.append("⚠️ 조건에 맞는 후보가 없다 — 평소 전략 유지")
    notify.send("\n".join(lines))
    return {**out, "panic": True, "candidates": [c.ticker for c in cands]}


# ══════════════════════════════════════════════════════════════
# 매매
# ══════════════════════════════════════════════════════════════
def _budget() -> float:
    p = get()
    b = float(p.get("slot_budget", 0) or 0)
    if b > 0:
        return b
    bal = trading.get_balance()
    b = max(bal["net_asset"], bal["cash"]) / config.PANIC_SLOTS
    put(slot_budget=b)
    return b


def _buy(c: dict, budget: float) -> tuple[dict | None, str]:
    try:
        q = quotes.get_price(c["ticker"])
    except Exception as e:
        logger.warning("%s 현재가 조회 실패: %s", c["ticker"], e)
        return None, "시세실패"
    if q["is_halted"] or q["is_watch"]:
        return None, "정지/경고"
    price = q["price"]
    if price <= 0:
        return None, "시세실패"
    limit = offset_ticks(price, config.ENTRY_LIMIT_TICKS)
    info = trading.get_buyable(c["ticker"], limit)
    qty = min(int(info["qty_no_margin"]), int(budget // limit))
    if qty <= 0:
        return None, "자금부족"
    r = trading.buy(c["ticker"], qty, limit)
    if not r["success"]:
        notify.send(f"⚠️ {c['name']}({c['ticker']}) 매수 주문 거부: {r['message']}")
        return None, "주문거부"
    pos = {
        "ticker": c["ticker"], "name": c["name"], "qty": qty, "entry_price": limit, "entry_date": _today(),
        "entry_order_no": r["order_no"], "entry_org_no": r["org_no"],
        "strategy": "panic", "source": "패닉 매수",
        "signal": {k: c[k] for k in ("ret5", "ret1", "rank", "val20")},
        "breakout_price": 0, "take_profit_price": 0, "stop_price": 0, "hold_days": 1, "atr": 0,
        "filled": False, "dry_run": r.get("dry_run", False),
    }
    state.add_position(pos)
    state.mark_traded_today(c["ticker"])
    notify.send(f"🟢 패닉 매수\n{P.describe(c)}\n{qty:,}주 @{limit:,}원 (약 {qty * limit:,}원)"
                + ("\n(DRY_RUN — 실제 주문 아님)" if r.get("dry_run") else ""))
    return pos, ""


def _fill() -> dict:
    """빈 패닉 슬롯을 후보 순서대로 채운다 (못 사는 종목은 건너뜀)."""
    p = get()
    held = {x["ticker"] for x in state.get_positions()}
    tried = set(p.get("tried") or [])
    need = config.PANIC_SLOTS - len(panic_positions())
    entered, skipped = [], []
    if need <= 0:
        return {"entered": [], "skipped": []}
    budget = _budget()
    for c in p.get("candidates") or []:
        if len(entered) >= need:
            break
        if c["ticker"] in held or c["ticker"] in tried:
            continue
        pos, why = _buy(c, budget)
        if pos:
            entered.append(pos["ticker"])
            held.add(c["ticker"])
        else:
            skipped.append(f"{c['name']}({c['ticker']}) — {why}")
            if why in ("정지/경고", "주문거부"):
                tried.add(c["ticker"])
    put(tried=sorted(tried))
    return {"entered": entered, "skipped": skipped}


def enter() -> dict:
    """대기 신호가 있으면: 평소 보유를 모두 팔고 후보를 산다. 이미 오늘 들어갔으면 빈 슬롯만 다시 채운다."""
    p = get()
    if p.get("status") == "active" and p.get("entry_date") == _today():
        return {"refill": _fill()}
    if p.get("status") != "pending":
        return {"skipped": "대기 신호 없음"}
    if state.is_paused():
        return {"skipped": "일시정지"}
    sold = []
    for pos in state.get_positions():
        if pos.get("strategy") == "panic":
            continue
        rec = trader.exit_position(pos, "패닉 모드 전환 (시장 급락 다음 날)")
        if rec:
            sold.append(rec)
    if sold:
        from jobs import rotation
        rotation._wait_sold({r["ticker"] for r in sold})
    bal = trading.get_balance()
    put(status="active", entry_date=_today(), day_idx=1, counted=_today(), tried=[],
        slot_budget=max(bal["net_asset"], bal["cash"]) / config.PANIC_SLOTS)
    res = _fill()
    notify.send(f"🔄 패닉 모드 진입 {_today()} — 매도 {len(sold)} | 매수 {len(res['entered'])}/{config.PANIC_SLOTS}"
                + (f"\n못 산 후보: " + "; ".join(res["skipped"][:5]) if res["skipped"] else "")
                + f"\n{config.PANIC_HOLD_DAYS}거래일째 15:15 에 전부 팔고 평소 전략으로 돌아간다")
    return {"sold": sold, **res}


def bump() -> dict:
    """거래일 아침마다 보유 일수를 올린다 (산 날 = 1일째)."""
    p = get()
    if p.get("status") != "active" or p.get("counted") == _today():
        return p
    return put(day_idx=int(p.get("day_idx", 0) or 0) + 1, counted=_today())


def maybe_exit() -> list[dict]:
    """마감 작업에서: 보유 PANIC_HOLD_DAYS 일째면 패닉 종목을 전부 팔고 평소 전략 재정렬을 예약한다."""
    p = get()
    if p.get("status") != "active":
        return []
    if int(p.get("day_idx", 0) or 0) < config.PANIC_HOLD_DAYS:
        return []
    out = []
    for pos in panic_positions():
        rec = trader.exit_position(pos, f"패닉 모드 {config.PANIC_HOLD_DAYS}거래일 만료")
        if rec:
            out.append(rec)
    put(status="idle", slot_budget=0, tried=[])
    from jobs import rotation
    rotation.set_rot(cycle_start=None, day_idx=0, rebalanced=None, slot_budget=0)   # 다음 날 평소 전략 새로 정렬
    notify.send(f"🏁 패닉 모드 종료 {_today()} — {len(out)}종목 매도, 내일 평소 전략 순위로 다시 산다")
    return out


def status_line() -> str:
    if not config.PANIC_ENABLED:
        return ""
    p = get()
    st = p.get("status", "idle")
    if st == "pending":
        return f"🚨 패닉 대기: {p.get('signal_date')} 급락({float(p.get('drop', 0)) * 100:+.2f}%) — 다음 개장에 진입"
    if st == "active":
        return f"🚨 패닉 보유 {int(p.get('day_idx', 0))}/{config.PANIC_HOLD_DAYS}거래일 (진입 {p.get('entry_date')})"
    return f"패닉 모드 대기 중 (시장 평균 −{config.PANIC_MKT_DROP_PCT:g}% 이하일 때 발동)"
