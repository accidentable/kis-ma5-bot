"""
core/trader.py — 주문 실행 + 청산 판단

전략(신호)과 한투 API(주문) 사이를 잇는다. 여기서만 주문을 낸다.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import config
from core import notify, state, strategy
from core.kis import quotes, trading
from core.kis.tick import offset_ticks, round_to_tick
from core.strategy import Candidate

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 진입
# ══════════════════════════════════════════════════════════════
def compute_entry_qty(ticker: str, limit_price: int) -> tuple[int, dict]:
    """
    매수 수량. 한투의 '미수없는매수수량' 에 POSITION_PCT 를 곱한다.
    미수(증거금 대출)를 쓰면 대회 수익률 계산이 꼬이고 반대매매 위험이 생기므로
    max_buy_qty(미수 포함) 는 쓰지 않는다.
    """
    info = trading.get_buyable(ticker, limit_price)
    base = info["qty_no_margin"]
    qty = int(base * config.POSITION_PCT / 100)
    return max(qty, 0), info


def enter(c: Candidate) -> tuple[Optional[dict], str]:
    """
    돌파 후보 매수.

    반환: (포지션 dict 또는 None, 실패 사유)
    사유가 '자금부족' 이면 호출부가 다음 순위 후보로 넘어갈 수 있다.
    주가가 높아 1주도 못 사는 건 신호의 문제가 아니라 계좌 사정이라,
    그 한 종목 때문에 그날 매매를 통째로 건너뛸 이유가 없다.
    """
    if state.is_paused():
        logger.info("일시정지 상태 — 매수 생략")
        return None, "일시정지"
    if len(state.get_positions()) >= config.MAX_POSITIONS:
        logger.info("보유 한도 도달 — 매수 생략")
        return None, "보유한도"

    # 지정가를 현재가보다 몇 틱 위로 둬서 즉시 체결을 노린다.
    # 시장가를 안 쓰는 이유: 전액 주문이라 호가가 얇으면 슬리피지를 그대로 맞는다.
    limit = offset_ticks(c.price, config.ENTRY_LIMIT_TICKS)

    qty, buyable = compute_entry_qty(c.ticker, limit)
    if qty <= 0:
        logger.info(
            "%s(%s) 매수 불가 — 주가 %s원 / 주문가능 %s원",
            c.name, c.ticker, f"{limit:,}", f"{buyable['cash']:,.0f}",
        )
        return None, "자금부족"

    result = trading.buy(c.ticker, qty, limit)
    if not result["success"]:
        notify.send(f"⚠️ {c.name}({c.ticker}) 매수 주문 거부: {result['message']}")
        return None, "주문거부"

    pos = {
        "ticker": c.ticker,
        "name": c.name,
        "qty": qty,
        "entry_price": limit,          # 체결 확인 후 실제 평단으로 갱신된다
        "entry_date": date.today().isoformat(),
        "entry_order_no": result["order_no"],
        "entry_org_no": result["org_no"],
        "breakout_price": round(c.breakout_price, 2),  # 매일 아침 갱신
        "take_profit_price": round(limit * (1 + config.TAKE_PROFIT_PCT / 100)),
        "stop_price": round(limit * (1 - config.STOP_LOSS_PCT / 100)) if config.USE_STOP_LOSS else 0,
        "hold_days": 1,                # 진입일이 1일차
        "atr": round(c.atr, 2),
        "signal": {
            "prev_disparity": round(c.prev_disparity, 2),
            "trend_disparity": round(c.trend_disparity, 2),
            "below_days": c.below_days,
            "thrust": round(c.thrust, 3),
            "score": c.score,
        },
        "filled": False,
        "dry_run": result.get("dry_run", False),
    }
    state.add_position(pos)
    state.mark_traded_today(c.ticker)

    notify.send(
        f"🟢 매수 주문\n"
        f"{c.name}({c.ticker}) {qty:,}주 @{limit:,}원 (약 {qty * limit:,}원)\n"
        f"{strategy.describe(c)}\n"
        f"익절 {pos['take_profit_price']:,}원(+{config.TAKE_PROFIT_PCT:.0f}%) | "
        f"5일선 {c.breakout_price:,.0f} 이탈 시 청산 | 최대 {config.MAX_HOLD_TRADING_DAYS}거래일"
        + ("\n(DRY_RUN — 실제 주문 아님)" if result.get("dry_run") else "")
    )
    return pos, ""


# ══════════════════════════════════════════════════════════════
# 청산
# ══════════════════════════════════════════════════════════════
def intraday_exit_line(pos: dict) -> float:
    """
    장중 5일선 이탈 판정선 = 5일선 − 여유폭.
    여유폭은 진입 시 저장한 ATR × INTRADAY_MA5_ATR_BUFFER. ATR 이 없으면 % 대체값.
    """
    breakout = float(pos.get("breakout_price", 0) or 0)
    atr = float(pos.get("atr", 0) or 0)
    if config.INTRADAY_MA5_ATR_BUFFER <= 0:
        return breakout
    if atr > 0:
        return breakout - config.INTRADAY_MA5_ATR_BUFFER * atr
    return breakout * (1 - config.INTRADAY_MA5_PCT_BUFFER_FALLBACK / 100)


def take_profit_target(pos: dict) -> float:
    """종가 익절선 = 진입가 +TAKE_PROFIT_PCT%. 체결 후 실제 평단으로 갱신된 값을 우선 쓴다."""
    stored = float(pos.get("take_profit_price", 0) or 0)
    if stored > 0:
        return stored
    entry = float(pos.get("entry_price", 0) or 0)
    return entry * (1 + config.TAKE_PROFIT_PCT / 100)


def intraday_tp_line(pos: dict) -> float:
    """
    장중 익절 판정선 = 종가 익절선 + 여유폭. 손절선을 5일선 아래로 내린 것과 대칭으로,
    장중엔 익절선을 위로 올려 스파이크에 성급히 팔지 않게 한다. 종가엔 여유 없이 +3%.
    여유폭은 ATR × TP_INTRADAY_ATR_BUFFER. ATR 이 없으면 % 대체값.
    """
    target = take_profit_target(pos)
    atr = float(pos.get("atr", 0) or 0)
    if config.TP_INTRADAY_ATR_BUFFER <= 0:
        return target
    if atr > 0:
        return target + config.TP_INTRADAY_ATR_BUFFER * atr
    return target * (1 + config.INTRADAY_MA5_PCT_BUFFER_FALLBACK / 100)


def evaluate_exit(
    pos: dict, price: float, *, closing: bool = False, stop_blackout: bool = False,
) -> Optional[str]:
    """
    청산 사유를 판정한다. 없으면 None.

    closing=True        장 마감 직전 점검. 보유기간 만료는 이때만 본다
                        (장중에 기간 만료로 털면 그날 종가 반등을 통째로 버린다).
    stop_blackout=True  손절 유예 구간(개장~STOP_BLACKOUT_UNTIL). 급이탈·손절을 보지 않고
                        익절만 본다. 분할매수로 개장 변동성을 받아내는 동안 흔들려도 안 판다.
    """
    # 분할 주문을 걸어놨는데 한 주도 안 채워졌으면 팔 것도 없다.
    # 기준가 대비 +3% 라고 '익절' 로 잡으면 유령 수익이 이력에 남는다.
    if pos.get("accumulating") and int(pos.get("qty", 0) or 0) <= 0:
        return None

    entry = float(pos.get("entry_price", 0) or 0)
    if price <= 0 or entry <= 0:
        return None

    pnl_pct = (price / entry - 1) * 100

    if not stop_blackout and config.USE_STOP_LOSS and pos.get("stop_price") and price <= pos["stop_price"]:
        return f"손절 {pnl_pct:+.2f}%"

    # 익절: 종가엔 +3% 그대로, 장중엔 여유폭만큼 더 오른 자리에서만.
    if closing:
        if pnl_pct >= config.TAKE_PROFIT_PCT:
            return f"익절 {pnl_pct:+.2f}%"
    else:
        if price >= intraday_tp_line(pos):
            return f"익절 {pnl_pct:+.2f}%"

    breakout = float(pos.get("breakout_price", 0) or 0)
    if strategy.is_ma5_broken(breakout, price):
        if closing:
            # 종가 판정은 여유 없이 5일선 그대로 — 원래 규칙
            return f"5일선 이탈 ({breakout:,.0f} 아래)"
        if config.EXIT_ON_INTRADAY_MA5_BREAK and not stop_blackout:
            # 장중엔 잡음과 붕괴를 구분한다. 5일선보다 여유폭만큼 더 내려가야 이탈로 친다.
            line = intraday_exit_line(pos)
            if price < line:
                return f"5일선 급이탈 ({line:,.0f} 아래, 5일선 {breakout:,.0f})"

    if closing and int(pos.get("hold_days", 1)) >= config.MAX_HOLD_TRADING_DAYS:
        return f"{config.MAX_HOLD_TRADING_DAYS}거래일 만료 {pnl_pct:+.2f}%"

    return None


def _exit_ticks(reason: str) -> int:
    """
    청산 사유별 지정가 깊이. 익절은 급할 게 없으니 얕게, 나머지는 확실히 체결되게.
    사유 문자열은 evaluate_exit 가 만든다.
    """
    return config.TP_EXIT_LIMIT_TICKS if reason.startswith("익절") else config.EXIT_LIMIT_TICKS
    # "5일선 이탈" / "5일선 급이탈" / "만료" / "손절" 은 전부 깊은 지정가


def exit_position(pos: dict, reason: str, price: Optional[float] = None) -> Optional[dict]:
    """보유 포지션 전량 매도."""
    ticker = pos["ticker"]

    if price is None:
        try:
            price = quotes.get_price(ticker)["price"]
        except Exception as e:
            logger.error("%s 현재가 조회 실패: %s", ticker, e)
            notify.send_error(f"{ticker} 청산 중 현재가 조회 실패", e)
            return None

    # 분할 매수가 아직 걸려 있으면 먼저 거둔다. 파는 동안 아래 주문이 채워지면 다시 보유 상태가 된다.
    if pos.get("entry_orders"):
        n = cancel_entry_orders(pos)
        if n:
            logger.info("%s 청산 전 미체결 분할 매수 %d건 취소", ticker, n)

    # 실제 보유 수량은 잔고가 진실이다 (부분체결·수동매도 가능성)
    qty = int(pos.get("qty", 0))
    try:
        for h in trading.get_balance()["holdings"]:
            if h["ticker"] == ticker:
                qty = h["sellable_qty"] or h["qty"]
                break
    except Exception as e:
        logger.warning("잔고 확인 실패, 기록상 수량으로 진행: %s", e)

    if qty <= 0:
        logger.warning("%s 매도 가능 수량 0 — 포지션만 정리", ticker)
        return state.close_position(ticker, price, f"{reason} (보유 없음)")

    # 현재가보다 몇 틱 아래에 지정가를 낸다. 깊이는 사유에 따라 다르다.
    limit = offset_ticks(price, -_exit_ticks(reason))
    result = trading.sell(ticker, qty, limit)
    if not result["success"]:
        notify.send(f"⚠️ {pos.get('name', ticker)} 매도 주문 거부: {result['message']}")
        return None

    record = state.close_position(ticker, limit, reason, qty=qty)
    state.mark_traded_today(ticker)
    if record:
        emoji = "✅" if record["pnl_pct"] > 0 else "🔴"
        notify.send(
            f"{emoji} 매도 주문 — {reason}\n"
            f"{record.get('name', ticker)}({ticker}) {qty:,}주 @{limit:,}원\n"
            f"진입 {record['entry_price']:,}원 → {limit:,}원  {record['pnl_pct']:+.2f}% "
            f"({record['pnl_amount']:+,}원)\n"
            f"보유 {record.get('hold_days', 0)}거래일"
            + ("\n(DRY_RUN — 실제 주문 아님)" if result.get("dry_run") else "")
        )
    return record


# ══════════════════════════════════════════════════════════════
# 동기화
# ══════════════════════════════════════════════════════════════
def refresh_breakout_prices() -> None:
    """
    보유 종목의 5일선 기준가를 오늘 값으로 갱신한다.
    5일선은 매일 움직이므로 진입 당시 값을 계속 쓰면 청산 판정이 틀어진다.
    보유일차도 여기서 하루 올린다. (거래일 아침마다 1회 호출)
    """
    for pos in state.get_positions():
        ticker = pos["ticker"]
        try:
            candles = quotes.get_daily_candles(ticker, days=10)
            candles = _drop_today(candles)
            if len(candles) < config.MA_PERIOD:
                continue
            closes = [c["close"] for c in candles]
            breakout = sum(closes[-(config.MA_PERIOD - 1):]) / (config.MA_PERIOD - 1)

            # 08:40 준비와 09:05 진입이 같은 날 둘 다 부르므로, 보유일차는 하루 한 번만 올린다.
            today = date.today().isoformat()
            already = pos.get("hold_refreshed") == today
            entry_day = pos.get("entry_date") == today
            hold = int(pos.get("hold_days", 1)) + (0 if (entry_day or already) else 1)

            state.update_position(ticker, breakout_price=round(breakout, 2), hold_days=hold, hold_refreshed=today)
            logger.info("%s 5일선 기준가 갱신: %.0f (보유 %d일차)", ticker, breakout, hold)
        except Exception as e:
            logger.error("%s 기준가 갱신 실패: %s", ticker, e)


def sync_fills() -> None:
    """
    미체결 매수가 채워졌는지 확인해 실제 평단으로 포지션을 보정한다.
    지정가 주문이라 체결가가 주문가와 다를 수 있다.
    """
    if config.DRY_RUN:
        return
    try:
        balance = trading.get_balance()
    except Exception as e:
        logger.error("체결 동기화 실패: %s", e)
        return

    held = {h["ticker"]: h for h in balance["holdings"]}
    for pos in state.get_positions():
        h = held.get(pos["ticker"])
        if not h or h["avg_price"] <= 0:
            continue
        if pos.get("filled") and abs(h["avg_price"] - pos["entry_price"]) < 1:
            continue
        state.update_position(
            pos["ticker"],
            entry_price=round(h["avg_price"]),
            qty=h["qty"],
            filled=True,
            take_profit_price=round(h["avg_price"] * (1 + config.TAKE_PROFIT_PCT / 100)),
            stop_price=round(h["avg_price"] * (1 - config.STOP_LOSS_PCT / 100)) if config.USE_STOP_LOSS else 0,
        )
        logger.info("%s 체결 반영: 평단 %.0f × %d주", pos["ticker"], h["avg_price"], h["qty"])


def cancel_all_pending() -> list[dict]:
    """미체결 주문 전량 취소."""
    cancelled = []
    try:
        pending = trading.get_pending_orders()
    except Exception as e:
        logger.error("미체결 조회 실패: %s", e)
        return cancelled

    for o in pending:
        try:
            trading.cancel_order(o["order_no"], o["org_no"], o["remain_qty"], order_dvsn=o["order_dvsn"])
            cancelled.append(o)
            logger.info("미체결 취소: %s %s %d주", o["ticker"], o["side"], o["remain_qty"])
        except Exception as e:
            logger.error("%s 주문 취소 실패: %s", o["order_no"], e)

    # 매수가 한 주도 안 채워진 채 취소됐다면 포지션 기록도 지운다
    for o in cancelled:
        if o["side"] != "buy" or o["filled_qty"] > 0:
            continue
        pos = state.get_position(o["ticker"])
        if pos and not pos.get("filled"):
            state.close_position(o["ticker"], pos["entry_price"], "미체결 취소")

    return cancelled


def _drop_today(candles: list[dict]) -> list[dict]:
    """장중 조회 시 섞여 들어오는 당일 미완성 봉을 제거한다."""
    today = date.today().strftime("%Y%m%d")
    return [c for c in candles if c["date"] != today]


# ══════════════════════════════════════════════════════════════
# 프리마켓 분할 진입
# ══════════════════════════════════════════════════════════════
def _split_quantities(total: int, n: int) -> list[int]:
    """total 주를 n 개로 나눈다. 나머지는 앞(높은 가격) 쪽부터 1주씩 더 준다."""
    n = max(1, min(n, total))
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def enter_split(c: Candidate, ref_price: float, *, source: str = "08:50 눌림대기") -> tuple[Optional[dict], str]:
    """
    눌림 대기 주문. 기준가 + PREOPEN_OFFSET_ATR×ATR 에 지정가를 걸어두고 장 초반 급락을 기다린다.
    SPLIT_TRANCHES ≥ 2 면 그 아래로 SPLIT_STEP_ATR 간격으로 더 건다.
    동시호가부터 유효하므로 시가가 지정가 이하면 시가에 채워진다.
    체결 여부는 sync_fills 가 잔고로 확인하고, SPLIT_CANCEL_AT 에 finalize_accumulation 이 정리한다.
    """
    if state.is_paused():
        return None, "일시정지"
    if len(state.get_positions()) >= config.MAX_POSITIONS:
        return None, "보유한도"
    if ref_price <= 0:
        return None, "가격없음"

    offset = config.PREOPEN_OFFSET_ATR * c.atr if c.atr > 0 else 0.0
    top = round_to_tick(ref_price * (1 + config.PREOPEN_CHASE_PCT / 100) + offset, "down")
    if top <= 0:
        return None, "가격없음"
    total, buyable = compute_entry_qty(c.ticker, top)
    if total <= 0:
        logger.info("%s(%s) 눌림 대기 불가 — 지정가 %s원 / 주문가능 %s원",
                    c.name, c.ticker, f"{top:,}", f"{buyable['cash']:,.0f}")
        return None, "자금부족"

    step = config.SPLIT_STEP_ATR * c.atr if c.atr > 0 else ref_price * 0.01
    orders: list[dict] = []
    for i, qty in enumerate(_split_quantities(total, config.SPLIT_TRANCHES)):
        price = round_to_tick(top - i * step, "down")
        if price <= 0:
            break
        r = trading.buy(c.ticker, qty, price)
        if not r["success"]:
            notify.send(f"⚠️ {c.name}({c.ticker}) {i + 1}차 분할매수 거부: {r['message']}")
            continue
        orders.append({"order_no": r["order_no"], "org_no": r["org_no"],
                       "qty": qty, "price": price, "dry_run": bool(r.get("dry_run"))})
    if not orders:
        return None, "주문거부"

    dry = orders[0]["dry_run"]
    pos = {
        "ticker": c.ticker,
        "name": c.name,
        # DRY_RUN 은 실제 체결이 없어 sync_fills 가 아무것도 못 채운다. 흐름을 볼 수 있게 전량 체결로 가정한다.
        "qty": total if dry else 0,
        "target_qty": total,
        "entry_price": top,            # 체결 후 실제 평단으로 갱신된다
        "entry_date": date.today().isoformat(),
        "entry_order_no": orders[0]["order_no"],
        "entry_org_no": orders[0]["org_no"],
        "entry_orders": orders,
        "accumulating": True,
        "breakout_price": round(c.breakout_price, 2),
        "take_profit_price": round(top * (1 + config.TAKE_PROFIT_PCT / 100)),
        "stop_price": round(top * (1 - config.STOP_LOSS_PCT / 100)) if config.USE_STOP_LOSS else 0,
        "hold_days": 1,
        "atr": round(c.atr, 2),
        "signal": {
            "prev_disparity": round(c.prev_disparity, 2),
            "ma_squeeze": round(c.ma_squeeze, 2),
            "vol_contraction": round(c.vol_contraction, 3),
            "below_days": c.below_days,
            "thrust": round(c.thrust, 3),
            "score": c.score,
            "premarket_price": ref_price,
        },
        "filled": dry,
        "dry_run": dry,
        "source": source,
    }
    state.add_position(pos)
    # 당일매매 표시는 실제로 채워졌을 때(finalize/청산) 한다. 한 주도 안 채워지면 재스캔에서 다시 살 수 있어야 한다.

    cancel_hhmm = f"{config.SPLIT_CANCEL_AT[:2]}:{config.SPLIT_CANCEL_AT[2:]}"
    if len(orders) == 1:
        head = (f"🟦 눌림 대기 — 기준가 {ref_price:,.0f}원, 지정가 {orders[0]['price']:,}원 "
                f"({config.PREOPEN_OFFSET_ATR:+g}ATR = {offset:+,.0f}원)")
        lines = f"  {total:,}주 — 여기까지 빠지면 체결. {cancel_hhmm}까지 안 오면 취소 후 현재가 매수"
    else:
        head = f"🟦 분할 대기 (기준가 {ref_price:,.0f}원)"
        lines = "\n".join(f"  {i + 1}차 {o['qty']:,}주 @{o['price']:,}원" for i, o in enumerate(orders))
    notify.send(
        f"{head}\n{c.name}({c.ticker})\n{lines}\n"
        f"{strategy.describe(c)}\n"
        f"급이탈 감시는 {config.STOP_BLACKOUT_UNTIL[:2]}:{config.STOP_BLACKOUT_UNTIL[2:]}부터"
        + ("\n(DRY_RUN — 실제 주문 아님)" if dry else "")
    )
    return pos, ""


def cancel_entry_orders(pos: dict) -> int:
    """분할 매수 중 아직 안 채워진 주문을 취소한다. 취소 요청한 건수를 돌려준다."""
    orders = pos.get("entry_orders") or []
    if not orders:
        return 0

    pending: Optional[dict[str, dict]]
    try:
        pending = {o["order_no"]: o for o in trading.get_pending_orders()}
    except Exception as e:
        logger.error("미체결 조회 실패 — 기록상 주문으로 취소 시도: %s", e)
        pending = None

    n = 0
    for o in orders:
        if o.get("dry_run"):
            continue  # DRY_RUN 주문은 실제로 없다
        p = pending.get(o["order_no"]) if pending is not None else None
        if pending is not None and p is None:
            continue  # 이미 다 체결됐거나 취소됨
        remain = int(p["remain_qty"]) if p else int(o["qty"])
        dvsn = p["order_dvsn"] if p else config.ORD_DVSN_LIMIT
        try:
            trading.cancel_order(o["order_no"], o["org_no"], remain, order_dvsn=dvsn)
            n += 1
        except Exception as e:
            logger.error("%s 분할 주문 %s 취소 실패: %s", pos["ticker"], o["order_no"], e)
    return n


def finalize_accumulation() -> list[dict]:
    """
    SPLIT_CANCEL_AT 에 호출. 안 채워진 분할 주문을 취소하고 포지션을 확정한다.
    한 주도 안 채워졌으면 포지션을 지워 슬롯을 비운다 — 같은 틱의 재진입 루프가 현재가로 다시 산다.
    """
    out: list[dict] = []
    for pos in state.get_positions():
        if not pos.get("accumulating"):
            continue
        ticker, name = pos["ticker"], pos.get("name", pos["ticker"])

        cancelled = cancel_entry_orders(pos)
        sync_fills()
        cur = state.get_position(ticker)
        if cur is None:
            continue

        qty = int(cur.get("qty", 0) or 0)
        if qty <= 0:
            state.remove_position(ticker)
            notify.send(f"⚪ {name}({ticker}) 눌림이 안 왔다 — 대기 주문 취소, 현재가로 다시 본다.")
            out.append({"ticker": ticker, "result": "미체결", "cancelled": cancelled})
            continue

        state.update_position(ticker, accumulating=False)
        state.mark_traded_today(ticker)
        target = int(cur.get("target_qty", qty) or qty)
        notify.send(
            f"✅ {name}({ticker}) 눌림 체결 확정 — {qty:,}/{target:,}주, 평단 {float(cur['entry_price']):,.0f}원"
            + (f", 미체결 {cancelled}건 취소" if cancelled else "")
            + f"\n지금부터 급이탈 감시 (5일선 {float(cur.get('breakout_price', 0)):,.0f} / 급이탈선 {intraday_exit_line(cur):,.0f})"
        )
        out.append({"ticker": ticker, "result": "확정", "qty": qty, "cancelled": cancelled})
    return out
