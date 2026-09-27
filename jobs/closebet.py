"""
jobs/closebet.py — 종가 베팅 (STRATEGY=closebet, 선택형)

당일 강세로 마감하는 테마주(거래대금 상위)를 장마감 동시호가에 사서, 다음 날 장전 동시호가(시가)에 판다.

  sell_open()   CB_SELL_TIME(08:45)  보유 전량을 하한가 지정가로 → 시가 동시호가에서 시가에 체결
  check_open()  CB_CHECK_TIME(09:05) 아직 남은 보유(정지·VI 등)는 현재가 −2틱으로 다시 매도
  buy_close()   CB_BUY_TIME(15:21)   거래대금 순위 → 조건 판정 → 장마감 동시호가 매수 (현재가 + CB_BUY_TICKS 틱)
  report()      15:40                마감 리포트 (15:30 체결 뒤)

조건 (backtest/lab/closebet.py 의 IS 1위 설정과 같다)
  당일 거래대금 CB_RANK_TOP 위 안 보통주, 등락률 CB_MIN_CHANGE_PCT ~ CB_MAX_CHANGE_PCT %,
  IBS = (현재가 − 저가)/(고가 − 저가) ≥ CB_MIN_IBS, 거래대금 ≥ CB_MIN_VALUE,
  (CB_REGIME) 코스닥지수 종가가 100일선 위. 거래대금 큰 순으로 CB_SLOTS 종목.

백테스트 (KRX 전종목 2015.7 ~ 2026.9, 상장폐지 포함, 60만원·한 달): 한 달 평균 IS +0.4% / VAL −0.8% / TEST +0.3%.
관찰된 '강세 마감 종목의 다음 날 시가 갭'(평균 +0.2 ~ +0.5%)이 매도세·수수료(왕복 약 0.23%)에 대부분 먹힌다.
판정은 15:21 가격으로 하므로 백테스트(종가로 판정)보다 약간 불리하다.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import config
from core import notify, state, trader, universe
from core.kis import quotes, trading
from core.kis.tick import offset_ticks, round_to_tick

logger = logging.getLogger(__name__)


def _today() -> str:
    return date.today().isoformat()


def _open_day(force: bool) -> bool:
    return force or quotes.is_open_day(date.today())


# ══════════════════════════════════════════════════════════════
# 판정
# ══════════════════════════════════════════════════════════════
def regime_on() -> tuple[bool, str]:
    """코스닥지수 종가(최근 확정치)가 100일 이동평균 위인가. 조회 실패면 끈다 (안전 쪽)."""
    if not config.CB_REGIME:
        return True, "국면 필터 꺼짐"
    try:
        rows = quotes.get_index_daily("1001", bars=110)
    except Exception as e:
        logger.warning("코스닥지수 조회 실패: %s", e)
        return False, f"코스닥지수 조회 실패 ({e})"
    closes = [r["close"] for r in rows]
    if len(closes) < 100:
        return False, f"코스닥지수 일봉 부족 ({len(closes)})"
    ma = sum(closes[-100:]) / 100
    on = closes[-1] > ma
    return on, f"코스닥 {closes[-1]:,.2f} {'>' if on else '≤'} 100일선 {ma:,.2f}"


def _common_stocks() -> set[str]:
    """보통주 코드 집합 (ETF·스팩·우선주·관리 등 제외). 하루 캐시."""
    return {s["ticker"] for s in universe.get_large_universe(100_000)}


def candidates() -> tuple[list[dict], dict]:
    """지금 가격 기준 종가 베팅 후보. 거래대금 큰 순."""
    stats: dict = {"rank": 0, "rejects": {}}
    try:
        ranked = quotes.get_value_rank("0000")
    except Exception as e:
        logger.error("거래대금 순위 조회 실패: %s", e)
        stats["error"] = str(e)
        return [], stats
    try:
        common = _common_stocks()
    except Exception as e:
        logger.warning("보통주 목록 실패 — 코드 끝자리로만 거른다: %s", e)
        common = None
    stats["rank"] = len(ranked)
    out = []

    def rej(k):
        stats["rejects"][k] = stats["rejects"].get(k, 0) + 1

    for r in ranked[: config.CB_RANK_TOP]:
        t = r["ticker"]
        if (common is not None and t not in common) or not t.endswith("0"):
            rej("보통주아님")
            continue
        try:
            q = quotes.get_price(t)
        except Exception as e:
            logger.warning("%s 현재가 조회 실패: %s", t, e)
            rej("시세실패")
            continue
        if q["is_halted"] or q["is_watch"]:
            rej("정지/경고")
            continue
        price, hi, lo = q["price"], q["high"], q["low"]
        chg = q["change_pct"]
        ibs = (price - lo) / (hi - lo) if hi > lo else 0.0
        if not (config.CB_MIN_CHANGE_PCT <= chg < config.CB_MAX_CHANGE_PCT):
            rej("등락률")
            continue
        if q["upper_limit"] and price >= q["upper_limit"]:
            rej("상한가")
            continue
        if ibs < config.CB_MIN_IBS:
            rej("IBS")
            continue
        if q["value"] < config.CB_MIN_VALUE:
            rej("거래대금")
            continue
        out.append({"ticker": t, "name": r["name"] or q["name"], "price": price, "change_pct": chg,
                    "ibs": round(ibs, 3), "value": q["value"], "upper_limit": q["upper_limit"]})
    out.sort(key=lambda x: -x["value"])
    return out, stats


# ══════════════════════════════════════════════════════════════
# 작업
# ══════════════════════════════════════════════════════════════
def buy_close(force: bool = False) -> dict:
    """15:21 — 장마감 동시호가 매수."""
    if not _open_day(force):
        return {"skipped": "휴장일"}
    if state.is_paused():
        return {"skipped": "일시정지"}
    held = {p["ticker"] for p in state.get_positions()}
    slots = config.CB_SLOTS - len(held)
    if slots <= 0:
        return {"skipped": "보유한도"}
    on, why = regime_on()
    if not on:
        notify.send(f"🌙 종가 베팅 쉼 — {why}")
        return {"skipped": "국면", "why": why}
    cands, stats = candidates()
    try:
        bal = trading.get_balance()
        budget = max(bal["net_asset"], bal["cash"]) / config.CB_SLOTS
    except Exception as e:
        logger.error("잔고 조회 실패: %s", e)
        return {"skipped": "잔고실패"}
    entered, skipped = [], []
    for c in cands:
        if len(entered) >= slots:
            break
        if c["ticker"] in held:
            continue
        limit = offset_ticks(c["price"], config.CB_BUY_TICKS)
        if c["upper_limit"]:
            limit = min(limit, int(c["upper_limit"]))
        info = trading.get_buyable(c["ticker"], limit)
        qty = min(int(info["qty_no_margin"]), int(budget // limit))
        if qty <= 0:
            skipped.append(f"{c['name']}({c['ticker']}) 자금부족")
            continue
        r = trading.buy(c["ticker"], qty, limit)
        if not r["success"]:
            skipped.append(f"{c['name']}({c['ticker']}) 주문거부 {r['message']}")
            continue
        pos = {
            "ticker": c["ticker"], "name": c["name"], "qty": qty, "entry_price": limit,
            "entry_date": _today(), "entry_order_no": r["order_no"], "entry_org_no": r["org_no"],
            "strategy": "closebet", "source": "장마감 동시호가",
            "signal": {k: c[k] for k in ("price", "change_pct", "ibs", "value")},
            "breakout_price": 0, "take_profit_price": 0, "stop_price": 0, "hold_days": 1, "atr": 0,
            "filled": False, "dry_run": r.get("dry_run", False),
        }
        state.add_position(pos)
        state.mark_traded_today(c["ticker"])
        entered.append(pos)
        held.add(c["ticker"])
    lines = [f"🌙 종가 베팅 {_today()} — {why}",
             f"거래대금 순위 {stats.get('rank', 0)} → 후보 {len(cands)} → 주문 {len(entered)}"]
    for p in entered:
        s = p["signal"]
        lines.append(f"🟢 {p['name']}({p['ticker']}) {p['qty']:,}주 @{p['entry_price']:,}원 "
                     f"({s['change_pct']:+.1f}%, IBS {s['ibs']:.2f}, 거래대금 {s['value'] / 1e8:,.0f}억)")
    if skipped:
        lines.append("못 산 후보: " + "; ".join(skipped[:5]))
    if stats.get("rejects"):
        lines.append(f"탈락: {stats['rejects']}")
    if config.DRY_RUN:
        lines.append("(DRY_RUN — 실제 주문 아님)")
    notify.send("\n".join(lines))
    return {"entered": entered, "skipped": skipped, "stats": stats}


def _sell_all(reason: str, *, at_open: bool) -> list[dict]:
    """보유(어제 이전에 산 것) 전량 매도. at_open 이면 하한가 지정가 → 시가 동시호가 체결."""
    out = []
    for pos in state.get_positions():
        if pos.get("entry_date") == _today():
            continue            # 오늘 산 건 내일 판다
        try:
            q = quotes.get_price(pos["ticker"])
        except Exception as e:
            logger.error("%s 시세 조회 실패: %s", pos["ticker"], e)
            continue
        if at_open:
            px = q["lower_limit"] or round_to_tick((q["prev_close"] or q["price"]) * 0.75, "up")
            qty = int(pos.get("qty", 0) or 0)
            try:
                for h in trading.get_balance()["holdings"]:
                    if h["ticker"] == pos["ticker"]:
                        qty = h["sellable_qty"] or h["qty"]
            except Exception as e:
                logger.warning("잔고 확인 실패, 기록 수량으로: %s", e)
            if qty <= 0:
                out.append(state.close_position(pos["ticker"], float(pos["entry_price"]), f"{reason} (보유 없음)"))
                continue
            r = trading.sell(pos["ticker"], qty, int(px))
            if not r["success"]:
                notify.send(f"⚠️ {pos.get('name', '')}({pos['ticker']}) 시가 매도 주문 거부: {r['message']}")
                continue
            # 체결가는 시가다. 기록은 전일 종가 기준으로 남기고, 실제 손익은 잔고·체결내역이 진실이다
            rec = state.close_position(pos["ticker"], float(q["prev_close"] or q["price"]), reason, qty=qty)
            state.mark_traded_today(pos["ticker"])
            out.append(rec)
        else:
            rec = trader.exit_position(pos, reason, price=q["price"])
            if rec:
                out.append(rec)
    return [r for r in out if r]


def sell_open(force: bool = False) -> dict:
    """08:45 — 장전 동시호가 매도."""
    if not _open_day(force):
        return {"skipped": "휴장일"}
    trader.sync_fills()
    sold = _sell_all("종가 베팅 익일 시가 매도", at_open=True)
    if sold:
        notify.send("🌅 종가 베팅 청산 주문 (시가 동시호가)\n"
                    + "\n".join(f"· {r.get('name', '')}({r['ticker']}) {r.get('sold_qty', 0):,}주" for r in sold))
    return {"sold": sold}


def check_open(force: bool = False) -> dict:
    """09:05 — 시가에 안 팔린 보유(정지·VI·미체결)를 현재가로 다시 판다."""
    if not _open_day(force):
        return {"skipped": "휴장일"}
    trader.sync_fills()
    left = []
    try:
        held = {h["ticker"]: h for h in trading.get_balance()["holdings"]}
    except Exception as e:
        logger.error("잔고 조회 실패: %s", e)
        return {"skipped": "잔고실패"}
    if config.DRY_RUN:
        return {"left": []}
    closed_today = {h["ticker"] for h in state.get_history(limit=20) if h.get("exit_date") == _today()
                    and "종가 베팅" in str(h.get("exit_reason", ""))}
    for t in closed_today:
        h = held.get(t)
        if not h or h["qty"] <= 0:
            continue
        # 시가 매도가 안 채워졌다 — 미체결 취소 후 현재가 매도
        try:
            for o in trading.get_pending_orders():
                if o["ticker"] == t and o["side"] == "sell":
                    trading.cancel_order(o["order_no"], o["org_no"], o["remain_qty"], order_dvsn=o["order_dvsn"])
            price = quotes.get_price(t)["price"]
            r = trading.sell(t, h["sellable_qty"] or h["qty"], offset_ticks(price, -config.EXIT_LIMIT_TICKS))
            left.append({"ticker": t, "qty": h["qty"], "resent": r["success"]})
        except Exception as e:
            logger.error("%s 재매도 실패: %s", t, e)
            left.append({"ticker": t, "error": str(e)})
    if left:
        notify.send("⚠️ 시가에 안 팔린 종가 베팅 종목 재매도: " + ", ".join(x["ticker"] for x in left))
    return {"left": left}


def report(force: bool = False) -> dict:
    if not _open_day(force):
        return {"skipped": "휴장일"}
    trader.sync_fills()
    lines = [f"📕 마감 리포트 {_today()} — 종가 베팅"]
    try:
        bal = trading.get_balance()
        lines.append(f"순자산 {bal['net_asset']:,.0f}원 | 예수금 {bal['cash']:,.0f}원 | 평가손익 {bal['pnl_amount']:+,.0f}원")
        for h in bal["holdings"]:
            lines.append(f"· {h['name']}({h['ticker']}) {h['qty']:,}주 평단 {h['avg_price']:,.0f}")
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")
    notify.send("\n".join(lines))
    return {"ok": True}


def status_text() -> str:
    lines = ["전략: 종가 베팅 (선택형)"]
    try:
        bal = trading.get_balance()
        lines.append(f"💰 순자산 {bal['net_asset']:,.0f}원 | 예수금 {bal['cash']:,.0f}원")
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")
    for p in state.get_positions():
        lines.append(f"📌 {p.get('name', '')}({p['ticker']}) {int(p.get('qty', 0)):,}주 @{float(p.get('entry_price', 0)):,.0f} "
                     f"({p.get('entry_date', '')} 매수 → 다음 날 시가 매도)")
    if state.is_paused():
        lines.append("⏸ 일시정지 중")
    return "\n".join(lines)


def now_hhmm() -> str:
    return datetime.now().strftime("%H%M")
