"""
jobs/contest.py — 대회 모드 (STRATEGY=contest): 한 달 안에 +30% 한 번을 노리는 집중 모멘텀. 계산은 core/contest.py.

하루 일정 (전부 거래일만)
  prep()       CT_PREP_TIME(08:20)   월초면 기준 순자산 기록·락 해제. 대상 350종목 일봉(어제까지) 받아 캐시
  sell_open()  CT_SELL_TIME(08:45)   어제 마감에 정한 매도(손절·추적·만료·락·조건용) 를 장 시작 동시호가에 (하한가 지정가 → 시가 체결)
  check_open() CT_CHECK_TIME(09:05)  아직 안 팔린 것은 현재가 −2틱으로 다시
  scan()       CT_SCAN_TIME(15:05)   현재가 350개 → 폭락 판정 → 후보 순위 → 오늘 살 것 결정. 폭락 전환이면 보유를 지금(장중) 판다
  buy_close()  CT_BUY_TIME(15:20)    scan 이 정한 종목을 장 마감 동시호가 매수 (현재가 + CT_BUY_TICKS 틱, 상한가 이내)
  evaluate()   CT_EVAL_TIME(15:40)   종가로 손절·추적·만료 판정, 순자산으로 목표 락 판정 → 내일 아침 매도 목록. 마감 리포트
상태는 state.json 의 "contest" 에 둔다: month, anchor(월초 순자산), locked, hist(일봉 캐시), universe, plan(오늘 결정), pending_exit, fillers_done
포지션은 공용 positions 에 strategy="contest", kind=momentum|crash|filler, entry_date, peak_close 로 기록한다.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import config
from core import notify, state, trader, universe
from core.contest import Snap, crash_candidates, crash_signal, describe, exit_reasons, lock_hit, rank_momentum, split_universe
from core.kis import quotes, trading
from core.kis.tick import offset_ticks, round_to_tick

logger = logging.getLogger(__name__)


def _today() -> str:
    return date.today().isoformat()


def _month() -> str:
    return date.today().strftime("%Y-%m")


def _open_day(force: bool) -> bool:
    return force or quotes.is_open_day(date.today())


def _won(x: float) -> str:
    """돈을 짧게: 1.23억 / 9,967만"""
    x = float(x or 0)
    return f"{x / 1e8:.2f}억" if abs(x) >= 1e8 else f"{x / 1e4:,.0f}만"


def _md(d: str | None = None) -> str:
    d = d or _today()
    return f"{d[5:7]}/{d[8:10]}"


def _pos_line(p: dict, price: float | None = None) -> str:
    entry = float(p.get("entry_price", 0) or 0)
    px = price if price else 0
    pnl = f" ({px / entry - 1:+.1%})" if (entry and px) else ""
    kind = {"crash": " 폭락", "filler": " 조건용"}.get(p.get("kind", ""), "")
    return f"{p.get('name', '')} {int(p['qty']):,}주{pnl}{kind} · {_held_days(p)}일째"


def get() -> dict:
    return dict(state.load().get("contest") or {})


def put(**fields) -> dict:
    data = state.load()
    ct = dict(data.get("contest") or {})
    ct.update(fields)
    data["contest"] = ct
    state.save(data)
    return ct


BUYC, SELLC = 0.00065, 0.00265        # 종이 계좌 비용 (수수료 0.015% + 슬리피지 0.05%, 매도엔 세금 0.2% 추가)


def _nav() -> float:
    """순자산. DRY_RUN 이면 종이 계좌(현금 + 보유 평가) — 실계좌 잔고로는 수량이 안 나온다."""
    if config.DRY_RUN:
        ct = get()
        cash = float(ct.get("paper_cash", config.CT_PAPER_CAP) or 0)
        val = 0.0
        for p in _positions():
            try:
                val += int(p["qty"]) * float(quotes.get_price(p["ticker"])["price"])
            except Exception:
                val += int(p["qty"]) * float(p.get("entry_price", 0) or 0)
        return cash + val
    bal = trading.get_balance()
    return float(max(bal["net_asset"], bal["cash"]))


def _paper_buy(qty: int, price: float) -> None:
    if config.DRY_RUN:
        ct = get(); put(paper_cash=float(ct.get("paper_cash", config.CT_PAPER_CAP) or 0) - qty * price * (1 + BUYC))


def _paper_sell(rec: dict | None) -> None:
    if config.DRY_RUN and rec:
        ct = get(); put(paper_cash=float(ct.get("paper_cash", config.CT_PAPER_CAP) or 0) + int(rec.get("sold_qty", 0) or 0) * float(rec.get("exit_price", 0) or 0) * (1 - SELLC))


def _positions() -> list[dict]:
    return [p for p in state.get_positions() if p.get("strategy") == "contest"]


def _held_days(pos: dict) -> int:
    try:
        return quotes.trading_days_between(date.fromisoformat(pos["entry_date"]), date.today())
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════
# 08:20 준비
# ══════════════════════════════════════════════════════════════
def _month_reset(ct: dict) -> dict:
    if ct.get("month") == _month():
        return ct
    if config.DRY_RUN and "paper_cash" not in ct:
        put(paper_cash=config.CT_PAPER_CAP)
    try:
        anchor = _nav()
    except Exception as e:
        logger.error("월초 순자산 조회 실패: %s", e)
        anchor = float(ct.get("anchor", 0) or 0)
    ct = put(month=_month(), anchor=anchor, locked=False, locked_date="", fillers_done="")
    notify.send(f"📅 대회 모드 새 달 {_month()} — 기준 순자산 {anchor:,.0f}원, 목표 +{config.CT_LOCK_PCT:g}% ({anchor * (1 + config.CT_LOCK_PCT / 100):,.0f}원)")
    return ct


def build_universe() -> list[dict]:
    stocks = universe.get_large_universe(config.CT_UNIVERSE_PULL)
    return split_universe(stocks, config.CT_KOSPI_N, config.CT_KOSDAQ_N)


def prep(force: bool = False) -> dict:
    if not _open_day(force):
        return {"skipped": "휴장일"}
    ct = _month_reset(get())
    uni = build_universe()
    hist, fail = {}, []
    need = config.CT_LOOKBACK + 25
    for s in uni:
        try:
            candles = [c for c in quotes.get_daily_candles(s["ticker"], days=need) if c["date"] < date.today().strftime("%Y%m%d")]
        except Exception as e:
            fail.append(s["ticker"]); logger.warning("%s 일봉 실패: %s", s["ticker"], e); continue
        hist[s["ticker"]] = {"closes": [float(c["close"]) for c in candles], "values": [float(c.get("value", 0) or 0) for c in candles],
                             "date": candles[-1]["date"] if candles else ""}
    put(universe=uni, hist=hist, hist_date=_today())
    notify.send(morning_brief(uni, hist, fail, ct))
    return {"universe": len(uni), "hist": len(hist), "fail": fail}


def morning_brief(uni: list[dict], hist: dict, fail: list[str], ct: dict) -> str:
    """08:20 브리핑 — 한 줄에 하나씩, 휴대폰 폭에 맞게."""
    anchor = float(ct.get("anchor", 0) or 0)
    lines = [f"🌅 {_md()} 아침 브리핑" + (f"  (일봉 실패 {len(fail)})" if fail else ""), ""]
    try:
        nav = _nav()
        lines.append(f"💰 {'종이 ' if config.DRY_RUN else ''}{_won(nav)}" + (f"  (월초 {nav / anchor - 1:+.2%})" if anchor else ""))
    except Exception as e:
        logger.warning("브리핑 잔고 조회 실패: %s", e)
    pos = _positions()
    if pos:
        for p in pos:
            try:
                px = float(quotes.get_price(p["ticker"])["price"])
            except Exception:
                px = 0
            lines.append("📌 " + _pos_line(p, px))
    else:
        lines.append("📌 보유 없음")
    pend = ct.get("pending_exit") or []
    if pend:
        names = {p["ticker"]: p.get("name", "") for p in pos}
        lines.append("⏰ 08:45 매도: " + ", ".join(f"{names.get(x['ticker'], x['ticker'])} ({x['reason'].split(' (')[0]})" for x in pend))
    if ct.get("locked"):
        lines += ["", f"🔒 {_md(ct.get('locked_date', ''))} 목표 달성 — 이번 달 매수 없음"]
        return "\n".join(lines)
    snaps = []
    for s_ in uni:
        h = hist.get(s_["ticker"]) or {}
        c = list(h.get("closes") or [])
        if len(c) < config.CT_LOOKBACK + 2 or c[-1] <= 0 or c[-2] <= 0:
            continue
        snaps.append(Snap(ticker=s_["ticker"], name=s_.get("name", ""), market=s_.get("market", ""), price=c[-1], prev_close=c[-2],
                          change_pct=(c[-1] / c[-2] - 1) * 100, upper_limit=0, lower_limit=0, value=0, closes=c[:-1], values=list(h.get("values") or [])[:-1]))
    cands, st = rank_momentum(snaps)
    held = {p["ticker"] for p in pos}
    lines += ["", f"📈 어제 종가 기준 후보 ({config.CT_LOOKBACK}일 수익률)", "   15:05 에 오늘 가격으로 확정"]
    for i, s_ in enumerate(cands[:config.CT_BRIEF_N], 1):
        lines.append(f"{i}. {s_.name} {s_.ret_n:+.0%}  (어제 {s_.change_pct:+.1f}%)" + ("  ← 보유" if s_.ticker in held else ""))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 15:05 판정
# ══════════════════════════════════════════════════════════════
def snapshots(ct: dict) -> tuple[list[Snap], list[str]]:
    hist = ct.get("hist") or {}
    out, fail = [], []
    for s in ct.get("universe") or []:
        try:
            q = quotes.get_price(s["ticker"])
        except Exception as e:
            fail.append(s["ticker"]); logger.debug("%s 현재가 실패: %s", s["ticker"], e); continue
        if q.get("is_halted") or not q["price"] > 0:
            continue
        h = hist.get(s["ticker"]) or {}
        out.append(Snap(ticker=s["ticker"], name=s.get("name") or q.get("name", ""), market=s.get("market", ""),
                        price=float(q["price"]), prev_close=float(q.get("prev_close") or 0), change_pct=float(q.get("change_pct") or 0),
                        upper_limit=float(q.get("upper_limit") or 0), lower_limit=float(q.get("lower_limit") or 0),
                        value=float(q.get("value") or 0), closes=list(h.get("closes") or []), values=list(h.get("values") or [])))
    return out, fail


def decide(snaps: list[Snap], ct: dict) -> dict:
    """오늘 살 것을 정한다 (주문은 안 한다). 반환 plan dict."""
    held = _positions()
    held_t = {p["ticker"] for p in held}
    crash_on, cinfo = crash_signal(snaps)
    plan = {"date": _today(), "mode": "none", "buy": [], "sell_now": [], "crash": cinfo, "stats": {}}
    if ct.get("locked"):
        plan["mode"] = "locked"
        return plan
    in_crash = any(p.get("kind") == "crash" for p in held)
    if crash_on and not in_crash:
        cands, stats = crash_candidates(snaps, cinfo["mkt"])
        picks = [s for s in cands if s.ticker not in held_t][:config.CT_SLOTS]
        plan.update(mode="crash", stats=stats, sell_now=[p["ticker"] for p in held],
                    buy=[{"ticker": s.ticker, "name": s.name, "price": s.price, "upper_limit": s.upper_limit, "kind": "crash",
                          "signal": {"change_pct": s.change_pct, "mkt": cinfo["mkt"], "val_ratio": s.val_ratio}} for s in picks])
        return plan
    if in_crash:
        plan["mode"] = "crash_hold"
        return plan
    cands, stats = rank_momentum(snaps)
    free = config.CT_SLOTS - len(held)
    picks = [s for s in cands if s.ticker not in held_t][:max(free, 0)]
    plan.update(mode="momentum", stats=stats,
                buy=[{"ticker": s.ticker, "name": s.name, "price": s.price, "upper_limit": s.upper_limit, "kind": "momentum",
                      "signal": {"ret_n": s.ret_n, "change_pct": s.change_pct}} for s in picks],
                top=[{"ticker": s.ticker, "name": s.name, "ret_n": s.ret_n, "change_pct": s.change_pct} for s in cands[:10]])
    if picks and config.CT_FILLER_N > 0 and ct.get("fillers_done") != _month():
        chosen = {b["ticker"] for b in plan["buy"]} | held_t
        fill = [s for s in cands if s.ticker not in chosen][:config.CT_FILLER_N]
        plan["buy"] += [{"ticker": s.ticker, "name": s.name, "price": s.price, "upper_limit": s.upper_limit, "kind": "filler", "signal": {}} for s in fill]
    return plan


def scan(force: bool = False) -> dict:
    """15:05 — 판정. 폭락 전환이면 보유를 지금 판다 (동시호가 전에 현금을 만들어야 15:20 매수가 된다)."""
    if not _open_day(force):
        return {"skipped": "휴장일"}
    ct = get()
    if ct.get("hist_date") != _today() or not ct.get("hist"):
        prep(force=True)
        ct = get()
    snaps, fail = snapshots(ct)
    plan = decide(snaps, ct)
    if plan["mode"] == "crash" and plan["sell_now"]:
        sold = []
        for p in _positions():
            rec = trader.exit_position(p, "폭락 전환 매도")
            if rec:
                _paper_sell(rec); sold.append(rec)
        plan["sold_now"] = [r["ticker"] for r in sold]
    put(plan=plan)
    notify.send(scan_text(plan, snaps_n=len(snaps), fail=len(fail)))
    return plan


def scan_text(plan: dict, snaps_n: int = 0, fail: int = 0) -> str:
    c = plan.get("crash") or {}
    lines = [f"🔎 {_md(plan.get('date'))} 15:05 판정",
             f"시장 {c.get('mkt', 0):+.2%}" + (f"  (폭락 기준 {c['threshold']:+.2%})" if c.get("threshold") else "") + (f"  조회 실패 {fail}" if fail else ""), ""]
    mode = plan.get("mode")
    if mode == "locked":
        lines.append("🔒 이번 달 목표 달성 — 매수 없음")
    elif mode == "crash":
        lines.append(f"🚨 폭락 전환!  보유 {len(plan.get('sell_now', []))}종목 매도")
        for b in plan["buy"]:
            lines.append(f"🟢 {b['name']} {b['price']:,.0f}원  (오늘 {b['signal'].get('change_pct', 0):+.1f}%)")
        rj = plan.get("stats", {}).get("rejects")
        if rj:
            lines.append("   제외: " + ", ".join(f"{k} {v}" for k, v in rj.items()))
    elif mode == "crash_hold":
        lines.append("폭락 전환 종목 보유 중 — 만료까지 대기")
    else:
        st = plan.get("stats", {})
        cut = st.get("cutoff")
        lines.append(f"📈 모멘텀 상위 {st.get('top', 0)}종목" + (f"  (컷 {cut:+.0%})" if cut is not None else ""))
        for i, t in enumerate(plan.get("top", [])[:5], 1):
            lines.append(f"{i}. {t['name']} {t['ret_n']:+.0%}  (오늘 {t['change_pct']:+.1f}%)")
        buys = [b for b in plan["buy"] if b["kind"] == "momentum"]
        fill = [b for b in plan["buy"] if b["kind"] == "filler"]
        lines.append("")
        lines.append("🟢 오늘 매수: " + (", ".join(b["name"] for b in buys) if buys else "없음 (보유 중)"))
        if fill:
            lines.append("   조건용 1주: " + ", ".join(b["name"] for b in fill))
        rj = st.get("rejects")
        if rj:
            lines.append("   제외: " + ", ".join(f"{k} {v}" for k, v in rj.items()))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 15:20 매수
# ══════════════════════════════════════════════════════════════
def buy_close(force: bool = False) -> dict:
    if not _open_day(force):
        return {"skipped": "휴장일"}
    if state.is_paused():
        return {"skipped": "일시정지"}
    ct = get()
    if ct.get("locked"):
        return {"skipped": "locked"}
    plan = ct.get("plan") or {}
    if plan.get("date") != _today():
        plan = scan(force=True)
    if not plan.get("buy"):
        return {"skipped": plan.get("mode", "none")}
    held = {p["ticker"] for p in _positions()}
    if sum(1 for p in _positions() if p.get("kind") != "filler") >= config.CT_SLOTS and plan.get("mode") != "crash":
        return {"skipped": "slots"}
    try:
        nav = _nav()
    except Exception as e:
        logger.error("잔고 조회 실패: %s", e)
        return {"skipped": "잔고실패"}
    entered, skipped = [], []
    real = [b for b in plan["buy"] if b["kind"] != "filler"]
    budget = nav * 0.995 / max(config.CT_SLOTS, 1)
    for b in plan["buy"]:
        if b["ticker"] in held:
            continue
        try:
            q = quotes.get_price(b["ticker"])
            price, upper = float(q["price"]), float(q.get("upper_limit") or 0)
        except Exception:
            price, upper = float(b["price"]), float(b.get("upper_limit") or 0)
        limit = offset_ticks(price, config.CT_BUY_TICKS)
        if upper:
            limit = min(limit, int(upper))
        if b["kind"] == "filler":
            qty = 1
        elif config.DRY_RUN:
            qty = int(budget // limit)
        else:
            try:
                info = trading.get_buyable(b["ticker"], limit)
                qty = min(int(info["qty_no_margin"]), int(budget // limit))
            except Exception as e:
                logger.warning("%s 매수가능 조회 실패, 예산으로: %s", b["ticker"], e)
                qty = int(budget // limit)
        if qty <= 0:
            skipped.append(f"{b['name']} 자금부족"); continue
        r = trading.buy(b["ticker"], qty, limit)
        if not r["success"]:
            skipped.append(f"{b['name']} 주문거부 {r['message']}"); continue
        pos = {"ticker": b["ticker"], "name": b["name"], "qty": qty, "entry_price": limit, "entry_date": _today(),
               "entry_order_no": r["order_no"], "entry_org_no": r["org_no"], "strategy": "contest", "kind": b["kind"],
               "source": "장마감 동시호가", "signal": b.get("signal", {}), "peak_close": limit,
               "breakout_price": 0, "take_profit_price": 0, "stop_price": 0, "hold_days": 1, "atr": 0,
               "filled": False, "dry_run": r.get("dry_run", False)}
        state.add_position(pos)
        state.mark_traded_today(b["ticker"])
        _paper_buy(qty, limit)
        entered.append(pos)
    if any(p["kind"] == "filler" for p in entered):
        put(fillers_done=_month())
    lines = [f"🌙 {_md()} 종가 매수" + ("  (DRY_RUN · 주문 안 보냄)" if config.DRY_RUN else "") + ("  🚨 폭락 전환" if plan.get("mode") == "crash" else ""), ""]
    for p in [p for p in entered if p["kind"] != "filler"]:
        lines.append(f"🟢 {p['name']} {p['qty']:,}주 @{p['entry_price']:,}원")
    fill = [p["name"] for p in entered if p["kind"] == "filler"]
    if fill:
        lines.append("   조건용 1주: " + ", ".join(fill))
    if skipped:
        lines.append("⚠️ 못 삼: " + "; ".join(skipped[:5]))
    lines += ["", f"💰 {'종이 ' if config.DRY_RUN else ''}{_won(nav)}"]
    notify.send("\n".join(lines))
    return {"entered": entered, "skipped": skipped}


# ══════════════════════════════════════════════════════════════
# 15:40 마감 판정
# ══════════════════════════════════════════════════════════════
def evaluate(force: bool = False, send_report: bool = True) -> dict:
    if not _open_day(force):
        return {"skipped": "휴장일"}
    trader.sync_fills()
    ct = get()
    pending, lines, fillers = [], [], []
    for p in _positions():
        try:
            close = float(quotes.get_price(p["ticker"])["price"])
        except Exception as e:
            logger.error("%s 종가 조회 실패: %s", p["ticker"], e); continue
        peak = max(float(p.get("peak_close", 0) or 0), close)
        state.update_position(p["ticker"], peak_close=peak, hold_days=_held_days(p) + 1)
        reasons = exit_reasons({**p, "peak_close": peak, "entry_idx": 0}, close, _held_days(p))
        entry = float(p.get("entry_price", 0) or 0)
        if p.get("kind") == "filler":                       # 조건용 1주는 한 줄로 묶는다
            fillers.append(p.get("name", ""))
            if reasons:
                pending.append({"ticker": p["ticker"], "reason": "; ".join(reasons)})
            continue
        kind = {"crash": " 폭락"}.get(p.get("kind", ""), "")
        lines.append(f"📌 {p.get('name', '')} {int(p['qty']):,}주{kind}")
        lines.append(f"   {entry:,.0f} → {close:,.0f}  ({close / entry - 1:+.1%})" if entry else f"   종가 {close:,.0f}")
        lines.append(f"   고점 {peak:,.0f} · {_held_days(p)}일째")
        if reasons:
            lines.append("   ⏰ 내일 시가 매도: " + ", ".join(r.split(" (")[0] for r in reasons))
            pending.append({"ticker": p["ticker"], "reason": "; ".join(reasons)})
    if fillers:
        lines.append("📎 조건용 1주: " + ", ".join(fillers) + " → 내일 시가 매도")
    locked = bool(ct.get("locked"))
    nav = None
    try:
        nav = _nav()
    except Exception as e:
        logger.error("순자산 조회 실패: %s", e)
    anchor = float(ct.get("anchor", 0) or 0)
    if nav is not None and not locked and lock_hit(nav, anchor):
        locked = True
        put(locked=True, locked_date=_today())
        pending = [{"ticker": p["ticker"], "reason": f"목표 +{config.CT_LOCK_PCT:g}% 달성 락"} for p in _positions()]
        lines += ["", f"🎯 목표 달성!  {nav / anchor - 1:+.1%}", "   내일 시가 전량 매도 → 월말까지 현금"]
    put(pending_exit=pending)
    head = [f"🏁 {_md()} 마감" + ("  (종이 계좌)" if config.DRY_RUN else "")]
    if nav is not None:
        head.append(f"💰 {_won(nav)}" + (f"  (월초 {nav / anchor - 1:+.2%} · 목표 +{config.CT_LOCK_PCT:g}%)" if anchor else ""))
    head.append("")
    if not lines:
        lines.append("📌 보유 없음" + ("  🔒 락 상태" if locked else ""))
    if send_report:
        notify.send("\n".join(head + lines))
    return {"pending": pending, "locked": locked, "nav": nav}


# ══════════════════════════════════════════════════════════════
# 08:45 / 09:05 매도
# ══════════════════════════════════════════════════════════════
def _sell_at_open(pos: dict, reason: str) -> dict | None:
    try:
        q = quotes.get_price(pos["ticker"])
    except Exception as e:
        logger.error("%s 시세 조회 실패: %s", pos["ticker"], e)
        return None
    px = q["lower_limit"] or round_to_tick((q["prev_close"] or q["price"]) * 0.75, "up")
    qty = int(pos.get("qty", 0) or 0)
    try:
        for h in trading.get_balance()["holdings"]:
            if h["ticker"] == pos["ticker"]:
                qty = h["sellable_qty"] or h["qty"]
    except Exception as e:
        logger.warning("잔고 확인 실패, 기록 수량으로: %s", e)
    if qty <= 0:
        return state.close_position(pos["ticker"], float(pos["entry_price"]), f"{reason} (보유 없음)")
    r = trading.sell(pos["ticker"], qty, int(px))
    if not r["success"]:
        notify.send(f"⚠️ {pos.get('name', '')}({pos['ticker']}) 시가 매도 주문 거부: {r['message']}")
        return None
    rec = state.close_position(pos["ticker"], float(q["prev_close"] or q["price"]), reason, qty=qty)
    state.mark_traded_today(pos["ticker"])
    _paper_sell(rec)
    return rec


def sell_open(force: bool = False) -> dict:
    """08:45 — 어제 정한 매도를 장 시작 동시호가에."""
    if not _open_day(force):
        return {"skipped": "휴장일"}
    trader.sync_fills()
    ct = get()
    pend = {x["ticker"]: x["reason"] for x in ct.get("pending_exit") or []}
    if ct.get("locked"):
        for p in _positions():
            pend.setdefault(p["ticker"], "목표 달성 락")
    sold, left = [], []
    for p in _positions():
        if p["ticker"] not in pend:
            continue
        rec = _sell_at_open(p, pend[p["ticker"]])
        (sold if rec else left).append(p["ticker"] if not rec else rec)
    put(pending_exit=[{"ticker": t, "reason": pend[t]} for t in left])
    if sold:
        notify.send(f"🌅 {_md()} 시가 매도 주문\n\n" + "\n".join(f"🔴 {r.get('name', '')} {int(r.get('sold_qty', 0) or 0):,}주\n   {r.get('exit_reason', '').split(' (')[0]}" for r in sold))
    return {"sold": sold, "left": left}


def check_open(force: bool = False) -> dict:
    """09:05 — 아직 남은 매도 대상은 현재가 −2틱으로."""
    if not _open_day(force):
        return {"skipped": "휴장일"}
    ct = get()
    left = {x["ticker"]: x["reason"] for x in ct.get("pending_exit") or []}
    if not left:
        return {"sold": []}
    sold = []
    for p in _positions():
        if p["ticker"] in left:
            rec = trader.exit_position(p, left[p["ticker"]])
            if rec:
                _paper_sell(rec); sold.append(rec)
    put(pending_exit=[{"ticker": t, "reason": r} for t, r in left.items() if t not in {s["ticker"] for s in sold}])
    return {"sold": sold}


# ══════════════════════════════════════════════════════════════
# 텔레그램
# ══════════════════════════════════════════════════════════════
def status_text() -> str:
    ct = get()
    anchor = float(ct.get("anchor", 0) or 0)
    lines = ["🏆 대회 모드" + ("  (DRY_RUN)" if config.DRY_RUN else ""), f"{config.CT_LOOKBACK}일 모멘텀 1위 {config.CT_SLOTS}종목 · 손절 {config.CT_STOP_PCT:g}% · 고점 {config.CT_TRAIL_PCT:g}% · {config.CT_HOLD_DAYS}일", ""]
    try:
        nav = _nav()
        lines.append(f"💰 {'종이 ' if config.DRY_RUN else ''}{_won(nav)}" + (f"  (월초 {nav / anchor - 1:+.2%} · 목표 +{config.CT_LOCK_PCT:g}%)" if anchor else ""))
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")
    if ct.get("locked"):
        lines.append(f"🔒 {_md(ct.get('locked_date', ''))} 목표 달성 — 월말까지 현금")
    pos = _positions()
    if not pos:
        lines.append("📌 보유 없음")
    for p in pos:
        try:
            price = float(quotes.get_price(p["ticker"])["price"])
        except Exception:
            price = 0
        entry = float(p.get("entry_price", 0) or 0)
        lines.append("📌 " + _pos_line(p, price))
        lines.append(f"   {entry:,.0f} → {price:,.0f} · 고점 {float(p.get('peak_close', 0) or 0):,.0f}")
    pend = ct.get("pending_exit") or []
    if pend:
        names = {p["ticker"]: p.get("name", "") for p in pos}
        lines.append("⏰ 내일 시가 매도: " + ", ".join(f"{names.get(x['ticker'], x['ticker'])} ({x['reason'].split(' (')[0]})" for x in pend))
    return "\n".join(lines)
