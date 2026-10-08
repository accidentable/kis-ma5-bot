"""
jobs/contest.py — 대회 모드 (STRATEGY=contest): 한 달 안에 +30% 한 번을 노리는 집중 모멘텀. 계산은 core/contest.py.

하루 일정 (전부 거래일만)
  prep()       CT_PREP_TIME(08:20)   새 기간(달 또는 CONTEST_START)이면 기준 순자산 기록·락 해제. 대상 350종목 일봉(어제까지) 받아 캐시
  sell_open()  CT_SELL_TIME(08:45)   어제 마감에 정한 매도(손절·추적·만료·락·조건용) 를 장 시작 동시호가에 (하한가 지정가 → 시가 체결)
  check_open() CT_CHECK_TIME(09:05)  아직 안 팔린 것은 현재가 −2틱으로 다시
  scan()       CT_SCAN_TIME(15:05)   현재가 350개 → 폭락 판정 → 후보 순위 → 오늘 살 것 결정. 폭락 전환이면 보유를 지금(장중) 판다
  buy_close()  CT_BUY_TIME(15:20)    scan 이 정한 종목을 장 마감 동시호가 매수 (현재가 + CT_BUY_TICKS 틱, 상한가 이내)
  evaluate()   CT_EVAL_TIME(15:40)   종가로 손절·추적·만료 판정, 순자산으로 목표 락 판정 → 내일 아침 매도 목록. 마감 리포트
상태는 state.json 의 "contest" 에 둔다: month, anchor(기간 시작 순자산), locked, hist(일봉 캐시), universe, plan(오늘 결정), pending_exit, fillers_done
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


def _period() -> str:
    """
    목표 락·기준 순자산·조건용 매매를 세는 기간 키. 보통은 달("2026-10")이지만,
    CONTEST_START 가 있고 그날이 지났으면 대회 시작일("2026-10-12") 하나로 끝까지 간다.
    대회가 달을 넘어가도(10/12~11/20) 11/1 에 기준이 리셋되거나 락이 풀리지 않게 하려는 것.
    """
    start = config.CONTEST_START
    if start and _today() >= start:
        return start
    return date.today().strftime("%Y-%m")


def lock_pct(ct: dict | None = None) -> float:
    """목표 락 %. 텔레그램 /target 으로 바꾼 값이 있으면 그걸, 없으면 CT_LOCK_PCT. 0 = 락 없음."""
    v = (get() if ct is None else ct).get("lock_pct")
    return float(v) if v is not None else config.CT_LOCK_PCT


def _target(pct: float) -> str:
    return f"+{pct:g}%" if pct > 0 else "없음"


def _period_label(key: str) -> str:
    return f"대회 {key[5:7]}/{key[8:10]}~" if len(key) == 10 else key


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
def _period_reset(ct: dict) -> dict:
    if ct.get("month") == _period():
        return ct
    if config.DRY_RUN and "paper_cash" not in ct:
        put(paper_cash=config.CT_PAPER_CAP)
    try:
        anchor = _nav()
    except Exception as e:
        logger.error("기간 시작 순자산 조회 실패: %s", e)
        anchor = float(ct.get("anchor", 0) or 0)
    ct = put(month=_period(), anchor=anchor, locked=False, locked_date="", fillers_done="")
    pct = lock_pct(ct)
    notify.send(f"📅 대회 모드 새 기간 {_period_label(_period())} — 기준 순자산 {anchor:,.0f}원, 목표 {_target(pct)}"
                + (f" ({anchor * (1 + pct / 100):,.0f}원)" if pct > 0 else ""))
    return ct


def build_universe() -> list[dict]:
    stocks = universe.get_large_universe(config.CT_UNIVERSE_PULL)
    return split_universe(stocks, config.CT_KOSPI_N, config.CT_KOSDAQ_N)


def prep(force: bool = False) -> dict:
    if not _open_day(force):
        return {"skipped": "휴장일"}
    _auto_reconcile()
    ct = _period_reset(get())
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
        lines.append(f"💰 {'종이 ' if config.DRY_RUN else ''}{_won(nav)}" + (f"  (기준 대비 {nav / anchor - 1:+.2%})" if anchor else ""))
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
        lines += ["", f"🔒 {_md(ct.get('locked_date', ''))} 목표 달성 — 이번 기간 매수 없음"]
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
        rest = [s for s in cands if s.ticker not in held_t]
        picks = rest[:config.CT_SLOTS]

        def _crash_item(s):
            return {"ticker": s.ticker, "name": s.name, "price": s.price, "upper_limit": s.upper_limit, "kind": "crash",
                    "signal": {"change_pct": s.change_pct, "mkt": cinfo["mkt"], "val_ratio": s.val_ratio}}
        plan.update(mode="crash", stats=stats, sell_now=[p["ticker"] for p in held],
                    buy=[_crash_item(s) for s in picks],
                    backup=[_crash_item(s) for s in rest[config.CT_SLOTS:config.CT_SLOTS + config.CT_FALLBACK_N]])
        return plan
    if in_crash:
        plan["mode"] = "crash_hold"
        return plan
    cands, stats = rank_momentum(snaps)
    free = config.CT_SLOTS - len(held)
    rest = [s for s in cands if s.ticker not in held_t]
    picks = rest[:max(free, 0)]

    def _mom_item(s):
        return {"ticker": s.ticker, "name": s.name, "price": s.price, "upper_limit": s.upper_limit, "kind": "momentum",
                "signal": {"ret_n": s.ret_n, "change_pct": s.change_pct}}
    plan.update(mode="momentum", stats=stats,
                buy=[_mom_item(s) for s in picks],
                # 고른 종목이 1주도 못 살 만큼 비쌀 때(소액 실전) 15:20 에 내려갈 다음 순위
                backup=[_mom_item(s) for s in rest[len(picks):len(picks) + config.CT_FALLBACK_N]] if picks else [],
                top=[{"ticker": s.ticker, "name": s.name, "ret_n": s.ret_n, "change_pct": s.change_pct} for s in cands[:10]])
    if picks and config.CT_FILLER_N > 0 and ct.get("fillers_done") != _period():
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
        lines.append("🔒 이번 기간 목표 달성 — 매수 없음")
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
def _limit_qty(b: dict, budget: float) -> tuple[int, int]:
    """장마감 동시호가 지정가와 수량. 조건용은 1주, 나머지는 예산(과 실계좌 미수없는 매수가능 수량) 안에서."""
    try:
        q = quotes.get_price(b["ticker"])
        price, upper = float(q["price"]), float(q.get("upper_limit") or 0)
    except Exception:
        price, upper = float(b["price"]), float(b.get("upper_limit") or 0)
    limit = offset_ticks(price, config.CT_BUY_TICKS)
    if upper:
        limit = min(limit, int(upper))
    if b["kind"] == "filler":
        return limit, 1
    if config.DRY_RUN:
        return limit, int(budget // limit)
    try:
        info = trading.get_buyable(b["ticker"], limit)
        return limit, min(int(info["qty_no_margin"]), int(budget // limit))
    except Exception as e:
        logger.warning("%s 매수가능 조회 실패, 예산으로: %s", b["ticker"], e)
        return limit, int(budget // limit)


def _first_affordable(backups: list[dict], held: set[str], budget: float) -> tuple[dict, int, int] | None:
    """다음 순위 중 1주 이상 살 수 있는 첫 종목. 15:05 가격으로 이미 예산을 넘는 건 조회도 안 한다."""
    while backups:
        b = backups.pop(0)
        if b["ticker"] in held or float(b.get("price") or 0) > budget:
            continue
        limit, qty = _limit_qty(b, budget)
        if qty > 0:
            return b, limit, qty
    return None


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
    budget = nav * 0.995 / max(config.CT_SLOTS, 1)
    backups = list(plan.get("backup") or [])
    for b in plan["buy"]:
        if b["ticker"] in held:
            continue
        limit, qty = _limit_qty(b, budget)
        if qty <= 0 and b["kind"] != "filler":
            # 1주도 못 사면(소액 계좌에 비싼 종목) 다음 순위로 내려간다. 순위는 15:05 판정 그대로.
            alt = _first_affordable(backups, held, budget)
            if alt:
                skipped.append(f"{b['name']} 1주 {limit:,}원 > 예산 → {alt[0]['name']}")
                b, limit, qty = alt
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
        held.add(b["ticker"])
        entered.append(pos)
    if any(p["kind"] == "filler" for p in entered):
        put(fillers_done=_period())
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
    _auto_reconcile()
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
    pct = lock_pct(ct)
    if nav is not None and not locked and lock_hit(nav, anchor, pct):
        locked = True
        put(locked=True, locked_date=_today())
        pending = [{"ticker": p["ticker"], "reason": f"목표 +{pct:g}% 달성 락"} for p in _positions()]
        lines += ["", f"🎯 목표 달성!  {nav / anchor - 1:+.1%}", "   내일 시가 전량 매도 → 기간 끝까지 현금"]
    put(pending_exit=pending)
    head = [f"🏁 {_md()} 마감" + ("  (종이 계좌)" if config.DRY_RUN else "")]
    if nav is not None:
        head.append(f"💰 {_won(nav)}" + (f"  (기준 대비 {nav / anchor - 1:+.2%} · 목표 {_target(pct)})" if anchor else ""))
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
        lines.append(f"💰 {'종이 ' if config.DRY_RUN else ''}{_won(nav)}" + (f"  (기준 대비 {nav / anchor - 1:+.2%} · 목표 {_target(lock_pct(ct))})" if anchor else ""))
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")
    if ct.get("locked"):
        lines.append(f"🔒 {_md(ct.get('locked_date', ''))} 목표 달성 — 기간 끝까지 현금")
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


# ══════════════════════════════════════════════════════════════
# 텔레그램 /target — 목표 락 보기·바꾸기
# ══════════════════════════════════════════════════════════════
TARGET_HELP = "/target          지금 목표 보기\n/target 70       목표 +70% 로\n/target off      락 없이 끝까지 매매"


def set_target(arg: str | None = None) -> str:
    """
    목표 락 % 를 보거나 바꾼다. 봇마다(실전·모의) 자기 상태에 따로 저장되고, 기간이 바뀌어도 유지된다.
    이미 락이 걸렸는데 새 목표가 지금 수익보다 높으면(또는 off) 락을 풀고 내일 시가 락 매도도 취소한다.
    순위권 밖이라 더 밀어붙여야 할 때 쓴다.
    """
    ct = get()
    anchor = float(ct.get("anchor", 0) or 0)
    try:
        nav = _nav()
    except Exception as e:
        logger.warning("순자산 조회 실패: %s", e)
        nav = None
    now = f"{nav / anchor - 1:+.2%}" if (nav and anchor) else "?"

    if arg is None:
        pct = lock_pct(ct)
        goal = f" ({_won(anchor * (1 + pct / 100))})" if (pct > 0 and anchor) else ""
        state_txt = f"🔒 락 ({_md(ct.get('locked_date', ''))} 달성)" if ct.get("locked") else "매매 중"
        return f"🎯 목표 {_target(pct)}{goal}\n기준 {_won(anchor)} · 지금 {now} · {state_txt}\n\n{TARGET_HELP}"

    a = arg.strip().lower().lstrip("+").rstrip("%")
    if a in ("off", "없음", "0"):
        pct = 0.0
    else:
        try:
            pct = float(a)
        except ValueError:
            return f"⚠️ 숫자로 넣어줘: {arg}\n\n{TARGET_HELP}"
        if not 0 < pct <= 1000:
            return f"⚠️ 0 초과 1000 이하로: {arg}"

    before = lock_pct(ct)
    put(lock_pct=pct)
    lines = [f"🎯 목표 {_target(before)} → {_target(pct)}  (기준 {_won(anchor)} · 지금 {now})"]
    if ct.get("locked"):
        if pct <= 0 or (nav is not None and not lock_hit(nav, anchor, pct)):
            pend = [x for x in ct.get("pending_exit") or [] if "락" not in x.get("reason", "")]
            put(locked=False, locked_date="", pending_exit=pend)
            lines.append("🔓 락 해제 — 내일 시가 락 매도 취소, 15:20 부터 다시 매수")
        else:
            lines.append("🔒 지금 수익이 새 목표도 넘어서 락 유지")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 텔레그램 /deposit — 입출금을 기준 순자산에 반영
# ══════════════════════════════════════════════════════════════
DEPOSIT_HELP = ("/deposit 20만      20만원 입금 → 기준에 더함\n/deposit -10만     10만원 출금 → 기준에서 뺌\n"
                "/deposit          입출금 기록 보기\n넣은 날 15:40 마감 판정 전에 보내야 입금이 수익으로 안 잡힌다.")


def _parse_amount(arg: str) -> float | None:
    """'20만' · '200000' · '-10만' · '1억 5천만' → 원. 못 읽으면 None."""
    from core.manual import _parse_money
    t = arg.replace(",", "").replace(" ", "").strip()
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    if t.replace(".", "", 1).isdigit():
        v = float(t)
    else:
        v = _parse_money(t)
    return sign * v if v else None


def deposit(arg: str | None = None) -> str:
    """
    입금(+)·출금(−)을 기준 순자산에 더해, 돈을 넣고 뺀 게 수익률·목표 락에 잡히지 않게 한다.
    기준은 기간 시작 순자산이라 입금을 그냥 두면 +100% 수익으로 보고 락을 걸어 전량 매도해버린다.
    이미 락이 걸렸는데 고친 기준으로는 목표 미달이면 락을 풀고 시가 락 매도도 취소한다.
    """
    ct = get()
    anchor = float(ct.get("anchor", 0) or 0)
    log = list(ct.get("deposits") or [])
    if arg is None:
        hist = "\n".join(f"· {_md(d['date'])} {d['amount']:+,.0f}원" for d in log[-10:]) or "· 없음"
        return f"💵 기준 순자산 {anchor:,.0f}원\n입출금 기록\n{hist}\n\n{DEPOSIT_HELP}"

    amt = _parse_amount(arg)
    if not amt:
        return f"⚠️ 금액을 못 읽었다: {arg}\n\n{DEPOSIT_HELP}"
    if anchor + amt <= 0:
        return f"⚠️ 기준 순자산 {anchor:,.0f}원보다 많이 뺄 수는 없다"

    new_anchor = anchor + amt
    log.append({"date": _today(), "amount": amt})
    fields = {"anchor": new_anchor, "deposits": log}
    if config.DRY_RUN:                                   # 종이 계좌는 현금도 같이 움직인다
        fields["paper_cash"] = float(ct.get("paper_cash", config.CT_PAPER_CAP) or 0) + amt
    put(**fields)

    lines = [f"💵 {'입금' if amt > 0 else '출금'} {abs(amt):,.0f}원 반영",
             f"기준 순자산 {anchor:,.0f} → {new_anchor:,.0f}원"]
    try:
        nav = _nav()
        pct = lock_pct(ct)
        lines.append(f"지금 {nav / new_anchor - 1:+.2%} · 목표 {_target(pct)}")
        if ct.get("locked") and not lock_hit(nav, new_anchor, pct):
            pend = [x for x in ct.get("pending_exit") or [] if "락" not in x.get("reason", "")]
            put(locked=False, locked_date="", pending_exit=pend)
            lines.append("🔓 입금 때문에 걸린 락이라 해제 — 시가 락 매도 취소")
    except Exception as e:
        logger.warning("순자산 조회 실패: %s", e)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 손매매 반영 — /sync · /adopt, 08:20 준비와 15:40 마감 때 자동
# ══════════════════════════════════════════════════════════════
def _sold_price_today(ticker: str) -> float | None:
    """오늘 이 종목 매도 체결 평균가. 없거나 조회 실패면 None."""
    try:
        fills = [f for f in trading.get_today_fills() if f["ticker"] == ticker and f["side"] == "sell"]
    except Exception as e:
        logger.warning("%s 체결 조회 실패: %s", ticker, e)
        return None
    qty = sum(int(f["qty"]) for f in fills)
    return sum(int(f["qty"]) * float(f["avg_price"]) for f in fills) / qty if qty else None


def _auto_reconcile() -> None:
    """작업 시작 때 손매매를 반영한다. 실패해도 작업은 계속한다."""
    try:
        reconcile(include_today=True, announce=True)
    except Exception as e:
        logger.warning("계좌 맞춤 실패: %s", e)


def reconcile(include_today: bool = True, announce: bool = True) -> dict:
    """
    계좌 잔고를 진실로 보고 봇 포지션을 맞춘다 (손매매·앱 매매 반영).
      봇엔 있는데 계좌에 없음 → '수동 매도' 로 이력에 남기고 정리 (오늘 매도 체결가, 없으면 현재가)
      수량이 다름           → 계좌 수량으로
      계좌엔 있는데 봇은 모름 → 수동 보유. 봇은 안 건드린다 (/adopt 로 넘길 수 있다)
    include_today=False 면 오늘 산 미체결 포지션은 건드리지 않는다 (장중 /sync — 동시호가 체결 전일 수 있다).
    DRY_RUN 은 종이 포지션이라 실계좌와 비교하지 않는다.
    """
    if config.DRY_RUN:
        return {"skipped": "DRY_RUN"}
    bal = trading.get_balance()
    held = {h["ticker"]: h for h in bal["holdings"]}
    closed, resized, unfilled = [], [], []
    for p in _positions():
        h = held.get(p["ticker"])
        if not include_today and p.get("entry_date") == _today() and not p.get("filled"):
            continue
        if h is None and p.get("entry_date") == _today() and not p.get("filled"):
            state.remove_position(p["ticker"])               # 오늘 낸 매수가 안 채워졌다 — 판 게 아니라 이력엔 안 남긴다
            unfilled.append({"ticker": p["ticker"], "name": p.get("name", "")})
            continue
        if h is None:
            px = _sold_price_today(p["ticker"])
            if px is None:
                try:
                    px = float(quotes.get_price(p["ticker"])["price"])
                except Exception:
                    px = float(p.get("entry_price", 0) or 0)
            rec = state.close_position(p["ticker"], px, "수동 매도 (계좌에 없음)")
            if rec:
                closed.append(rec)
        elif int(h["qty"]) != int(p["qty"]):
            state.update_position(p["ticker"], qty=int(h["qty"]))
            resized.append({"ticker": p["ticker"], "name": p.get("name", ""), "from": int(p["qty"]), "to": int(h["qty"])})
    if closed or unfilled:
        gone = {r["ticker"] for r in closed + unfilled}
        put(pending_exit=[x for x in get().get("pending_exit") or [] if x["ticker"] not in gone])
    bot_t = {p["ticker"] for p in _positions()}
    manual = [h for t, h in held.items() if t not in bot_t]
    out = {"closed": closed, "resized": resized, "unfilled": unfilled, "manual": manual, "cash": bal["cash"], "nav": bal["net_asset"]}
    if announce and (closed or resized or unfilled):
        notify.send(sync_text(out, changed_only=True))
    return out


def sync_text(r: dict, changed_only: bool = False) -> str:
    if r.get("skipped"):
        return "DRY_RUN 이라 실계좌와 맞추지 않는다."
    lines = ["🔄 계좌 맞춤"]
    for c in r["closed"]:
        lines.append(f"🔴 {c.get('name', '')} {int(c.get('sold_qty', 0)):,}주 정리 — 계좌에 없음 (수동 매도로 기록, {c['pnl_pct']:+.2f}%)")
    for x in r.get("unfilled", []):
        lines.append(f"⚪ {x['name']} 오늘 매수 미체결 — 포지션 지움")
    for x in r["resized"]:
        lines.append(f"✏️ {x['name']} 수량 {x['from']:,} → {x['to']:,}주")
    if not changed_only:
        if not (r["closed"] or r["resized"] or r.get("unfilled")):
            lines.append("봇 포지션 = 계좌 (바뀐 것 없음)")
        bot = _positions()
        lines.append("")
        lines.append("🤖 봇 관리: " + (", ".join(f"{p.get('name', '')} {int(p['qty']):,}주" for p in bot) if bot else "없음"))
    if r["manual"]:
        lines.append("✋ 수동 보유 (봇이 안 건드림): " + ", ".join(f"{h['name']}({h['ticker']}) {h['qty']:,}주" for h in r["manual"]))
        if not changed_only:
            lines.append("   봇에게 넘기려면 /adopt 종목코드")
    if not changed_only:
        lines.append(f"\n💰 순자산 {r['nav']:,.0f}원 · 예수금 {r['cash']:,.0f}원")
    return "\n".join(lines)


def adopt(ticker: str | None) -> str:
    """수동으로 산 종목을 봇 포지션으로 넘긴다. 그날부터 손절·추적·보유일 규칙을 적용한다."""
    if not ticker:
        return "종목코드를 넣어줘. 예) /adopt 053800"
    if config.DRY_RUN:
        return "DRY_RUN 에선 실계좌 종목을 넘길 수 없다."
    ticker = ticker.strip()
    if state.get_position(ticker):
        return f"{ticker} 는 이미 봇이 관리 중이다."
    h = next((h for h in trading.get_balance()["holdings"] if h["ticker"] == ticker), None)
    if h is None:
        return f"계좌에 {ticker} 가 없다. /sync 로 보유를 확인해줘."
    avg, px = float(h["avg_price"]), float(h["price"] or h["avg_price"])
    pos = {"ticker": ticker, "name": h["name"], "qty": int(h["qty"]), "entry_price": round(avg), "entry_date": _today(),
           "entry_order_no": "", "entry_org_no": "", "strategy": "contest", "kind": "momentum",
           "source": "수동 편입", "signal": {}, "peak_close": max(avg, px),
           "breakout_price": 0, "take_profit_price": 0, "stop_price": 0, "hold_days": 1, "atr": 0,
           "filled": True, "dry_run": False}
    state.add_position(pos)
    n = sum(1 for p in _positions() if p.get("kind") != "filler")
    over = f"\n⚠️ 봇 포지션 {n}개 > 슬롯 {config.CT_SLOTS}개 — 팔릴 때까지 새로 안 산다" if n > config.CT_SLOTS else ""
    return (f"🤝 {h['name']} {int(h['qty']):,}주 봇에게 넘김 (평단 {avg:,.0f}원)\n"
            f"손절 {avg * (1 - config.CT_STOP_PCT / 100):,.0f}원 · 고점 −{config.CT_TRAIL_PCT:g}% · 오늘부터 {config.CT_HOLD_DAYS}거래일{over}")
