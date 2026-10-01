"""
tests/test_contest.py — 대회 모드 (STRATEGY=contest) 흐름 점검. API 키 없이 돈다.

    python tests/test_contest.py

가짜 시세·잔고·주문으로 준비 → 판정 → 종가 매수 → 마감 판정 → 시가 매도 → 목표 락 → 폭락 전환 → 월초 리셋을 돈다.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.DATA_DIR = tempfile.mkdtemp(prefix="ct-test-")
config.STATE_BACKEND = "local"
config.DRY_RUN = True
config.TELEGRAM_BOT_TOKEN = ""
config.TELEGRAM_ALLOWED_CHAT_IDS = []
config.STRATEGY = "contest"
config.CT_SLOTS = 1
config.CT_LOOKBACK = 20
config.CT_TOP_PCT = 30.0          # 표본이 작아 상위 30% 로
config.CT_STOP_PCT = 10.0
config.CT_TRAIL_PCT = 10.0
config.CT_HOLD_DAYS = 10
config.CT_LOCK_PCT = 30.0
config.CT_CRASH_ENABLED = True
config.CT_CRASH_MKT_PCT = 3.0
config.CT_CRASH_SIGMA = 3.0
config.CT_CRASH_DROP_PCT = 7.0
config.CT_CRASH_VR_MAX = 3.0
config.CT_CRASH_IDIO_MULT = 3.0
config.CT_CRASH_HOLD = 5
config.CT_FILLER_N = 2
config.CT_BUY_TICKS = 5
config.CT_KOSPI_N = 6
config.CT_KOSDAQ_N = 4
config.CT_CRASH_MIN_N = 5

from core import contest as core_ct, notify, state, trader, universe as universe_mod   # noqa: E402
from core.kis import quotes, trading                                                 # noqa: E402
from jobs import contest                                                             # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


# ── 가짜 세상 ──────────────────────────────────────────────────
# (코드, 이름, 시장, 시총, 20일 전 종가, 오늘 가격) — 20일 수익률 = 오늘/20일전 − 1
STOCKS = [
    ("100010", "폭주A", "KOSPI", 9e12, 10_000, 14_500),     # +45% → 1위
    ("100020", "강세B", "KOSPI", 8e12, 10_000, 13_000),     # +30% → 2위
    ("100030", "상승C", "KOSPI", 7e12, 10_000, 12_000),     # +20%
    ("100040", "상한가D", "KOSPI", 6e12, 10_000, 16_000),   # +60% 이지만 오늘 상한가 → 제외
    ("100050", "하한가이력E", "KOSPI", 5e12, 10_000, 15_000),  # +50% 이지만 20일 안 하한가 → 제외
    ("100060", "보합F", "KOSPI", 4e12, 10_000, 10_100),
    ("100070", "코스피7위G", "KOSPI", 3e12, 10_000, 10_000),  # 코스피 7위 → 대상 밖 (CT_KOSPI_N=6)
    ("200010", "코닥H", "KOSDAQ", 2e12, 10_000, 11_000),
    ("200020", "코닥I", "KOSDAQ", 1.5e12, 10_000, 9_000),
    ("200030", "코닥J", "KOSDAQ", 1.2e12, 10_000, 10_500),
    ("200040", "신규K", "KOSDAQ", 1.1e12, 10_000, 12_500),    # 일봉 5개뿐 → 제외
    ("200050", "코닥5위L", "KOSDAQ", 1.0e12, 10_000, 10_000),  # 코스닥 5위 → 대상 밖
]
INFO = {s[0]: s for s in STOCKS}
TODAY = date.today().strftime("%Y%m%d")
PRICE = {s[0]: float(s[5]) for s in STOCKS}          # 오늘 가격 (시험 중 바꾼다)
CHG = {s[0]: 0.5 for s in STOCKS}                     # 오늘 등락률 %
HELD = {}                                             # ticker → 보유 거래일 수
NAV = {"v": 100_000_000.0}
ORDERS, SENT = [], []


def fake_universe(n, force=False):
    return [{"ticker": c, "name": nm, "market": m, "marcap": cap} for c, nm, m, cap, *_ in STOCKS]


def fake_candles(ticker, days=40):
    c0, now = INFO[ticker][4], INFO[ticker][5]
    n = 5 if ticker == "200040" else 45
    out = []
    for i in range(n):
        d = f"2026{(i % 12) + 1:02d}{(i % 28) + 1:02d}"               # 오늘보다 작기만 하면 된다
        close = c0 + (now - c0) * max(0, i - (n - 21)) / 20           # 마지막 20일 동안 c0 → now 로
        if ticker == "100050" and i == n - 10:
            close = c0 * 0.6                                           # 하한가 흉내
        out.append({"date": f"2025{d[4:]}", "open": close, "high": close, "low": close, "close": close, "volume": 1000, "value": 1e9})
    return out


def fake_price(ticker, market="J"):
    p = PRICE[ticker]
    prev = p / (1 + CHG[ticker] / 100)
    up = round(prev * 1.3); lo = round(prev * 0.7)
    if ticker == "100040":
        p = up; CHG[ticker] = 30.0
    return {"ticker": ticker, "name": INFO[ticker][1], "price": p, "prev_close": prev, "change_pct": CHG[ticker], "volume": 1,
            "value": 2e9, "upper_limit": up, "lower_limit": lo, "status_code": "00", "is_halted": False, "is_watch": False, "open": p, "high": p, "low": p}


def fake_balance():
    return {"holdings": [{"ticker": p["ticker"], "name": p.get("name", ""), "qty": p["qty"], "sellable_qty": p["qty"], "avg_price": p["entry_price"],
                          "price": PRICE[p["ticker"]], "eval_amount": 0, "pnl_amount": 0, "pnl_pct": 0} for p in state.get_positions()],
            "cash": NAV["v"], "d2_cash": NAV["v"], "total_eval": NAV["v"], "net_asset": NAV["v"], "pnl_amount": 0}


def fake_order(side):
    def f(ticker, qty, price, market=False):
        ORDERS.append({"side": side, "ticker": ticker, "qty": qty, "price": price})
        return {"success": True, "order_no": f"T{len(ORDERS):04d}", "org_no": "0", "message": "TEST", "ticker": ticker, "qty": qty, "price": price, "side": side, "dry_run": True}
    return f


def fake_exit(pos, reason, price=None):
    ORDERS.append({"side": "sell", "ticker": pos["ticker"], "qty": pos["qty"], "price": PRICE[pos["ticker"]], "now": True})
    return state.close_position(pos["ticker"], PRICE[pos["ticker"]], reason)


universe_mod.get_large_universe = fake_universe
quotes.get_daily_candles = fake_candles
quotes.get_price = fake_price
quotes.is_open_day = lambda d=None: True
trading.get_balance = fake_balance
trading.get_buyable = lambda t, px: {"cash": NAV["v"], "qty_no_margin": 10_000_000, "amount_no_margin": NAV["v"], "qty_max": 10_000_000, "amount_max": NAV["v"]}
trading.buy = fake_order("buy")
trading.sell = fake_order("sell")
trader.sync_fills = lambda: None
trader.exit_position = fake_exit
notify.send = lambda text, chat_id=None: SENT.append(text)
contest._held_days = lambda pos: HELD.get(pos["ticker"], 0)


def main() -> int:
    print("── 준비 ─────────────────────────────")
    r = contest.prep(force=True)
    ct = contest.get()
    uni = ct["universe"]
    check("대상 = 코스피 6 + 코스닥 4", [u["ticker"] for u in uni] == ["100010", "100020", "100030", "100040", "100050", "100060", "200010", "200020", "200030", "200040"], str([u["ticker"] for u in uni]))
    check("일봉 캐시 생성", r["hist"] == 10 and len(ct["hist"]["100010"]["closes"]) == 45)
    check("월초 기준 순자산 기록 · 락 해제", ct["anchor"] == 100_000_000 and ct["locked"] is False and ct["month"] == date.today().strftime("%Y-%m"))
    brief = SENT[-1]
    check("아침 브리핑: 어제 종가 기준 상위 후보 + 순자산 (상한가는 아직 모르니 D 가 1위)", "아침 브리핑" in brief and "1. 상한가D" in brief and "2. 폭주A" in brief and "💰 종이" in brief, brief.replace("\n", " / "))

    print("── 판정 (모멘텀) ─────────────────────")
    snaps, fail = contest.snapshots(ct)
    check("현재가 10개", len(snaps) == 10 and not fail)
    plan = contest.decide(snaps, ct)
    buys = [b for b in plan["buy"] if b["kind"] == "momentum"]
    check("모멘텀 모드", plan["mode"] == "momentum", plan["mode"])
    check("1위 = 20일 +45% 폭주A (상한가 D · 하한가이력 E · 신규 K 제외)", [b["ticker"] for b in buys] == ["100010"], str([b["ticker"] for b in buys]))
    rj = plan["stats"]["rejects"]
    check("탈락 사유: 상한가 1 · 하한가이력 1 · 일봉부족 1", rj.get("상한가") == 1 and rj.get("하한가이력") == 1 and rj.get("일봉부족") == 1, str(rj))
    fill = [b["ticker"] for b in plan["buy"] if b["kind"] == "filler"]
    check("조건용 1주 후보 2개 = 다음 순위 (강세B · 상승C)", fill == ["100020", "100030"], str(fill))

    print("── 종가 매수 ─────────────────────────")
    contest.put(plan=plan)
    r = contest.buy_close(force=True)
    pos = {p["ticker"]: p for p in contest._positions()}
    check("폭주A 매수: 종이 1억 × 99.5% / 1슬롯 ÷ (현재가+5틱)", "100010" in pos and pos["100010"]["qty"] == int(100_000_000 * 0.995 // pos["100010"]["entry_price"]), str(pos.get("100010")))
    check("종이 현금 차감", contest.get()["paper_cash"] < 100_000_000 - pos["100010"]["qty"] * pos["100010"]["entry_price"] + 1, str(contest.get()["paper_cash"]))
    check("매수 지정가 = 현재가 + 5틱 (14,500 → 14,550, 호가 10원)", pos.get("100010", {}).get("entry_price") == 14_550, str(pos.get("100010", {}).get("entry_price")))
    check("조건용 1주 2종목", all(pos[t]["qty"] == 1 and pos[t]["kind"] == "filler" for t in ("100020", "100030")))
    check("조건용 거래는 달에 한 번 표시", contest.get().get("fillers_done") == date.today().strftime("%Y-%m"))
    check("추가 매수 없음 (슬롯 참)", contest.decide(snaps, contest.get())["buy"] == [])

    print("── 마감 판정 · 시가 매도 ───────────────")
    PRICE["100010"] = 15_000
    r = contest.evaluate(force=True, send_report=False)
    pend = {x["ticker"]: x["reason"] for x in r["pending"]}
    check("+1.7% 보유 유지, 조건용 2개만 내일 매도", "100010" not in pend and set(pend) == {"100020", "100030"}, str(pend))
    check("고점 갱신 15,000", float(contest._positions()[0]["peak_close"]) == 15_000)
    ORDERS.clear()
    r = contest.sell_open(force=True)
    sells = [o for o in ORDERS if o["side"] == "sell"]
    check("조건용 2종목 하한가 지정가 매도 (시가 체결)", sorted(o["ticker"] for o in sells) == ["100020", "100030"] and all(o["price"] == round(PRICE[o["ticker"]] / 1.005 * 0.7) for o in sells), str(sells))
    check("이력에 사유 기록", any("조건용" in h.get("exit_reason", "") for h in state.get_history()))
    PRICE["100010"] = 13_000                      # 매수가 14,750 대비 −11.9%
    r = contest.evaluate(force=True, send_report=False)
    check("손절 −10% → 내일 시가 매도 (고점 15,000 대비 추적도 걸림)", r["pending"] and "손절" in r["pending"][0]["reason"] and "추적" in r["pending"][0]["reason"], str(r["pending"]))
    ORDERS.clear(); contest.sell_open(force=True)
    check("손절 매도 주문 · 포지션 비움", ORDERS and ORDERS[0]["ticker"] == "100010" and not contest._positions())

    print("── 보유 만료 · 추적 ─────────────────────")
    PRICE["100010"] = 14_500; CHG["100010"] = 0.5
    contest.put(plan=contest.decide(contest.snapshots(contest.get())[0], contest.get())); contest.buy_close(force=True)
    HELD["100010"] = 10
    r = contest.evaluate(force=True, send_report=False)
    check("보유 10일 만료 → 매도 예정", r["pending"] and "만료" in r["pending"][0]["reason"], str(r["pending"]))
    HELD["100010"] = 3
    PRICE["100010"] = 20_000; contest.evaluate(force=True, send_report=False)
    PRICE["100010"] = 17_900                      # 고점 20,000 대비 −10.5%, 매수가 대비 +21%
    r = contest.evaluate(force=True, send_report=False)
    check("추적 −10% → 매도 예정 (손절 아님)", r["pending"] and "추적" in r["pending"][0]["reason"] and "손절" not in r["pending"][0]["reason"], str(r["pending"]))
    contest.sell_open(force=True)

    print("── 목표 락 ─────────────────────────────")
    PRICE["100010"] = 14_500
    contest.put(plan=contest.decide(contest.snapshots(contest.get())[0], contest.get())); contest.buy_close(force=True)
    PRICE["100010"] = 14_550 * 1.33                 # 종이 계좌: 보유 +33% → 순자산 약 +32.8%
    r = contest.evaluate(force=True, send_report=False)
    check("종이 순자산 +32% → 락, 전량 매도 예정", r["locked"] and r["pending"] and "락" in r["pending"][0]["reason"], str(r))
    plan = contest.decide(contest.snapshots(contest.get())[0], contest.get())
    check("락 상태면 판정 = locked, 매수 없음", plan["mode"] == "locked" and not plan["buy"])
    ORDERS.clear(); contest.sell_open(force=True)
    check("락 매도 체결 주문", ORDERS and ORDERS[0]["ticker"] == "100010" and not contest._positions())
    check("락 상태에선 종가 매수 건너뜀", contest.buy_close(force=True).get("skipped") == "locked")

    print("── 월초 리셋 ────────────────────────────")
    contest.put(month="2000-01")
    contest.prep(force=True)
    ct = contest.get()
    check("새 달: 기준 순자산 = 지금 종이 순자산(락 매도 뒤 현금), 락 해제, 조건용 리셋", abs(ct["anchor"] - ct["paper_cash"]) < 1 and ct["anchor"] > 125_000_000 and not ct["locked"] and ct["fillers_done"] == "", str(ct["anchor"]))
    contest.put(paper_cash=100_000_000.0, anchor=100_000_000.0)

    print("── 폭락 전환 ────────────────────────────")
    NAV["v"] = 100_000_000
    config.CT_FILLER_N = 0
    contest.put(plan=contest.decide(contest.snapshots(contest.get())[0], contest.get())); contest.buy_close(force=True)
    check("폭락 전 보유 = 폭주A", [p["ticker"] for p in contest._positions()] == ["100010"])
    for t in PRICE:                                # 시장 −5%: 전 종목 −5%, 급락주 셋
        CHG[t] = -5.0; PRICE[t] = INFO[t][5] * 0.95
    CHG["100030"] = -9.0; PRICE["100030"] = INFO["100030"][5] * 0.91       # 급락 1
    CHG["200010"] = -12.0; PRICE["200010"] = INFO["200010"][5] * 0.88      # 급락 2 (더 큼)
    CHG["100060"] = -22.0; PRICE["100060"] = INFO["100060"][5] * 0.78      # 시장(약 −6.8%)의 3배 넘게 → 종목 악재 의심 제외
    CHG["200020"] = -30.0; PRICE["200020"] = INFO["200020"][5] * 0.70      # 하한가 → 제외
    CHG["100040"] = 30.0
    snaps, _ = contest.snapshots(contest.get())
    on, info = core_ct.crash_signal(snaps)
    check("시장 −3% & 3σ 둘 다 → 폭락 판정", on, str(info))
    ORDERS.clear()
    plan = contest.scan(force=True)
    check("폭락 전환: 보유 폭주A 를 지금 팔고 급락주 1개 매수 예정 (코닥H −12%)", plan["mode"] == "crash" and plan["sold_now"] == ["100010"] and [b["ticker"] for b in plan["buy"]] == ["200010"], str(plan["buy"]) + str(plan.get("sold_now")))
    rj = plan["stats"]["rejects"]
    check("하한가 · 혼자 유독 빠짐 제외", rj.get("하한가") == 1 and rj.get("종목악재의심") == 1, str(rj))
    contest.buy_close(force=True)
    pos = contest._positions()
    check("급락주 매수 kind=crash", pos and pos[0]["ticker"] == "200010" and pos[0]["kind"] == "crash")
    HELD["200010"] = 2
    r = contest.evaluate(force=True, send_report=False)
    check("폭락 전환 2일째: 보유 (손절 규칙 안 씀)", not r["pending"])
    HELD["200010"] = 5
    r = contest.evaluate(force=True, send_report=False)
    check("5일째 → 만료 매도 예정", r["pending"] and "폭락 전환" in r["pending"][0]["reason"], str(r["pending"]))
    plan = contest.decide(snaps, contest.get())
    check("폭락 보유 중엔 새 전환 없음", plan["mode"] == "crash_hold")

    print("── 상태 문구 ────────────────────────────")
    txt = contest.status_text()
    check("상태 문구에 전략·계좌·보유", "대회 모드" in txt and "💰" in txt and "코닥H" in txt, txt.replace("\n", " / "))
    check("판정 문구", "판정" in contest.scan_text(plan))
    check("설정 요약", "대회 모드" in config.summary())

    shutil.rmtree(config.DATA_DIR, ignore_errors=True)
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
