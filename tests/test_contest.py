"""
tests/test_contest.py — 대회 모드 (STRATEGY=contest) 흐름 점검. API 키 없이 돈다.

    python tests/test_contest.py

가짜 시세·잔고·주문으로 준비 → 판정 → 종가 매수 → 마감 판정 → 시가 매도 → 목표 락 → 폭락 전환 → 월초 리셋을 돈다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
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
_orig_send = notify.send
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

    print("── 소액 계좌: 1위가 비싸면 다음 순위 ──────")
    for p in contest._positions():
        state.close_position(p["ticker"], PRICE[p["ticker"]], "테스트 정리")
    for t in PRICE:
        CHG[t] = 0.5; PRICE[t] = float(INFO[t][5])
    contest.put(locked=False, paper_cash=13_500.0)   # 예산 13,432원: 폭주A 14,550 은 1주도 못 삼, 강세B 13,050 은 1주
    plan = contest.decide(contest.snapshots(contest.get())[0], contest.get())
    check("다음 순위 후보를 판정에 같이 저장", [b["ticker"] for b in plan.get("backup", [])][:1] == ["100020"], str(plan.get("backup")))
    contest.put(plan=plan)
    contest.buy_close(force=True)
    pos = contest._positions()
    check("폭주A 건너뛰고 강세B 1주 매수", [(p["ticker"], p["qty"]) for p in pos] == [("100020", 1)], str([(p["ticker"], p["qty"]) for p in pos]))
    check("알림에 건너뛴 이유", "폭주A 1주 14,550원 > 예산 → 강세B" in SENT[-1], SENT[-1].replace("\n", " / "))
    contest.put(paper_cash=100_000_000.0)

    print("── 대회 기간 (CONTEST_START) ──────────")
    config.CONTEST_START = "2000-01-03"
    check("시작일이 지났으면 기간 키 = 시작일 (달이 바뀌어도 리셋 안 함)", contest._period() == "2000-01-03")
    contest.put(month="2000-01-03", anchor=1.0)
    contest.prep(force=True)
    check("같은 기간이면 기준 순자산 유지", contest.get()["anchor"] == 1.0, str(contest.get()["anchor"]))
    config.CONTEST_START = "2999-01-01"
    check("시작 전이면 달 단위", contest._period() == date.today().strftime("%Y-%m"))
    contest.prep(force=True)
    check("기간이 바뀌면 기준 다시 잡음", contest.get()["month"] == date.today().strftime("%Y-%m") and contest.get()["anchor"] > 1)
    config.CONTEST_START = ""

    print("── /target 목표 바꾸기 ──────────────────")
    contest.put(lock_pct=None, locked=False, pending_exit=[], anchor=100_000_000.0, paper_cash=100_000_000.0)
    check("기본 목표 = CT_LOCK_PCT", contest.lock_pct() == config.CT_LOCK_PCT)
    check("/target 만 치면 지금 목표 보여줌", "🎯 목표 +30%" in contest.set_target(None))
    contest.put(paper_cash=135_000_000.0)                  # 종이 순자산 +35%
    contest.set_target("40")
    r = contest.evaluate(force=True, send_report=False)
    check("목표 +40% 면 +35% 에선 락 안 걸림", not r["locked"] and contest.lock_pct() == 40.0, str(r["locked"]))
    contest.set_target("30")
    r = contest.evaluate(force=True, send_report=False)
    check("목표 +30% 로 낮추면 락", r["locked"])
    msg = contest.set_target("50")
    ct = contest.get()
    check("락 중에 목표를 지금 수익(+35%)보다 올리면 락 해제 · 락 매도 예약 취소",
          not ct["locked"] and not any("락" in x["reason"] for x in ct.get("pending_exit") or []) and "락 해제" in msg, msg.replace("\n", " / "))
    contest.set_target("30"); contest.evaluate(force=True, send_report=False)
    msg = contest.set_target("20")
    check("새 목표도 이미 넘었으면 락 유지", contest.get()["locked"] and "락 유지" in msg, msg)
    msg = contest.set_target("off")
    check("off → 락 없음, 해제", contest.lock_pct() == 0 and not contest.get()["locked"] and not core_ct.lock_hit(1e12, 1.0, 0), msg)
    check("숫자 아니면 안내", "숫자로" in contest.set_target("abc") and contest.lock_pct() == 0)
    from core import commands
    config.TELEGRAM_ALLOWED_CHAT_IDS = [1]
    check("텔레그램 /target 70", "→ +70%" in commands.handle("/target 70", 1) and contest.lock_pct() == 70.0)
    config.TELEGRAM_ALLOWED_CHAT_IDS = []
    contest.put(lock_pct=None, locked=False, pending_exit=[], paper_cash=100_000_000.0)

    print("── /deposit 입출금 ──────────────────────")
    contest.put(lock_pct=None, locked=False, pending_exit=[], anchor=100_000_000.0, paper_cash=100_000_000.0, deposits=[])
    check("금액 읽기: 20만 · 200000 · -10만 · 1억5천만",
          [contest._parse_amount(x) for x in ("20만", "200,000", "-10만", "1억5천만")] == [200_000, 200_000, -100_000, 150_000_000])
    msg = contest.deposit("1억")
    ct = contest.get()
    check("1억 입금 → 기준 2억, 종이 현금도 2억", ct["anchor"] == 200_000_000 and ct["paper_cash"] == 200_000_000 and "입금" in msg, msg.replace("\n", " / "))
    r = contest.evaluate(force=True, send_report=False)
    check("입금을 반영하면 +100% 로 안 보고 락 안 걸림", not r["locked"])
    contest.put(anchor=100_000_000.0)                       # 반영을 깜빡해 락이 걸린 상황
    r = contest.evaluate(force=True, send_report=False)
    check("(반영 안 하면 입금이 +100% 로 보여 락)", r["locked"])
    contest.put(paper_cash=100_000_000.0)
    msg = contest.deposit("1억")
    check("뒤늦게 반영해도 락 해제", not contest.get()["locked"] and "해제" in msg, msg.replace("\n", " / "))
    msg = contest.deposit("-5천만")
    check("출금 → 기준에서 뺌", contest.get()["anchor"] == 150_000_000 and "출금" in msg)
    check("기준보다 많이 빼면 거절", "많이 뺄" in contest.deposit("-10억"))
    check("못 읽는 금액 안내", "못 읽었다" in contest.deposit("많이"))
    check("기록 보기", "입출금 기록" in contest.deposit(None) and len(contest.get()["deposits"]) == 3)
    config.TELEGRAM_ALLOWED_CHAT_IDS = [1]
    check("텔레그램 /deposit 20 만 (띄어 써도)", "200,000원 반영" in commands.handle("/deposit 20 만", 1))
    config.TELEGRAM_ALLOWED_CHAT_IDS = []
    contest.put(lock_pct=None, locked=False, pending_exit=[], anchor=100_000_000.0, paper_cash=100_000_000.0)

    print("── 손매매 반영 /sync · /adopt ───────────")
    for p in contest._positions():
        state.close_position(p["ticker"], PRICE[p["ticker"]], "테스트 정리")
    HOLD = {}                                              # 실계좌 보유 (ticker → (qty, 평단))
    real_bal = lambda: {"holdings": [{"ticker": t, "name": INFO[t][1], "qty": q, "sellable_qty": q, "avg_price": a, "price": PRICE[t],
                                      "eval_amount": 0, "pnl_amount": 0, "pnl_pct": 0} for t, (q, a) in HOLD.items()],
                        "cash": 500_000.0, "d2_cash": 0, "total_eval": 0, "net_asset": 900_000.0, "pnl_amount": 0}
    saved = (trading.get_balance, getattr(trading, "get_today_fills", None))
    trading.get_balance = real_bal
    trading.get_today_fills = lambda target=None: [{"ticker": "100010", "side": "sell", "qty": 6, "avg_price": 15_000.0}]
    config.DRY_RUN = False
    base = {"strategy": "contest", "kind": "momentum", "peak_close": 14_000, "hold_days": 1, "filled": True}
    state.add_position({**base, "ticker": "100010", "name": "폭주A", "qty": 6, "entry_price": 14_000, "entry_date": "2000-01-01"})
    state.add_position({**base, "ticker": "100020", "name": "강세B", "qty": 10, "entry_price": 13_000, "entry_date": "2000-01-01"})
    state.add_position({**base, "ticker": "100030", "name": "상승C", "qty": 3, "entry_price": 12_000, "entry_date": date.today().isoformat(), "filled": False})
    contest.put(pending_exit=[{"ticker": "100010", "reason": "손절"}])
    HOLD.update({"100020": (7, 13_000.0), "200010": (9, 11_000.0)})   # 폭주A 는 앱에서 팔고, 강세B 3주 팔고, 코닥H 9주 삼
    r = contest.reconcile(include_today=False, announce=False)
    pos = {p["ticker"]: p for p in contest._positions()}
    h = {x["ticker"]: x for x in state.get_history(5)}
    check("계좌에 없는 폭주A → 수동 매도로 이력 (오늘 매도 체결가 15,000)", "100010" not in pos and h.get("100010", {}).get("exit_price") == 15_000.0 and "수동 매도" in h["100010"]["exit_reason"], str(h.get("100010")))
    check("수량 다른 강세B → 계좌 수량 7주", pos["100020"]["qty"] == 7)
    check("장중 /sync 는 오늘 미체결 매수(상승C)를 안 건드림", "100030" in pos)
    check("봇이 모르는 코닥H 는 수동 보유로만", [m["ticker"] for m in r["manual"]] == ["200010"] and "200010" not in pos)
    check("정리한 종목의 시가 매도 예약도 지움", not contest.get().get("pending_exit"))
    txt = contest.sync_text(r)
    check("/sync 문구: 정리 · 수량 · 수동 보유 · /adopt 안내", all(k in txt for k in ("폭주A", "7주", "코닥H", "/adopt")), txt.replace("\n", " / "))
    n_hist = len(state.get_history(200))
    r = contest.reconcile(include_today=True, announce=False)
    check("마감 뒤엔 오늘 미체결 매수를 지우되 이력엔 안 남김", "100030" not in {p["ticker"] for p in contest._positions()}
          and [x["ticker"] for x in r["unfilled"]] == ["100030"] and len(state.get_history(200)) == n_hist,
          f"{r['unfilled']} hist {n_hist}->{len(state.get_history(200))}")
    msg = contest.adopt("200010")
    pos = {p["ticker"]: p for p in contest._positions()}
    check("/adopt 코닥H → 봇 포지션 (평단 11,000 · 오늘부터)", pos.get("200010", {}).get("entry_price") == 11_000 and pos["200010"]["entry_date"] == date.today().isoformat() and "넘김" in msg, msg)
    check("슬롯 넘으면 경고", "슬롯" in msg)
    check("이미 관리 중 / 계좌에 없음 안내", "이미" in contest.adopt("200010") and "없다" in contest.adopt("100060"))
    config.TELEGRAM_ALLOWED_CHAT_IDS = [1]
    check("텔레그램 /sync", "계좌 맞춤" in commands.handle("/sync", 1))
    config.TELEGRAM_ALLOWED_CHAT_IDS = []
    trading.get_balance, trading.get_today_fills = saved
    config.DRY_RUN = True
    check("DRY_RUN 이면 맞추지 않음", contest.reconcile().get("skipped") == "DRY_RUN")
    for p in contest._positions():
        state.close_position(p["ticker"], PRICE[p["ticker"]], "테스트 정리")

    print("── 알림 머리말 · 설정 파일 ─────────────")
    got, orig_call = [], notify._call
    notify._call = lambda method, payload: got.append(payload["text"])
    config.TELEGRAM_ALLOWED_CHAT_IDS = [1]
    config.BOT_LABEL = "모의"; _orig_send("안녕")
    config.BOT_LABEL = ""; _orig_send("안녕")
    check("BOT_LABEL 이 있으면 [모의] 머리말, 비우면 그대로", got == ["[모의] 안녕", "안녕"], str(got))
    notify._call, config.TELEGRAM_ALLOWED_CHAT_IDS = orig_call, []
    env_file = os.path.join(config.DATA_DIR, "env.test")
    with open(env_file, "w", encoding="utf-8") as f:
        f.write("KIS_ENV=mock\nDATA_DIR=data-mock\n")
    env = {k: v for k, v in os.environ.items() if k not in ("KIS_ENV", "DATA_DIR", "BOT_LABEL")}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", "import config; print(config.KIS_ENV, config.DATA_DIR, config.BOT_LABEL)"],
                         cwd=root, env={**env, "ENV_FILE": env_file, "PYTHONIOENCODING": "utf-8"},
                         capture_output=True, text=True, encoding="utf-8").stdout.strip()
    check("ENV_FILE 로 설정 파일 선택 (모의 · data-mock · 머리말 모의)", out == "mock data-mock 모의", out)

    shutil.rmtree(config.DATA_DIR, ignore_errors=True)
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
