"""
tests/test_pipeline.py — 한투 API 없이 도는 통합 점검

실제 주문/시세 대신 가짜 응답을 끼워넣어 스캔 → 진입 → 감시(청산·재진입) → 마감의
흐름과 판정이 의도대로 도는지 확인한다. API 키 없이 실행된다.

    python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# 테스트용 설정 — 실제 .env 값과 무관하게 고정한다
config.DATA_DIR = tempfile.mkdtemp(prefix="ma5bot-test-")
config.STATE_BACKEND = "local"
config.DRY_RUN = True
config.TELEGRAM_BOT_TOKEN = ""
config.TELEGRAM_ALLOWED_CHAT_IDS = []
config.USE_TREND_FILTER = True
config.MIN_TRADING_VALUE = 0
config.MAX_POSITIONS = 1
config.POSITION_PCT = 100.0
config.INTRADAY_REENTRY = True
config.NO_SAME_DAY_REENTRY = True

from core import state, strategy, trader          # noqa: E402
from core.kis import quotes, trading              # noqa: E402
from core.kis.tick import offset_ticks, round_to_tick, tick_size  # noqa: E402
from jobs import close as close_job               # noqa: E402
from jobs import entry as entry_job               # noqa: E402
from jobs import monitor as monitor_job           # noqa: E402
from jobs import prep as prep_job                 # noqa: E402
from jobs import scan as scan_job                 # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


# ══════════════════════════════════════════════════════════════
# 가짜 시세
# ══════════════════════════════════════════════════════════════
N_BARS = 110  # required_bars()(=90) + 여유


def _candles(closes: list[float], half_ranges: list[float] | None = None) -> list[dict]:
    """half_ranges 는 봉별 고저 반폭(비율). 안 주면 전부 1.2%."""
    out = []
    for i, c in enumerate(closes):
        hr = half_ranges[i] if half_ranges else 0.012
        out.append({
            "date": f"2026{(i // 28) + 1:02d}{(i % 28) + 1:02d}",
            "open": c, "high": c * (1 + hr), "low": c * (1 - hr),
            "close": c, "volume": 500_000, "value": 5e10,
        })
    return out


def _uptrend_then_dip(base=50_000, n=N_BARS, dip_days=6, dip_pct=0.06):
    """가파른 상승추세(이평선이 벌어짐) 끝에 며칠 눌린 시계열."""
    closes = [base * (1 + 0.0035 * i) for i in range(n - dip_days)]
    peak = closes[-1]
    for d in range(dip_days):
        closes.append(peak * (1 - dip_pct * (d + 1) / dip_days))
    return closes


def _turnaround(base=50_000, n=N_BARS, down_days=70, dip_days=6):
    """하락하다 돌아서서 최근에 골든크로스가 난 뒤, 마지막 며칠 눌린 시계열."""
    closes = [base * (1 - 0.003 * i) for i in range(down_days)]
    last = closes[-1]
    up_days = n - down_days - dip_days
    closes += [last * (1 + 0.009 * (i + 1)) for i in range(up_days)]
    peak = closes[-1]
    closes += [peak * (1 - 0.05 * (d + 1) / dip_days) for d in range(dip_days)]
    return closes


def _squeeze(base=50_000, n=N_BARS, dip_days=3):
    """완만한 상승(20>60 유지) 속에 이평선이 한 점에 모이고, 마지막 며칠 살짝 눌린 시계열."""
    closes = [base * (1 + 0.0005 * i) for i in range(n - dip_days)]
    last = closes[-1]
    closes += [last * (1 - 0.02 * (d + 1) / dip_days) for d in range(dip_days)]
    return closes


SERIES = {
    "000001": _uptrend_then_dip(),                                   # 깊게 눌림, 이평선 벌어짐
    "000002": _uptrend_then_dip(dip_pct=0.03),                       # 얕게 눌림, 이평선 벌어짐
    "000003": [50_000 * (1 + 0.004 * i) for i in range(N_BARS)],     # 계속 상승 → 5일선 위, 탈락
    "000004": [50_000 * (1 - 0.004 * i) for i in range(N_BARS)],     # 하락추세 → 데드크로스, 탈락
    "000005": _turnaround(),                                         # 최근 골든크로스, 이평선 벌어짐
    "000006": _squeeze(),                                            # 이평선 수렴 + 변동폭 축소
}
# 000006 은 마지막 10봉의 변동폭을 1/3 로 줄여 "조여드는" 구간을 만든다
HALF_RANGES = {"000006": [0.012] * (N_BARS - 10) + [0.004] * 10}
PRICES: dict[str, float] = {}


def fake_get_daily_candles(ticker, days=40):
    return _candles(SERIES[ticker], HALF_RANGES.get(ticker))[-days:]


PREMARKET: dict[str, float] = {}   # market != "J" 로 조회할 때의 가격. 없으면 전일 종가(=5일선 아래)


def fake_get_price(ticker, market="J"):
    if market == "J":
        price = PRICES.get(ticker, SERIES[ticker][-1])
    else:
        price = PREMARKET.get(ticker, SERIES[ticker][-1])
    return {
        "ticker": ticker, "market": market, "name": f"종목{ticker[-1]}",
        "price": price,
        "prev_close": SERIES[ticker][-1],
        "open": 0, "high": 0, "low": 0, "change_pct": 0,
        "volume": 1000, "value": 1e10, "upper_limit": 0, "lower_limit": 0,
        "status_code": "00", "is_halted": False, "is_watch": False,
    }


ORDERS: list[dict] = []
UNAFFORDABLE: set[str] = set()   # 이 집합에 든 종목은 "1주도 못 사는" 상황
PENDING: list[dict] = []         # get_pending_orders 가 돌려줄 미체결 목록
CANCELS: list[dict] = []         # cancel_order 호출 기록
FILLED: dict[str, tuple[int, float]] = {}   # 잔고에 보이는 (수량, 평단). 분할 체결 시뮬레이션


def fake_place_order(ticker, qty, price, side, market=False):
    ORDERS.append({"ticker": ticker, "qty": qty, "price": price, "side": side})
    return {"success": True, "order_no": f"T{len(ORDERS):04d}", "org_no": "00000",
            "message": "TEST", "ticker": ticker, "qty": qty, "price": price,
            "side": side, "dry_run": config.DRY_RUN}


def fake_cancel_order(order_no, org_no, qty, order_dvsn="00"):
    CANCELS.append({"order_no": order_no, "qty": qty})
    PENDING[:] = [p for p in PENDING if p["order_no"] != order_no]
    return {"success": True, "message": "TEST", "order_no": order_no}


def _pending_from(orders, ticker):
    return [{"order_no": o["order_no"], "org_no": o["org_no"], "ticker": ticker, "name": "",
             "side": "buy", "qty": o["qty"], "filled_qty": 0, "remain_qty": o["qty"],
             "price": o["price"], "order_time": "", "order_dvsn": "00"} for o in orders]


def fake_get_buyable(ticker, price):
    budget = 10_000_000
    qty = 0 if ticker in UNAFFORDABLE else budget // max(price, 1)
    return {"cash": budget, "qty_no_margin": qty, "amount_no_margin": budget,
            "qty_max": qty, "amount_max": budget}


def fake_get_balance():
    holdings = []
    for p in state.get_positions():
        t = p["ticker"]
        if t in FILLED:
            qty, avg = FILLED[t]
        else:
            qty, avg = int(p.get("qty", 0) or 0), float(p["entry_price"])
        if qty <= 0:
            continue
        holdings.append({
            "ticker": t, "name": p.get("name", ""), "qty": qty,
            "sellable_qty": qty, "avg_price": avg,
            "price": PRICES.get(t, avg),
            "eval_amount": qty * avg, "pnl_amount": 0, "pnl_pct": 0,
        })
    return {"holdings": holdings, "cash": 10_000_000, "d2_cash": 10_000_000,
            "total_eval": 10_000_000, "net_asset": 10_000_000, "pnl_amount": 0}


UNIVERSE = [{"ticker": t, "name": f"종목{t[-1]}", "marcap": 100_000} for t in SERIES]


def install_fakes():
    from core import universe as universe_mod
    universe_mod.get_universe = lambda force=False: UNIVERSE
    scan_job.universe = universe_mod

    quotes.get_daily_candles = fake_get_daily_candles
    quotes.get_price = fake_get_price
    quotes.is_open_day = lambda d=None: True
    trading.place_order = fake_place_order
    trading.buy = lambda t, q, p, market=False: fake_place_order(t, q, p, "buy", market)
    trading.sell = lambda t, q, p, market=False: fake_place_order(t, q, p, "sell", market)
    trading.get_buyable = fake_get_buyable
    trading.get_balance = fake_get_balance
    trading.get_pending_orders = lambda: list(PENDING)
    trading.cancel_order = fake_cancel_order


def reset_traded_today():
    d = state.load()
    d["traded_today"] = {}
    state.save(d)


def clear_positions(reason="테스트 정리"):
    for p in state.get_positions():
        state.close_position(p["ticker"], p["entry_price"], reason)


def set_all_above_breakout(cands):
    for c in cands:
        PRICES[c.ticker] = c.breakout_price * 1.004


# ══════════════════════════════════════════════════════════════
def main() -> int:
    install_fakes()

    print("\n── 1차 필터 (전일 확정 조건) ─────────────────────")
    eligible, s1 = scan_job.build_candidates(UNIVERSE)
    codes = {c.ticker for c in eligible}
    by = {c.ticker: c for c in eligible}
    check("눌림 종목이 후보에 포함", {"000001", "000002", "000005", "000006"} <= codes,
          f"후보={sorted(codes)}")
    check("5일선 위 종목은 탈락", "000003" not in codes)
    check("데드크로스(20<60) 종목은 탈락", "000004" not in codes)
    check("최근 골든크로스 종목의 크로스 경과일 계산 (참고값)",
          0 <= by["000005"].cross_age < config.CROSS_LOOKBACK,
          f"000005 cross_age={by['000005'].cross_age}")
    check("수렴 종목의 이평선 수렴도가 가장 작다",
          by["000006"].ma_squeeze < min(by[t].ma_squeeze for t in ("000001", "000002", "000005")),
          " / ".join(f"{t}={by[t].ma_squeeze:.2f}%" for t in ("000006", "000001", "000002", "000005")))
    check("수렴 종목의 변동폭이 줄었다 (ATR5/20 < 1)",
          by["000006"].vol_contraction < 0.7 and by["000001"].vol_contraction > 0.9,
          f"000006={by['000006'].vol_contraction:.2f} 000001={by['000001'].vol_contraction:.2f}")

    print("\n── 2차 판정 (돌파) ───────────────────────────────")
    set_all_above_breakout(eligible)
    ranked, s2 = scan_job.find_breakouts(eligible)
    order = [c.ticker for c in ranked]
    check("돌파 종목 4개 검출", len(ranked) == 4, str(order))
    check("수렴+변동폭축소 종목이 1순위", order and order[0] == "000006",
          f"순서={order} | 1순위 상세={ranked[0].score_detail if ranked else ''}")
    check("이평선 벌어진 종목들 중엔 깊이 눌린 순", order.index("000001") < order.index("000002"),
          f"순서={order}")

    print("\n── 돌파 미달 / 추격 방어 ─────────────────────────")
    for c in eligible:
        PRICES[c.ticker] = c.breakout_price * 0.999
    none_ranked, _ = scan_job.find_breakouts(eligible)
    check("5일선 못 넘으면 진입 없음", len(none_ranked) == 0)

    target = eligible[0]
    PRICES[target.ticker] = target.prev_close * (1 + (config.MAX_CHASE_PCT + 2) / 100)
    chase, _ = scan_job.find_breakouts([target])
    check("급등 종목은 추격하지 않음", len(chase) == 0, target.reject)

    print("\n── 09:05 진입 ────────────────────────────────────")
    set_all_above_breakout(eligible)
    result = entry_job.run(force=True)
    positions = state.get_positions()
    check("포지션 1건 생성 (1순위)", len(positions) == 1 and positions[0]["ticker"] == "000006",
          f"{[p['ticker'] for p in positions]}")
    check("전액 매수 주문 발생", len(ORDERS) == 1 and ORDERS[0]["side"] == "buy",
          f"{ORDERS[0] if ORDERS else '없음'}")
    check("출처 기록", positions[0].get("source") == "09:05 진입")
    cached = state.get_daily_candidates()
    check("당일 후보 캐시 저장", cached is not None and len(cached) == 4,
          f"{len(cached) if cached else 0}종목")
    check("당일 매매 종목 기록", "000006" in state.get_traded_today())

    print("\n── 자금 부족 시 다음 순위로 폴백 ────────────────")
    clear_positions()
    reset_traded_today()
    ORDERS.clear()
    UNAFFORDABLE.add("000006")
    set_all_above_breakout(eligible)
    r2 = entry_job.run(force=True)
    held = state.get_positions()
    second = order[1]
    check("1순위를 건너뛰고 2순위를 매수", len(held) == 1 and held[0]["ticker"] == second,
          f"매수={held[0]['ticker'] if held else '없음'} (2순위 {second})")
    check("건너뛴 종목이 기록됨", len(r2.get("skipped", [])) == 1, str(r2.get("skipped")))
    check("매수 주문은 1건만", len(ORDERS) == 1, str(ORDERS))

    clear_positions()
    reset_traded_today()
    ORDERS.clear()
    UNAFFORDABLE.update({"000001", "000002", "000005"})
    r3 = entry_job.run(force=True)
    check("전부 자금 부족이면 매수 없음", len(state.get_positions()) == 0 and not ORDERS)
    check("자금 부족 종목 4건 보고", len(r3.get("skipped", [])) == 4, str(r3.get("skipped")))

    UNAFFORDABLE.clear()
    reset_traded_today()
    ORDERS.clear()
    entry_job.run(force=True)
    pos = state.get_positions()[0]
    entry_price = pos["entry_price"]

    print("\n── 청산 판정 ─────────────────────────────────────")
    atr = float(pos["atr"])
    check("보합에서는 청산 안 함", trader.evaluate_exit(pos, entry_price * 1.01) is None)

    tp_close = trader.take_profit_target(pos)
    tp_intra = trader.intraday_tp_line(pos)
    check("장중 익절선은 종가 익절선 + 버퍼×ATR", abs(tp_intra - (tp_close + config.TP_INTRADAY_ATR_BUFFER * atr)) < 2,
          f"종가 {tp_close:,.0f} / 장중 {tp_intra:,.0f} / ATR {atr:,.0f}")
    # 종가 익절선을 막 넘긴 값: 종가엔 익절, 장중엔 아직 안 팜(여유폭)
    just_over = tp_close + 1
    r = trader.evaluate_exit(pos, just_over, closing=True)
    check(f"종가엔 +{config.TAKE_PROFIT_PCT:g}% 넘기면 익절", r is not None and "익절" in r, str(r))
    r = trader.evaluate_exit(pos, just_over)
    check("장중엔 종가 익절선만 넘겨선 안 판다 (여유폭)", r is None, str(r))
    r = trader.evaluate_exit(pos, tp_intra + 1)
    check("장중에 장중 익절선 넘기면 익절", r is not None and "익절" in r, str(r))

    bo = pos["breakout_price"]
    line = trader.intraday_exit_line(pos)
    check("장중 급이탈선은 5일선 − 1×ATR", abs(line - (bo - atr)) < 1e-6,
          f"5일선 {bo:,.0f} / ATR {atr:,.0f} / 급이탈선 {line:,.0f}")
    r = trader.evaluate_exit(pos, bo * 0.99)
    check("장중에 5일선을 살짝 밑돌아도 청산 안 함 (잡음)", r is None, str(r))
    r = trader.evaluate_exit(pos, bo * 0.99, closing=True)
    check("종가엔 5일선 살짝 밑돌아도 청산", r is not None and "5일선 이탈" in r, str(r))
    r = trader.evaluate_exit(pos, line - 1)
    check("장중에 급이탈선 아래면 청산", r is not None and "급이탈" in r, str(r))

    r = trader.evaluate_exit({**pos, "hold_days": config.MAX_HOLD_TRADING_DAYS},
                             entry_price * 1.01, closing=True)
    check(f"{config.MAX_HOLD_TRADING_DAYS}거래일 만료 발동(마감시)", r is not None and "만료" in r, str(r))
    r = trader.evaluate_exit({**pos, "hold_days": config.MAX_HOLD_TRADING_DAYS},
                             entry_price * 1.01, closing=False)
    check("기간 만료는 장중에 발동 안 함", r is None)
    check("익절 청산은 얕은 지정가", trader._exit_ticks("익절 +3.10%") == config.TP_EXIT_LIMIT_TICKS)
    check("이탈 청산은 깊은 지정가", trader._exit_ticks("5일선 이탈") == config.EXIT_LIMIT_TICKS)

    print("\n── 감시: 청산 → 같은 회차에 재진입 ──────────────")
    monitor_job._now_hhmm = lambda: "1100"
    PRICES[pos["ticker"]] = trader.intraday_exit_line(pos) * 0.995   # 보유 종목 급이탈
    others = [c for c in eligible if c.ticker != pos["ticker"]]
    set_all_above_breakout(others)                                   # 나머지는 돌파 상태
    ORDERS.clear()
    m = monitor_job.run(force=True)
    check("급이탈로 장중 청산 실행", len(m["closed"]) == 1 and "급이탈" in m["closed"][0]["exit_reason"],
          m["closed"][0]["exit_reason"] if m["closed"] else "없음")
    check("매도 주문 전송", any(o["side"] == "sell" for o in ORDERS))
    held = state.get_positions()
    check("같은 회차에 다른 종목으로 재진입", len(held) == 1 and held[0]["ticker"] != pos["ticker"],
          f"매수={held[0]['ticker'] if held else '없음'} (판 종목 {pos['ticker']})")
    check("재진입 출처 기록", held and held[0].get("source") == "장중 재진입")
    check("당일 매매 기록에 둘 다", {pos["ticker"], held[0]["ticker"]} <= state.get_traded_today()
          if held else False, str(sorted(state.get_traded_today())))
    hist = state.get_history()
    last = next((h for h in reversed(hist) if "이탈" in str(h.get("exit_reason", ""))), {})
    check("이력에 5일선 (급)이탈 청산 기록", bool(last),
          f"{last.get('ticker')} {last.get('pnl_pct', 0):+.2f}%" if last else "없음")

    print("\n── 재진입 시간대 ─────────────────────────────────")
    clear_positions()
    remaining = [c for c in eligible if c.ticker not in state.get_traded_today()]
    check("아직 안 산 후보가 남아있음", len(remaining) == 2, str([c.ticker for c in remaining]))
    set_all_above_breakout(remaining)

    monitor_job._now_hhmm = lambda: "1445"
    ORDERS.clear()
    m = monitor_job.run(force=True)
    check("컷오프 이후엔 재진입 없음",
          not state.get_positions() and str(m["reentry"].get("skipped", "")).startswith("컷오프"),
          str(m["reentry"]))

    monitor_job._now_hhmm = lambda: "0900"
    m = monitor_job.run(force=True)
    check("시작 시각 전엔 재진입 없음",
          not state.get_positions() and str(m["reentry"].get("skipped", "")).startswith("시작 전"),
          str(m["reentry"]))

    monitor_job._now_hhmm = lambda: "1100"
    m = monitor_job.run(force=True)
    held = state.get_positions()
    check("시간대 안이면 재진입", len(held) == 1 and held[0]["ticker"] in {c.ticker for c in remaining},
          f"{[p['ticker'] for p in held]}")

    print("\n── 당일 매매 종목 소진 ───────────────────────────")
    clear_positions()
    for c in eligible:
        state.mark_traded_today(c.ticker)
    set_all_above_breakout(eligible)
    m = monitor_job.run(force=True)
    rej = m["reentry"].get("stats", {}).get("rejects", {})
    check("당일 매매한 종목은 다시 안 산다",
          not state.get_positions() and rej.get("보유/당일매매") == 4, str(rej))

    print("\n── 08:40 준비 → 08:50 프리마켓 분할 진입 (분할 3건 메커니즘) ──")
    # 분할 흐름은 잔고 기반 체결 동기화가 필요해서 실주문 모드로 돈다 (주문 자체는 전부 가짜).
    # 기본값은 선주문 1건이지만, 여기선 분할 메커니즘 자체를 검증하려고 3건·10:00 정리로 둔다.
    config.DRY_RUN = False
    config.SPLIT_TRANCHES = 3
    config.SPLIT_CANCEL_AT = "1000"
    config.PREOPEN_CHASE_PCT = 0.0
    config.PREOPEN_OFFSET_ATR = 0.0   # 1차 = 기준가 그대로 (분할 메커니즘만 검증)
    clear_positions()
    reset_traded_today()
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    PREMARKET.clear()
    d = state.load()
    d["daily_candidates"] = {}
    state.save(d)

    pr = prep_job.run(force=True)
    check("08:40 준비가 후보 4종목을 캐시", len(state.get_daily_candidates() or []) == 4, str(pr.get("eligible")))

    top = by["000006"]
    PREMARKET[top.ticker] = top.breakout_price * 1.004          # 프리마켓에서 1순위만 5일선 위
    r = prep_job.premarket_entry(force=True)
    pos = state.get_position(top.ticker)
    buys = [o for o in ORDERS if o["side"] == "buy"]
    check("1순위에 분할매수 3건", pos is not None and len(buys) == config.SPLIT_TRANCHES,
          f"{[(o['qty'], o['price']) for o in buys]}")
    check("가격이 기준가부터 아래로", all(buys[i]["price"] > buys[i + 1]["price"] for i in range(len(buys) - 1)))
    check("1차 = 프리마켓 가격(호가 보정)", buys[0]["price"] == round_to_tick(PREMARKET[top.ticker], "down"),
          f"{buys[0]['price']} vs {PREMARKET[top.ticker]:.0f}")
    gap = buys[0]["price"] - buys[1]["price"]
    check("분할 간격 ≈ 0.5×ATR", abs(gap - config.SPLIT_STEP_ATR * top.atr) <= tick_size(buys[0]["price"]),
          f"간격 {gap} / 0.5ATR {config.SPLIT_STEP_ATR * top.atr:.0f}")
    check("수량 합 = 목표", sum(o["qty"] for o in buys) == pos["target_qty"],
          f"{[o['qty'] for o in buys]} = {pos['target_qty']}")
    check("포지션은 '누적 중', 체결 전 수량 0",
          pos.get("accumulating") is True and pos["qty"] == 0 and pos.get("source") == "08:50 눌림대기")
    check("체결 전엔 당일매매로 안 잡힘", top.ticker not in state.get_traded_today())
    entry_job._now_hhmm = lambda: "0905"   # 정리 시각(10:00) 전 → 단일 진입은 물러서야 한다
    n_before = len(ORDERS)
    r905 = entry_job.run(force=True)
    check("09:05 단일 진입은 분할 진행 중이면 물러섬",
          r905.get("skipped") == "분할진행중" and len(ORDERS) == n_before, str(r905.get("skipped")))

    print("\n── 손절 유예 (10시 전) ───────────────────────────")
    check("한 주도 안 채워졌으면 익절도 안 잡음 (유령 수익 방지)",
          trader.evaluate_exit(pos, pos["entry_price"] * 1.05, stop_blackout=True) is None)
    FILLED[top.ticker] = (pos["target_qty"], float(buys[0]["price"]))   # 전량 체결됐다 치고
    trader.sync_fills()
    pos = state.get_position(top.ticker)
    check("체결 동기화로 수량·평단 반영", pos["qty"] == pos["target_qty"] and pos["filled"] is True,
          f"qty={pos['qty']} avg={pos['entry_price']}")
    line = trader.intraday_exit_line(pos)
    tp_line = trader.intraday_tp_line(pos)
    check("10시 전: 급이탈선 아래여도 안 팜", trader.evaluate_exit(pos, line * 0.99, stop_blackout=True) is None)
    check("10시 전: 익절은 발동 (장중 익절선 넘길 때)",
          "익절" in str(trader.evaluate_exit(pos, tp_line + 1, stop_blackout=True)))
    check("10시부터: 급이탈 발동", "급이탈" in str(trader.evaluate_exit(pos, line * 0.99, stop_blackout=False)))
    monitor_job._now_hhmm = lambda: "0930"
    PRICES[top.ticker] = line * 0.99
    m = monitor_job.run(force=True)
    check("09:30 감시는 급이탈을 무시", m["stop_blackout"] is True and not m["closed"], str(m["closed"]))

    print("\n── 익절 청산 시 남은 분할 주문 먼저 취소 ────────")
    PENDING.extend(_pending_from(pos["entry_orders"][1:], top.ticker))   # 2·3차 아직 대기 중
    monitor_job._now_hhmm = lambda: "0940"
    PRICES[top.ticker] = trader.intraday_tp_line(pos) + 1   # 장중 익절선 위
    ORDERS.clear()
    m = monitor_job.run(force=True)
    check("09:40 익절 청산 (유예 중에도)", len(m["closed"]) == 1 and "익절" in m["closed"][0]["exit_reason"],
          m["closed"][0]["exit_reason"] if m["closed"] else "없음")
    check("매도보다 먼저 미체결 분할 2건 취소",
          len(CANCELS) == 2 and ORDERS and ORDERS[0]["side"] == "sell" and not PENDING,
          f"취소 {len(CANCELS)} / 주문 {[o['side'] for o in ORDERS]}")
    check("청산 후 당일매매 기록", top.ticker in state.get_traded_today())

    print("\n── 10:00 전량 미체결 → 취소 → 같은 틱에 한 번에 매수 ──")
    clear_positions()
    reset_traded_today()
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    r = prep_job.premarket_entry(force=True)
    pos = state.get_position(top.ticker)
    check("다시 분할 걸림", pos is not None and pos.get("accumulating") and pos["qty"] == 0)
    PENDING.extend(_pending_from(pos["entry_orders"], top.ticker))   # 3건 전부 대기 중
    FILLED[top.ticker] = (0, 0.0)                                     # 잔고엔 없음
    set_all_above_breakout(eligible)                                  # 10:00 현재가는 여전히 5일선 위
    monitor_job._now_hhmm = lambda: "1000"
    ORDERS.clear()
    m = monitor_job.run(force=True)
    check("10:00 분할 3건 취소, 결과 '미체결'",
          len(CANCELS) == 3 and m["finalized"] and m["finalized"][0]["result"] == "미체결", str(m["finalized"]))
    check("미체결은 이력에 안 남음", not any("미체결" in str(h.get("exit_reason", "")) for h in state.get_history()))
    held = state.get_positions()
    check("같은 틱 재진입으로 한 번에 매수 (분할 아님)",
          len(held) == 1 and not held[0].get("accumulating") and held[0].get("source") == "장중 재진입",
          f"{[(p['ticker'], p.get('source')) for p in held]}")
    check("미체결이었던 1순위를 다시 살 수 있음", held and held[0]["ticker"] == top.ticker,
          held[0]["ticker"] if held else "없음")
    check("매수 주문 1건", len([o for o in ORDERS if o["side"] == "buy"]) == 1)

    print("\n── 10:00 일부 체결 → 나머지 취소하고 확정 ──────")
    clear_positions()
    reset_traded_today()
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    r = prep_job.premarket_entry(force=True)
    pos = state.get_position(top.ticker)
    first = pos["entry_orders"][0]
    FILLED[top.ticker] = (first["qty"], float(first["price"]))          # 1차만 체결
    PENDING.extend(_pending_from(pos["entry_orders"][1:], top.ticker))  # 2·3차 대기
    monitor_job._now_hhmm = lambda: "1000"
    PRICES[top.ticker] = float(first["price"])
    m = monitor_job.run(force=True)
    pos = state.get_position(top.ticker)
    check("일부 체결이면 확정: 나머지 2건 취소",
          len(CANCELS) == 2 and m["finalized"] and m["finalized"][0]["result"] == "확정", str(m["finalized"]))
    check("수량은 체결분, 누적 상태 해제",
          pos is not None and pos["qty"] == first["qty"] and not pos.get("accumulating"),
          f"qty={pos['qty'] if pos else None}")
    check("확정 후 당일매매 기록", top.ticker in state.get_traded_today())
    check("확정 후엔 재진입 안 함 (슬롯 점유)", m["reentry"].get("skipped") == "보유한도", str(m["reentry"]))

    print("\n── 기본값: 눌림 대기 1건 (−0.5ATR) → 10:00 안 빠지면 취소 후 현재가 매수 ──")
    config.SPLIT_TRANCHES = 1
    config.SPLIT_CANCEL_AT = "1000"
    config.PREOPEN_CHASE_PCT = 0.0
    config.PREOPEN_OFFSET_ATR = -0.5
    clear_positions()
    reset_traded_today()
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    r = prep_job.premarket_entry(force=True)
    pos = state.get_position(top.ticker)
    buys = [o for o in ORDERS if o["side"] == "buy"]
    check("눌림 대기 주문 1건", pos is not None and len(buys) == 1 and len(pos["entry_orders"]) == 1, str(buys))
    expect = round_to_tick(PREMARKET[top.ticker] - 0.5 * top.atr, "down")
    check("지정가 = 프리마켓가 − 0.5ATR (호가 보정)", buys[0]["price"] == expect,
          f"{buys[0]['price']} vs {expect} (프리마켓 {PREMARKET[top.ticker]:.0f}, ATR {top.atr:.0f})")
    check("지정가가 급이탈선(5일선−1ATR)보다 위", buys[0]["price"] > trader.intraday_exit_line(pos),
          f"{buys[0]['price']} > {trader.intraday_exit_line(pos):.0f}")
    check("수량 전량", buys[0]["qty"] == pos["target_qty"])
    check("출처 라벨", pos.get("source") == "08:50 눌림대기", str(pos.get("source")))

    entry_job._now_hhmm = lambda: "0905"
    n_before = len(ORDERS)
    r = entry_job.run(force=True)
    check("09:05: 대기 주문이 살아있으면 물러섬", r.get("skipped") == "분할진행중" and len(ORDERS) == n_before,
          str(r.get("skipped")))

    # 10:00 — 눌림이 안 왔다 → 취소 → 같은 틱에 현재가로 한 번에 매수
    PENDING.extend(_pending_from(pos["entry_orders"], top.ticker))
    FILLED[top.ticker] = (0, 0.0)
    set_all_above_breakout(eligible)
    monitor_job._now_hhmm = lambda: "1000"
    ORDERS.clear()
    m = monitor_job.run(force=True)
    held = state.get_positions()
    check("10:00 대기 주문 취소", len(CANCELS) == 1 and not PENDING, f"취소 {len(CANCELS)}")
    check("같은 틱에 현재가로 한 번에 매수",
          len(held) == 1 and not held[0].get("accumulating") and held[0].get("source") == "장중 재진입",
          f"{[(p['ticker'], p.get('source')) for p in held]}")
    check("10:00 이후는 현재가 +2틱 즉시 매수 (대기 아님)",
          ORDERS and ORDERS[-1]["side"] == "buy"
          and ORDERS[-1]["price"] == offset_ticks(PRICES[held[0]["ticker"]], config.ENTRY_LIMIT_TICKS),
          f"{ORDERS[-1] if ORDERS else '없음'}")

    # 눌림이 와서 체결된 경우: 10:00 에 확정
    clear_positions()
    reset_traded_today()
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    r = prep_job.premarket_entry(force=True)
    pos = state.get_position(top.ticker)
    FILLED[top.ticker] = (pos["target_qty"], float(pos["entry_orders"][0]["price"]))
    PRICES[top.ticker] = float(pos["entry_orders"][0]["price"])
    monitor_job._now_hhmm = lambda: "1000"
    m = monitor_job.run(force=True)
    pos = state.get_position(top.ticker)
    check("눌림 체결이면 10:00 확정",
          pos is not None and not pos.get("accumulating") and pos["qty"] == pos["target_qty"]
          and m["finalized"] and m["finalized"][0]["result"] == "확정",
          f"qty={pos['qty'] if pos else None} {m['finalized']}")
    check("확정 후 당일매매 기록", top.ticker in state.get_traded_today())

    print("\n── 08:50 에 못 걸었을 때: 10:00 전 진입은 눌림 대기, 이후는 즉시 매수 ──")
    clear_positions()
    reset_traded_today()
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    PREMARKET.clear()                      # 프리마켓에서 5일선 위 종목 없음
    r = prep_job.premarket_entry(force=True)
    check("프리마켓 돌파 없음 → 대기 주문 없음", not state.get_positions() and not r.get("entered"))
    set_all_above_breakout(eligible)
    entry_job._now_hhmm = lambda: "0905"
    ORDERS.clear()
    r = entry_job.run(force=True)
    held = state.get_positions()
    p = held[0] if held else None
    check("09:05 대체 진입도 눌림 대기 주문",
          p is not None and p.get("accumulating") and p.get("source") == "09:05 눌림대기",
          f"{(p['ticker'], p.get('source')) if p else '없음'}")
    if p:
        expect = round_to_tick(PRICES[p["ticker"]] - 0.5 * by[p["ticker"]].atr, "down")
        check("지정가 = 09:05 현재가 − 0.5ATR", ORDERS[-1]["price"] == expect,
              f"{ORDERS[-1]['price']} vs {expect}")

    clear_positions()
    reset_traded_today()
    ORDERS.clear()
    entry_job._now_hhmm = lambda: "1005"
    r = entry_job.run(force=True)
    held = state.get_positions()
    p = held[0] if held else None
    check("10:00 이후 진입은 현재가 즉시 매수",
          p is not None and not p.get("accumulating") and p.get("source") == "09:05 진입"
          and ORDERS[-1]["price"] == offset_ticks(PRICES[p["ticker"]], config.ENTRY_LIMIT_TICKS),
          f"{(p['ticker'], p.get('source'), ORDERS[-1]['price']) if p and ORDERS else '없음'}")

    config.DRY_RUN = True
    for x in (ORDERS, PENDING, CANCELS):
        x.clear()
    FILLED.clear()
    PREMARKET.clear()

    print("\n── 보유일차 증가 ─────────────────────────────────")
    clear_positions()
    reset_traded_today()
    ORDERS.clear()
    entry_job.run(force=True)
    p0 = state.get_positions()[0]
    check("진입일은 1일차", p0["hold_days"] == 1, str(p0["hold_days"]))
    state.update_position(p0["ticker"], entry_date="2020-01-01")
    trader.refresh_breakout_prices()
    p1 = state.get_positions()[0]
    check("이튿날 아침에 2일차", p1["hold_days"] == 2, str(p1["hold_days"]))

    print("\n── 마감 정리 ─────────────────────────────────────")
    state.update_position(p1["ticker"], hold_days=config.MAX_HOLD_TRADING_DAYS)
    PRICES[p1["ticker"]] = p1["entry_price"] * 1.02   # 5일선 위, 익절 미달
    c = close_job.run(force=True, send_report=False)
    check("기간 만료로 마감 청산", len(c["closed"]) == 1 and "만료" in c["closed"][0]["exit_reason"],
          c["closed"][0]["exit_reason"] if c["closed"] else "없음")

    print("\n" + "═" * 60)
    print(f"통과 {len(PASS)} / 실패 {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print("  실패:", f)
    shutil.rmtree(config.DATA_DIR, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
