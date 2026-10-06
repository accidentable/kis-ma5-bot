"""
backtest/engine.py — 현재 실전 규칙을 일봉으로 재현하는 시뮬레이터

판정은 실전 봇의 함수를 그대로 쓴다 (strategy.prepare / check_breakout / rank, tick 보정).
일봉이라 장중 순서를 알 수 없는 부분은 아래처럼 근사하고, 애매하면 불리한 쪽으로 잡는다.

  진입일 판정   08:50 프리마켓가 ≈ 시가. 시가 > 5일선 기준가, 전일 대비 +5% 이하면 돌파
  진입가        시가 − 0.5ATR 지정가. 저가가 거기 닿으면 그 가격에 체결 (장 초반 눌림)
                안 닿으면 10:00 현재가로 매수 — 알 수 없어 (시가+고가)/2 +2틱으로 잡는다
                (눌림이 안 온 날은 대개 오른 날이라 시가보다 비싸게 산다고 본다)
  진입일        장중 청산 판정 없음 (10시 유예 + 진입 시각 모호). 종가 판정만
  이후          시가가 급이탈선/장중 익절선을 넘어 열리면 시가에 청산
                저가가 급이탈선, 고가가 장중 익절선에 둘 다 닿으면 급이탈을 먼저로 본다
                안 닿으면 종가 판정: +3% 이상 익절 / 5일선 아래 이탈 / 3일차 만료
  재진입        청산한 날은 새로 사지 않는다 (실전은 14:30 전이면 산다 — 거래 수가 조금 적게 나온다)
  비용          매도세 SELL_TAX + 수수료 양쪽 + 지정가 틱 슬리피지 (tick.offset_ticks)
"""
from __future__ import annotations

import copy
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import fmean, median
from typing import Optional

import config
from core import strategy
from core.kis.tick import offset_ticks, round_to_tick

logger = logging.getLogger(__name__)

# ── 비용 가정 ─────────────────────────────────────────────────
SELL_TAX = 0.0020     # 증권거래세+농특세. 2026년 코스피·코스닥 0.20% 로 가정 (0.15% 면 거래당 0.05%p 유리)
FEE = 0.00015         # 증권사 수수료 편도

# ── 민감도 분석 스위치 (기본값 = 실전 규칙) ─────────────────────
ENTRY_MODE = "dip"          # dip: 시가−0.5ATR 눌림 대기 / open: 시가에 바로 매수
PROXY_MODE = "mid"          # 눌림 안 온 날 10시 매수가. mid: (시가+고가)/2 / open: 시가 (낙관적)
ENTRY_DAY_CLOSE_EXIT = True # 진입 당일 종가 판정(5일선 이탈·익절)을 할지
USE_TICK_SLIPPAGE = True    # 지정가 틱 슬리피지 반영


@dataclass
class Trade:
    ticker: str
    name: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    net: float            # 비용 뺀 수익률
    reason: str
    hold: int
    fill: str             # 눌림 / 10시
    score: float
    mtm: list = field(default_factory=list)   # [(date, 진입 대비 평가배수)]


def _buy_cost(p: float) -> float:
    return p * (1 + FEE)


def _sell_net(p: float) -> float:
    return p * (1 - FEE - SELL_TAX)


def precompute_signals(bars: dict[str, list[dict]], names: dict[str, str],
                       test_start: str) -> dict[str, list[tuple[str, int, strategy.Candidate]]]:
    """
    날짜별 돌파 후보. 전일까지의 일봉으로 prepare → 당일 시가로 check_breakout.
    유니버스와 무관하게 한 번만 계산하고, 랭킹은 유니버스별로 따로 한다.
    """
    W = strategy.required_bars() + 5
    hits: dict[str, list] = defaultdict(list)
    for n, (t, b) in enumerate(bars.items(), 1):
        for i in range(W, len(b)):
            d = b[i]["date"]
            if d < test_start:
                continue
            c = strategy.prepare(t, names.get(t, t), b[i - W:i])
            if not c.eligible:
                continue
            if strategy.check_breakout(c, b[i]["open"]):
                hits[d].append((t, i, c))
        if n % 50 == 0:
            logger.info("신호 계산 %d/%d", n, len(bars))
    return hits


def simulate_trade(b: list[dict], i: int, c: strategy.Candidate, name: str) -> Optional[Trade]:
    bar = b[i]
    o, h, l, cl = bar["open"], bar["high"], bar["low"], bar["close"]
    atr = c.atr if c.atr > 0 else o * 0.02

    in_ticks = config.ENTRY_LIMIT_TICKS if USE_TICK_SLIPPAGE else 0
    limit = round_to_tick(o + config.PREOPEN_OFFSET_ATR * atr, "down")
    if ENTRY_MODE == "open":
        fill, how = float(offset_ticks(o, in_ticks)), "시가"
    elif l <= limit:
        fill, how = float(limit), "눌림"
    else:
        proxy = (o + h) / 2 if PROXY_MODE == "mid" else o
        if c.prev_close > 0 and (proxy / c.prev_close - 1) * 100 > config.MAX_CHASE_PCT:
            return None   # 10시 재판정에서 추격 제한에 걸림
        fill, how = float(offset_ticks(proxy, in_ticks)), "10시"

    cost = _buy_cost(fill)
    tp_close = fill * (1 + config.TAKE_PROFIT_PCT / 100)
    tp_intra = tp_close + config.TP_INTRADAY_ATR_BUFFER * atr
    breakout = c.breakout_price
    mtm = []

    def done(j: int, px: float, reason: str, hold: int, ticks: int) -> Trade:
        ex = float(offset_ticks(px, -ticks)) if USE_TICK_SLIPPAGE else float(px)
        mtm.append((b[j]["date"], _sell_net(ex) / cost))
        return Trade(c.ticker, name, bar["date"], b[j]["date"], fill, ex,
                     _sell_net(ex) / cost - 1, reason, hold, how, c.score, mtm)

    tp_ticks, out_ticks = config.TP_EXIT_LIMIT_TICKS, config.EXIT_LIMIT_TICKS

    # 진입일: 종가 판정만
    if ENTRY_DAY_CLOSE_EXIT:
        if cl >= tp_close:
            return done(i, cl, "익절(종가)", 1, tp_ticks)
        if cl < breakout:
            return done(i, cl, "5일선 이탈", 1, out_ticks)
    mtm.append((bar["date"], _sell_net(cl) / cost))

    hold = 1
    for j in range(i + 1, len(b)):
        hold += 1
        o, h, l, cl = b[j]["open"], b[j]["high"], b[j]["low"], b[j]["close"]
        breakout = fmean(x["close"] for x in b[j - 4:j])
        stop_line = breakout - config.INTRADAY_MA5_ATR_BUFFER * atr

        if o <= stop_line:
            return done(j, o, "급이탈(갭)", hold, out_ticks)
        if o >= tp_intra:
            return done(j, o, "익절(갭)", hold, tp_ticks)
        if l <= stop_line:
            return done(j, stop_line, "급이탈", hold, out_ticks)
        if h >= tp_intra:
            return done(j, tp_intra, "익절(장중)", hold, tp_ticks)
        if cl >= tp_close:
            return done(j, cl, "익절(종가)", hold, tp_ticks)
        if cl < breakout:
            return done(j, cl, "5일선 이탈", hold, out_ticks)
        if hold >= config.MAX_HOLD_TRADING_DAYS:
            return done(j, cl, "만료", hold, out_ticks)
        mtm.append((b[j]["date"], _sell_net(cl) / cost))

    j = len(b) - 1
    return done(j, b[j]["close"], "데이터 끝", hold, out_ticks)


def run_universe(label: str, members: set[str], bars: dict[str, list[dict]], names: dict[str, str],
                 hits: dict[str, list], test_start: str) -> dict:
    # 유니버스별 랭킹: 같은 날 이 유니버스 후보끼리만 백분위를 낸다
    ranked: dict[str, list] = {}
    for d, lst in hits.items():
        mine = [(t, i, copy.copy(c)) for t, i, c in lst if t in members]
        if not mine:
            continue
        order = strategy.rank([c for _, _, c in mine])
        pos = {id(c): (t, i) for t, i, c in mine}
        ranked[d] = [(pos[id(c)][0], pos[id(c)][1], c) for c in order]

    dates = sorted({x["date"] for t in members if t in bars for x in bars[t] if x["date"] >= test_start})
    didx = {d: k for k, d in enumerate(dates)}

    # ① 실제 운용 경로: 1슬롯, 청산 다음 날부터 다시 진입
    trades: list[Trade] = []
    curve: list[tuple[str, float]] = []
    equity, k = 1.0, 0
    while k < len(dates):
        d = dates[k]
        tr = None
        for t, i, c in ranked.get(d, []):
            tr = simulate_trade(bars[t], i, c, names.get(t, t))
            if tr:
                break
        if tr is None:
            curve.append((d, equity))
            k += 1
            continue
        for dd, mult in tr.mtm:
            curve.append((dd, equity * mult))
        equity *= 1 + tr.net
        trades.append(tr)
        k = didx.get(tr.exit_date, k) + 1

    # ② 신호 품질: 매일의 1순위를 겹침 허용으로 전부 돌려 표본을 키운다
    signal_nets = []
    for d in dates:
        for t, i, c in ranked.get(d, []):
            tr = simulate_trade(bars[t], i, c, names.get(t, t))
            if tr:
                signal_nets.append(tr.net)
                break

    # ③ 벤치마크: 같은 유니버스 동일가중 보유
    first, last = dates[0], dates[-1]
    rets = []
    for t in members:
        b = bars.get(t)
        if not b:
            continue
        s = next((x for x in b if x["date"] >= first), None)
        e = b[-1]
        if s and s["date"] == first and e["date"] == last:
            rets.append(e["close"] / s["open"] - 1)
    bench = fmean(rets) if rets else float("nan")

    return {"label": label, "members": len(members), "with_data": sum(1 for t in members if t in bars),
            "trades": trades, "curve": curve, "signal_nets": signal_nets,
            "bench": bench, "bench_n": len(rets), "first": first, "last": last,
            "signal_days": len(ranked)}


def metrics(r: dict) -> dict:
    tr: list[Trade] = r["trades"]
    nets = [t.net for t in tr]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    eq = 1.0
    for x in nets:
        eq *= 1 + x

    peak, mdd = 1.0, 0.0
    for _, v in r["curve"]:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)

    d0 = datetime.strptime(r["first"], "%Y%m%d")
    d1 = datetime.strptime(r["last"], "%Y%m%d")
    years = max((d1 - d0).days / 365.25, 1e-9)

    by_year: dict[str, float] = defaultdict(lambda: 1.0)
    for t in tr:
        by_year[t.exit_date[:4]] *= 1 + t.net

    sn = r["signal_nets"]
    return {
        "trades": len(nets),
        "win": len(wins) / len(nets) if nets else 0,
        "avg_win": fmean(wins) if wins else 0,
        "avg_loss": fmean(losses) if losses else 0,
        "expect": fmean(nets) if nets else 0,
        "median": median(nets) if nets else 0,
        "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf"),
        "total": eq - 1,
        "cagr": eq ** (1 / years) - 1 if eq > 0 else -1,
        "mdd": mdd,
        "hold": fmean([t.hold for t in tr]) if tr else 0,
        "reasons": Counter(t.reason for t in tr),
        "fills": Counter(t.fill for t in tr),
        "by_year": dict(sorted(by_year.items())),
        "sig_n": len(sn),
        "sig_expect": fmean(sn) if sn else 0,
        "sig_win": sum(1 for x in sn if x > 0) / len(sn) if sn else 0,
        "bench": r["bench"],
        "bench_cagr": (1 + r["bench"]) ** (1 / years) - 1 if r["bench"] == r["bench"] else float("nan"),
        "years": years,
    }
