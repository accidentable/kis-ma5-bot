"""
backtest/run.py — 유니버스 비교 백테스트

  python -m backtest.run fetch     유니버스 구성 + 3년치 일봉 수집 (한투 API, 캐시)
  python -m backtest.run sim       캐시로 시뮬레이션 + 리포트 (API 호출 없음)
  python -m backtest.run all       둘 다
  python -m backtest.run sens      가정을 바꿔가며 결론이 버티는지 (민감도)
  python -m backtest.run snapshot  캐시를 backtest/snapshot.json.gz 한 파일로 묶는다 (git 에 올리는 용도)

캐시(data/)가 없고 스냅샷이 있으면 sim 이 스냅샷에서 캐시를 먼저 풀어낸다.
그래서 한투 키가 없는 환경(클라우드 세션 등)에서도 sim 은 돈다.

비교 대상: KOSPI100 / KOSPI200 / KOSPI200+KOSDAQ150. 규칙은 지금 실전 설정 그대로.
실전 봇과 같은 앱키의 유량을 나눠 쓰므로 장이 열려 있을 때는 돌리지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

# 골든크로스 경과일은 랭킹에 안 쓰는 참고값이라 탐색을 최소로 줄여 계산을 빠르게 한다.
config.CROSS_LOOKBACK = 1

from backtest import data, engine, universes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "bt_cache")
UNIV_CACHE = os.path.join(ROOT, "data", "bt_universes.json")
OUT = os.path.join(ROOT, "backtest", "results")
SNAPSHOT = os.path.join(ROOT, "backtest", "snapshot.json.gz")


def make_snapshot() -> None:
    """캐시 전체(유니버스 + 종목별 일봉)를 gzip JSON 한 파일로."""
    import gzip
    import json
    with open(UNIV_CACHE, encoding="utf-8") as f:
        univ = json.load(f)
    bars = {}
    for fn in sorted(os.listdir(CACHE)):
        if fn.endswith(".json"):
            with open(os.path.join(CACHE, fn), encoding="utf-8") as f:
                bars[fn[:-5]] = json.load(f)
    with gzip.open(SNAPSHOT, "wt", encoding="utf-8") as f:
        json.dump({"universes": univ, "bars": bars}, f, separators=(",", ":"))
    logger.info("스냅샷 저장: %s (%d종목, %.1f MB)", SNAPSHOT, len(bars), os.path.getsize(SNAPSHOT) / 1e6)


def restore_snapshot_if_needed() -> None:
    """캐시가 없고 스냅샷이 있으면 풀어낸다."""
    import gzip
    import json
    if os.path.exists(UNIV_CACHE) and os.path.isdir(CACHE) and os.listdir(CACHE):
        return
    if not os.path.exists(SNAPSHOT):
        return
    with gzip.open(SNAPSHOT, "rt", encoding="utf-8") as f:
        snap = json.load(f)
    os.makedirs(CACHE, exist_ok=True)
    with open(UNIV_CACHE, "w", encoding="utf-8") as f:
        json.dump(snap["universes"], f, ensure_ascii=False)
    for t, b in snap["bars"].items():
        with open(os.path.join(CACHE, f"{t}.json"), "w", encoding="utf-8") as f:
            json.dump(b, f)
    logger.info("스냅샷에서 캐시 복원: %d종목", len(snap["bars"]))

YEARS = 3
logger = logging.getLogger("backtest")


def _pct(x: float, digits: int = 2) -> str:
    return "-" if x != x else f"{x * 100:+.{digits}f}%"


def fetch(refresh: bool) -> None:
    u = universes.build(UNIV_CACHE)
    tickers = sorted({s["ticker"] for k in u for s in u[k]})
    until = date.today()
    since = until - timedelta(days=int(365.25 * YEARS) + 160)
    logger.info("수집 대상 %d종목, %s ~ %s", len(tickers), since, until)
    data.load_all(tickers, CACHE, since, until, refresh=refresh)


def sim() -> None:
    restore_snapshot_if_needed()
    u = universes.build(UNIV_CACHE)
    names = {s["ticker"]: s["name"] for k in u for s in u[k]}
    tickers = sorted(names)
    bars = data.load_all(tickers, CACHE, date(2000, 1, 1), date.today())  # 캐시만 읽는다
    missing = [t for t in tickers if t not in bars]
    if missing:
        logger.warning("캐시 없는 종목 %d개 — fetch 를 먼저 돌려라: %s", len(missing), missing[:10])

    last = max(b[-1]["date"] for b in bars.values() if b)
    y, m, d = int(last[:4]), int(last[4:6]), int(last[6:])
    test_start = f"{y - YEARS}{m:02d}{d:02d}"
    logger.info("시험 구간 %s ~ %s", test_start, last)

    hits = engine.precompute_signals(bars, names, test_start)

    sets = {
        "KOSPI100": {s["ticker"] for s in u["KOSPI100"]},
        "KOSPI200": {s["ticker"] for s in u["KOSPI200"]},
        "KOSPI200+KOSDAQ150": {s["ticker"] for s in u["KOSPI200"]} | {s["ticker"] for s in u["KOSDAQ150"]},
    }
    results = []
    for label, mem in sets.items():
        r = engine.run_universe(label, mem, bars, names, hits, test_start)
        results.append((r, engine.metrics(r)))
        logger.info("%s 완료: 거래 %d건", label, len(r["trades"]))

    os.makedirs(OUT, exist_ok=True)
    tag = last
    lines = [
        f"# 유니버스 비교 백테스트 ({results[0][0]['first']} ~ {last})",
        "",
        f"규칙: 실전 설정 그대로 (익절 +{config.TAKE_PROFIT_PCT:g}%·장중 +{config.TP_INTRADAY_ATR_BUFFER:g}ATR, "
        f"5일선 이탈·장중 −{config.INTRADAY_MA5_ATR_BUFFER:g}ATR, {config.MAX_HOLD_TRADING_DAYS}거래일, "
        f"진입 눌림 {config.PREOPEN_OFFSET_ATR:+g}ATR). 1종목 100%, 비용 매도세 {engine.SELL_TAX * 100:.2f}% + 수수료 양쪽 + 틱 슬리피지.",
        "",
        "| 항목 | " + " | ".join(r["label"] for r, _ in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    rows = [
        ("종목 수 (데이터 있음)", lambda r, m: f"{r['members']} ({r['with_data']})"),
        ("신호 있는 날", lambda r, m: f"{r['signal_days']}"),
        ("거래 수", lambda r, m: f"{m['trades']}"),
        ("승률", lambda r, m: f"{m['win'] * 100:.1f}%"),
        ("평균 이익 / 평균 손실", lambda r, m: f"{_pct(m['avg_win'])} / {_pct(m['avg_loss'])}"),
        ("**거래당 기대값 (비용 후)**", lambda r, m: f"**{_pct(m['expect'], 3)}**"),
        ("손익비 (PF)", lambda r, m: f"{m['pf']:.2f}"),
        ("누적 수익", lambda r, m: _pct(m["total"], 1)),
        ("연환산 (CAGR)", lambda r, m: _pct(m["cagr"], 1)),
        ("최대 낙폭 (MDD)", lambda r, m: _pct(m["mdd"], 1)),
        ("평균 보유일", lambda r, m: f"{m['hold']:.1f}"),
        ("벤치마크 (동일가중 보유)", lambda r, m: f"{_pct(m['bench'], 1)} (연 {_pct(m['bench_cagr'], 1)})"),
        ("신호 품질: 1순위 전부 (겹침 허용)", lambda r, m: f"{m['sig_n']}건, 기대값 {_pct(m['sig_expect'], 3)}, 승률 {m['sig_win'] * 100:.1f}%"),
    ]
    for title, fn in rows:
        lines.append(f"| {title} | " + " | ".join(fn(r, m) for r, m in results) + " |")

    lines += ["", "## 연도별 누적 (운용 경로)", "",
              "| 연도 | " + " | ".join(r["label"] for r, _ in results) + " |",
              "|---|" + "---|" * len(results)]
    years = sorted({y for _, m in results for y in m["by_year"]})
    for yr in years:
        lines.append(f"| {yr} | " + " | ".join(_pct(m["by_year"].get(yr, 1.0) - 1, 1) for _, m in results) + " |")

    lines += ["", "## 청산 사유 / 진입 방식", ""]
    for r, m in results:
        reasons = ", ".join(f"{k} {v}" for k, v in m["reasons"].most_common())
        fills = ", ".join(f"{k} {v}" for k, v in m["fills"].most_common())
        lines.append(f"- **{r['label']}** — 청산: {reasons} / 진입: {fills}")

    lines += ["", "## 근사와 한계", "",
              "- 일봉 재현이다. 08:50 프리마켓가는 시가로, 10시 매수가는 (시가+고가)/2 +2틱으로 잡았다.",
              "- 같은 날 급이탈선과 장중 익절선에 둘 다 닿으면 급이탈을 먼저로 본다 (보수적).",
              "- 청산한 날은 재진입하지 않는다. 실전은 14:30 전이면 다시 산다.",
              "- 지금의 지수 구성종목으로 과거를 돌렸다 (생존편향). 유니버스 간 상대 비교용이다.",
              "- 1슬롯 운용 경로는 어느 날 무엇을 잡았느냐에 크게 흔들린다. '신호 품질' 행이 표본이 더 크다."]

    report = os.path.join(OUT, f"universe_compare_{tag}.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for r, _ in results:
        p = os.path.join(OUT, f"trades_{r['label'].replace('+', '_')}_{tag}.csv")
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["종목", "이름", "진입일", "청산일", "진입가", "청산가", "수익률", "사유", "보유일", "진입방식", "점수"])
            for t in r["trades"]:
                w.writerow([t.ticker, t.name, t.entry_date, t.exit_date, f"{t.entry:.0f}", f"{t.exit:.0f}",
                            f"{t.net * 100:.2f}", t.reason, t.hold, t.fill, f"{t.score:.3f}"])

    print("\n".join(lines))
    print(f"\n리포트: {report}")


def sens(label: str = "KOSPI100") -> None:
    """가정을 하나씩 바꿔 결론이 버티는지 본다. 1순위 신호(겹침 허용) 기대값 기준."""
    restore_snapshot_if_needed()
    u = universes.build(UNIV_CACHE)
    names = {s["ticker"]: s["name"] for k in u for s in u[k]}
    bars = data.load_all(sorted(names), CACHE, date(2000, 1, 1), date.today())
    last = max(b[-1]["date"] for b in bars.values() if b)
    test_start = f"{int(last[:4]) - YEARS}{last[4:]}"
    hits = engine.precompute_signals(bars, names, test_start)
    members = {s["ticker"] for s in u[label]}

    variants = [
        ("① 기본 (실전 규칙)", {}),
        ("② 10시 매수가를 시가로 (낙관)", {"PROXY_MODE": "open"}),
        ("③ 비용·슬리피지 0 (세전 원가)", {"SELL_TAX": 0.0, "FEE": 0.0, "USE_TICK_SLIPPAGE": False}),
        ("④ ②+③ 가장 낙관적", {"PROXY_MODE": "open", "SELL_TAX": 0.0, "FEE": 0.0, "USE_TICK_SLIPPAGE": False}),
        ("⑤ 눌림 대기 없이 시가 매수", {"ENTRY_MODE": "open"}),
        ("⑥ 진입일 종가 청산 안 함", {"ENTRY_DAY_CLOSE_EXIT": False}),
        ("⑦ 장중 여유폭 0 (옛 규칙)", {"INTRADAY_MA5_ATR_BUFFER": 0.0, "TP_INTRADAY_ATR_BUFFER": 0.0}),
    ]
    lines = [f"# 민감도 분석 — {label} ({test_start} ~ {last})", "",
             "기준은 매일 1순위 신호를 겹침 허용으로 전부 돌린 거래당 기대값(비용 후). 표본이 운용 경로보다 크다.", "",
             "| 가정 | 신호 수 | 거래당 기대값 | 승률 | 운용 경로 누적 |", "|---|---|---|---|---|"]
    for title, kw in variants:
        saved = {}
        for k, v in kw.items():
            target = engine if hasattr(engine, k) else config
            saved[(target, k)] = getattr(target, k)
            setattr(target, k, v)
        r = engine.run_universe(label, members, bars, names, hits, test_start)
        m = engine.metrics(r)
        for (target, k), v in saved.items():
            setattr(target, k, v)
        lines.append(f"| {title} | {m['sig_n']} | {_pct(m['sig_expect'], 3)} | {m['sig_win'] * 100:.1f}% | {_pct(m['total'], 1)} |")
    lines += ["", "- ②는 눌림이 안 온 날 10시에 시가로 샀다고 보는 것이라 실제보다 유리하다. ①의 (시가+고가)/2 는 반대로 불리하다. 실제는 둘 사이.",
              "- ④는 비용까지 0으로 둔 상한선. 여기서도 거래당 +0.6% 수준이면, 비용(왕복 0.4~0.6%)을 넣은 현실에선 손익분기 근처다."]
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"sensitivity_{label}_{last}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n리포트: {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "sim", "all", "snapshot", "sens"])
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 다시 받기")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.cmd in ("fetch", "all"):
        fetch(args.refresh)
    if args.cmd in ("sim", "all"):
        sim()
    if args.cmd == "snapshot":
        make_snapshot()
    if args.cmd == "sens":
        sens()
    return 0


if __name__ == "__main__":
    sys.exit(main())
