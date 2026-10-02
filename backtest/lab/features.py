"""
backtest/lab/features.py — 기술적 지표 라이브러리 + 기간별 예측력(IC) 측정

  python -m backtest.lab.features     → backtest/results/lab_features.md

지표는 전부 그날 종가까지의 정보만 쓴다 (다음 날 시가에 사는 전략에 그대로 쓸 수 있다).
예측 대상: 다음 날 시가 → 21거래일 뒤 종가 수익률 (한 달 보유).
IC: 21거래일마다 표본일을 잡아, 그날 유니버스 안에서 지표 순위와 이후 수익률 순위의 상관(스피어만).
  IS 2011 ~ 2018 / VAL 2019 ~ 2024.9 / TEST 2024.10 ~ 로 나눠 평균·t 값을 본다.
상위 10% 초과수익: 지표 상위 10% 종목의 한 달 수익률 − 유니버스 평균 (비용 전).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from backtest.lab import engine as E

warnings.filterwarnings("ignore")
OUT = os.path.join(E.ROOT, "backtest", "results")
F32 = np.float32


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2 / (span + 1)
    out = np.full_like(x, np.nan, dtype=np.float64)
    prev = np.full(x.shape[1], np.nan)
    for t in range(x.shape[0]):
        v = x[t]
        prev = np.where(np.isnan(prev), v, np.where(np.isnan(v), prev, a * v + (1 - a) * prev))
        out[t] = prev
    return out


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = c - E._shift(c, 1)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = E._roll_mean(up, n), E._roll_mean(dn, n)
    return np.where(ad > 0, 100 - 100 / (1 + au / np.where(ad > 0, ad, 1)), 100.0)


def build(m) -> dict:
    """{이름: T×N float32}. 이름 앞 숫자는 분류."""
    I = m.ind
    c = m.c.astype(np.float64)
    o, h, l = (x.astype(np.float64) for x in (m.o, m.h, m.l))
    val = m.val.astype(np.float64)
    vol_sh = np.where(c > 0, val / c, np.nan)              # 거래량 대용 (거래대금 / 가격)
    prev = E._shift(c, 1)
    F: dict[str, np.ndarray] = {}

    def put(k, x):
        F[k] = np.asarray(x, dtype=F32)

    # 1 추세·모멘텀
    for n in (5, 20, 60, 120, 250):
        put(f"수익률{n}일", I[f"ret{n}"])
    put("모멘텀12-1", I["mom12_1"])
    # 2 이동평균·이격도
    for n in (5, 20, 60, 120):
        put(f"이격도{n}", c / I[f"ma{n}"] - 1)
    put("이평선 정배열 점수", (I["ma5"] > I["ma20"]).astype(float) + (I["ma20"] > I["ma60"]) + (I["ma60"] > I["ma120"]))
    put("20일선/60일선", I["ma20"] / I["ma60"] - 1)
    put("60일선/120일선", I["ma60"] / I["ma120"] - 1)
    ma20_slope = I["ma20"] / E._shift(I["ma20"], 5) - 1
    put("20일선 기울기(5일)", ma20_slope)
    # 3 오실레이터
    put("RSI2", I["rsi2"])
    put("RSI14", _rsi(c, 14))
    hh20, ll20 = E._roll_max(h, 20), E._roll_min(l, 20)
    put("스토캐스틱%K20", (c - ll20) / np.where(hh20 > ll20, hh20 - ll20, np.nan))
    sd20 = np.sqrt(E._roll_mean(c ** 2, 20) - I["ma20"] ** 2)
    put("볼린저%b", (c - (I["ma20"] - 2 * sd20)) / np.where(sd20 > 0, 4 * sd20, np.nan))
    put("볼린저 폭", 4 * sd20 / I["ma20"])
    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    put("MACD/주가", macd / c)
    put("MACD 히스토그램/주가", (macd - sig) / c)
    put("IBS(당일 종가 위치)", (c - l) / np.where(h > l, h - l, np.nan))
    # 4 거래량·거래대금
    v5, v20, v60 = E._roll_mean(val, 5), I["val20"], E._roll_mean(val, 60)
    put("거래대금 5일/60일", v5 / v60)
    put("거래대금 당일/20일", val / E._shift(v20, 1))
    put("회전율(20일 거래대금/시총)", v20 / m.cap)
    put("거래대금 규모(20일, log)", np.log(v20))
    up_v = E._roll_mean(np.where(c > prev, val, 0.0), 20)
    put("상승일 거래대금 비중(20일)", up_v / v20)
    # 5 매물대 (가격대별 거래량)
    for n in (60, 120):
        vwap = E._roll_mean(c * vol_sh, n) / E._roll_mean(vol_sh, n)
        put(f"VWAP{n} 대비", c / vwap - 1)
    # 현재가 위에 쌓인 거래량 비중 (120일): 위쪽 매물 부담
    above = np.zeros_like(c)
    tot = np.zeros_like(c)
    for k in range(120):
        ck, vk = (c, vol_sh) if k == 0 else (E._shift(c, k), E._shift(vol_sh, k))
        ok = np.isfinite(ck) & np.isfinite(vk)
        tot += np.where(ok, vk, 0)
        above += np.where(ok & (ck > c * 1.02), vk, 0)
    put("위쪽 매물 비중(120일, 현재가+2%↑)", np.where(tot > 0, above / tot, np.nan))
    # 6 변동성·복권성
    put("변동성20", np.sqrt(E._roll_mean(I["ret1"] ** 2, 20)))
    put("변동성60", I["vol60"])
    put("ATR14/주가", I["atr14"] / c)
    put("MAX20(최대 일간상승)", I["max20"])
    put("MIN20(최대 일간하락)", E._roll_min(I["ret1"], 20))
    # 7 가격 위치
    put("52주 최고가 대비", c / I["hi250"])
    put("52주 최저가 대비", c / E._roll_min(l, 250) - 1)
    put("20일 고점 대비", c / hh20 - 1)
    # 8 장중 vs 야간 (Lou·Polk·Skouras 줄다리기)
    put("장중수익 누적20", E._roll_mean(c / o - 1, 20) * 20)
    put("야간수익 누적20", E._roll_mean(o / prev - 1, 20) * 20)
    # 9 규모
    put("시가총액(log)", np.log(m.cap))
    put("주가 수준(log)", np.log(m.raw))
    return F


def forward(m, n: int = 21) -> np.ndarray:
    """t 일 신호 → t+1 시가 매수 → t+n 종가 매도 수익률 (비용 전)."""
    c, o = m.c.astype(np.float64), m.o.astype(np.float64)
    return E._shift(c, -n) / E._shift(o, -1) - 1


def spearman_rows(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    a = x[ok].argsort().argsort().astype(float)
    b = y[ok].argsort().argsort().astype(float)
    return float(np.corrcoef(a, b)[0, 1])


def top_spread(x, y, mask, q=0.1) -> float:
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    xs, ys = x[ok], y[ok]
    thr = np.quantile(xs, 1 - q)
    return float(ys[xs >= thr].mean() - ys.mean())


PERIODS = {"IS 2011~2018": ("20110101", "20181231"), "VAL 2019~2024.9": ("20190101", "20240930"),
           "TEST 2024.10~": ("20241001", "99999999")}


def universes(m) -> dict:
    I = m.ind
    base = (~np.isnan(m.c)) & (m.raw >= 1000) & (I["val20"] >= 1e9) & (m.val > 0) & np.isfinite(I["hi250"])
    return {"시총 상위 200": base & (I["caprank"] <= 200),
            "거래대금 30억↑ 전체": base & (I["val20"] >= 3e9)}


def main() -> int:
    m = E.load()
    F = build(m)
    y = forward(m, 21)
    U = universes(m)
    L = ["# 기술적 지표 예측력 (IC) — 한 달 보유 기준", "",
         "KRX 전종목(상장폐지 포함). 21거래일마다 표본일, 그날 유니버스 안에서 지표 순위 vs 다음 날 시가→21일 뒤 종가 수익률 순위.",
         "IC 는 평균 스피어만 상관, t 는 평균/표준오차. '상위10%' 는 지표 상위 10% 의 한 달 초과수익(비용 전).",
         "|IC| 0.02 이상이 여러 기간에서 같은 부호로 유지되면 쓸 만한 신호로 본다.", ""]
    for un, mask in U.items():
        L += [f"## {un}", "", "| 지표 | " + " | ".join(f"{p} IC (t) / 상위10%" for p in PERIODS) + " |",
              "|---|" + "---|" * len(PERIODS)]
        rows = []
        for k, x in F.items():
            cells, ics = [], []
            for p, (a, z) in PERIODS.items():
                s0, s1 = m.didx(a), min(m.didx(z), m.T - 22)
                ts = range(max(s0, 260), s1, 21)
                ic = np.array([spearman_rows(x[t], y[t], mask[t]) for t in ts])
                sp = np.array([top_spread(x[t], y[t], mask[t]) for t in ts])
                ic = ic[np.isfinite(ic)]
                tt = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 2 else 0
                cells.append(f"{ic.mean():+.3f} ({tt:+.1f}) / {np.nanmean(sp) * 100:+.1f}%")
                ics.append(ic.mean())
            rows.append((k, cells, ics))
            print(un, k, [f"{v:+.3f}" for v in ics], flush=True)
        for k, cells, _ in sorted(rows, key=lambda r: -abs(r[2][0])):
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    p = os.path.join(OUT, "lab_features.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
