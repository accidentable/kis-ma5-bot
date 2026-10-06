"""
backtest/lab/us/common.py — 미국 일봉 데이터 로더 + 공통 계산 (build_data.py 가 만든 /data/lab/us/prices.npz)

수정주가(adj) 비율로 시가·고가·저가도 보정해서 수익률 계산에 쓴다. 가격 수준 필터(페니주 제외)는 원래 종가.
대상 U: 거래됨 · 원래 종가 $5↑ · (현재) 시총 $2B↑ (전부) — 생존 편향 주의.
"""
from __future__ import annotations

import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

DIR = '/data/lab/us'
BUYC = SELLC = 0.003                     # 편도 0.30% (한투 해외주식 0.25% + 슬리피지)
PER = {'2011~2019': ('20110103', '20200101'), '2020~2025': ('20200101', '20260101'), '2026': ('20260101', '20991231')}


class US:
    def __init__(self):
        z = np.load(f'{DIR}/prices.npz', allow_pickle=False)
        self.dates = z['dates'].astype(str); self.tickers = z['tickers'].astype(str)
        self.T, self.N = len(self.dates), len(self.tickers)
        raw_c = z['close'].astype(float); adj = z['adj'].astype(float)
        f = np.where(raw_c > 0, adj / raw_c, np.nan)
        self.raw = raw_c
        self.o, self.h, self.l = z['open'].astype(float) * f, z['high'].astype(float) * f, z['low'].astype(float) * f
        self.c = adj; self.vol = z['volume'].astype(float)
        self.meta = json.load(open(f'{DIR}/meta.json'))
        self.cap = np.array([self.meta.get(t, {}).get('cap', 0.0) for t in self.tickers])
        self.sector = np.array([self.meta.get(t, {}).get('sector', '') for t in self.tickers])
        self.is_etf = np.array([self.meta.get(t, {}).get('sector', '') == 'ETF' for t in self.tickers])
        self.is_lev = np.array([self.meta.get(t, {}).get('industry', '') == 'leveraged' for t in self.tickers])
        self.tr = np.isfinite(self.c) & (self.vol > 0)
        self.cf = ffill(np.where(self.tr, self.c, np.nan))
        self.pc = lag(self.cf, 1)
        self.r1 = self.c / self.pc - 1
        self.U = self.tr & (self.raw >= 5) & ~self.is_etf & (self.cap >= 2e9)
        self.caprank = np.argsort(np.argsort(-self.cap)) + 1          # 현재 시총 순위
        self.mkt = np.nanmean(np.where(self.U, np.clip(self.r1, -.5, .5), np.nan), axis=1)
        self.sig20 = np.full(self.T, np.nan)
        for t in range(21, self.T): self.sig20[t] = np.nanstd(self.mkt[t - 20:t])
        self.crash = (self.mkt <= -0.03) & (self.mkt <= -3 * self.sig20)
        self.val = self.c * self.vol
        self.val20p = lag(roll_mean(self.val, 20), 1)
        self.vr = self.val / self.val20p
        self.vol60 = lag(roll_std(self.r1, 60), 1)
        self.ret20 = self.cf / lag(self.cf, 20) - 1
        self.ret5 = self.cf / lag(self.cf, 5) - 1
        self.ret60 = self.cf / lag(self.cf, 60) - 1
        self.hi250 = lag(roll_max(self.cf, 250), 1)
        self.hi20p = lag(roll_max(self.cf, 20), 1)

    def didx(self, d: str) -> int:
        return int(np.searchsorted(self.dates, d))

    def col(self, ticker: str) -> int:
        return int(np.where(self.tickers == ticker)[0][0])

    def periods(self, margin: int = 21):
        out = {}
        for pn, (a, b) in PER.items():
            out[pn] = (max(self.didx(a), 1), min(self.didx(b), self.T - margin))
        return out


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k):
    out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


def lead(x, k):
    out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out


def roll_max(x, n):
    pad = np.vstack([np.full((n - 1, x.shape[1]), np.nan), x])
    return np.nanmax(swv(pad, n, axis=0), axis=-1)


def roll_mean(x, n):
    pad = np.vstack([np.full((n - 1, x.shape[1]), np.nan), x])
    return np.nanmean(swv(pad, n, axis=0), axis=-1)


def roll_std(x, n):
    pad = np.vstack([np.full((n - 1, x.shape[1]), np.nan), x])
    return np.nanstd(swv(pad, n, axis=0), axis=-1)


def pct_mask(x, base, lo, hi):
    T, N = x.shape; out = np.zeros((T, N), bool)
    for t in range(T):
        ii = np.where(base[t] & np.isfinite(x[t]))[0]
        if len(ii) < 20: continue
        order = ii[np.argsort(x[t, ii])]; n = len(order); out[t, order[int(n * lo):int(n * hi)]] = True
    return out


def lists(mask, score, desc=True):
    out = []
    for t in range(mask.shape[0]):
        ii = np.where(mask[t] & np.isfinite(score[t]))[0]
        out.append(ii[np.argsort(-score[t, ii] if desc else score[t, ii])].tolist())
    return out
