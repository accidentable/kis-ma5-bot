"""
backtest/lab/panic_sim.py — 패닉 매수 연구용 공통 도구

  build_ctx()        데이터 + 패닉 연구용 추가 지표 (전부 그날 종가까지의 정보)
  episodes()         패닉일을 사건(에피소드)으로 묶기 — 가까운 날끼리 하나로
  event_study()      사건 조사: 진입(당일 종가 / 다음 날 시가) → n일 뒤 종가, 비용 후, 에피소드 단위 통계 포함
  run_month_hybrid() 한 달(21거래일) 대회 시뮬레이션: 평소 = 현재 봇 전략, 여기에
                     시장 패닉 모드(보유 전량 교체) 와 종목 패닉 매수(빈 슬롯 / 보유 교체) 를 얹는다

현재 봇 전략 (평소 모드): 시총 200, 주가 30만원 이하, 60일 수익률 > 0, 20일 내 +10% 급등일 없음,
종가/250일 최고가 순 2종목, 21거래일마다 교체(순위 안이면 유지), 빈 슬롯은 다음 날 시가에 채움,
보유 종목이 하루 +10%↑ 면 그날 종가 매도.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtest.lab import engine as E

COST_RT = 0.0035     # 사건 조사용 대략 왕복 비용 (시뮬레이션은 틱 단위로 따로 계산)


# ════════════════════════════════════════════════════════════
# 데이터
# ════════════════════════════════════════════════════════════
def build_ctx(sector_csv: str | None = None):
    m = E.load()
    I = m.ind
    c = m.c.astype(np.float64)
    o, h, l = (x.astype(np.float64) for x in (m.o, m.h, m.l))
    val = m.val.astype(np.float64)
    prev = E._shift(c, 1)
    X: dict = {}
    X["tradable"] = (~np.isnan(c)) & (m.raw >= 1000) & (val > 0) & (I["val20"] >= 1e9)
    X["afford"] = m.raw <= 300_000
    X["liquid"] = X["tradable"] & (I["val20"] >= 3e9)
    X["vrank"] = ((-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1) + 1).astype(np.int32)
    X["attn20"] = E._shift((E._roll_max((X["vrank"] <= 50).astype(np.float64), 20) > 0).astype(np.float64), 1) > 0
    mr = I["mkt_ret1"]
    X["mkt_ret1"] = mr
    # 베타: 직전 60일 (당일 제외)
    r1 = np.nan_to_num(I["ret1"])
    mrr = np.repeat(mr[:, None], r1.shape[1], axis=1)
    cov = E._roll_mean(r1 * mrr, 60) - E._roll_mean(r1, 60) * E._roll_mean(mrr, 60)
    var = E._roll_mean(mr[:, None] ** 2, 60) - E._roll_mean(mr[:, None], 60) ** 2
    beta = E._shift(cov / var, 1)
    del mrr, cov
    X["beta60"] = np.clip(beta, -1, 4)
    X["idio1"] = I["ret1"] - X["beta60"] * mr[:, None]
    X["gap"] = o / prev - 1
    X["ret2"] = c / E._shift(c, 2) - 1
    X["ret3"] = c / E._shift(c, 3) - 1
    X["dd20"] = c / E._roll_max(h, 20) - 1
    X["dd60"] = c / E._roll_max(h, 60) - 1
    X["z1"] = I["ret1"] / E._shift(I["vol60"], 1)                       # 오늘 하락의 표준편차 배수
    X["atr_dist20"] = (c - I["ma20"]) / I["atr14"]
    X["vr1"] = val / E._shift(I["val20"], 1)                            # 오늘 거래대금 / 20일 평균
    down = (I["ret1"] < 0).astype(np.int16)
    cons = np.zeros_like(down)
    for t in range(1, m.T):
        cons[t] = np.where(down[t] > 0, cons[t - 1] + 1, 0)
    X["down_days"] = cons
    # 시장 지표
    liq = X["liquid"]
    X["breadth_dn3"] = np.nanmean(np.where(liq, (I["ret1"] <= -0.03).astype(float), np.nan), axis=1)
    mk = I["mkt"]
    X["mkt_ret3"] = mk / E._shift(mk[:, None], 3)[:, 0] - 1
    X["mkt_dd60"] = mk / E._roll_max(mk[:, None], 60)[:, 0] - 1
    X["mkt_above200"] = mk > E._roll_mean(mk[:, None], 200)[:, 0]
    X["mkt_vol20"] = np.sqrt(E._roll_mean(mr[:, None] ** 2, 20))[:, 0]
    # KOSPI200 대용 (전일 시총 가중) · 코스닥 대용
    mkts = np.array([str(x) for x in np.load(E.NPZ, allow_pickle=True)["market"]])
    capp = np.nan_to_num(E._shift(m.cap.astype(np.float64), 1))
    for key, sel, n in (("k200", mkts == "KOSPI", 200), ("kq", np.isin(mkts, ["KOSDAQ", "KOSDAQ GLOBAL"]), 150)):
        w = np.where(sel[None, :], capp, 0)
        thr = -np.sort(-w, axis=1)[:, n - 1:n]
        w = np.where(w >= thr, w, 0)
        X[f"{key}_ret1"] = (w * np.clip(r1, -0.3, 0.3)).sum(1) / np.maximum(w.sum(1), 1)
    X["is_kosdaq"] = np.isin(mkts, ["KOSDAQ", "KOSDAQ GLOBAL"])
    # 업종 (선택)
    if sector_csv:
        import pandas as pd
        d = pd.read_csv(sector_csv, dtype=str)
        lab = dict(zip(d["Symbol"].str.zfill(6), d["Sector"]))
        secs = sorted({s for s in lab.values() if isinstance(s, str)})
        sid = {s: i for i, s in enumerate(secs)}
        col = np.array([sid.get(lab.get(cd), -1) for cd in m.codes])
        sr = np.full((m.T, len(secs)), np.nan)
        for s_ in range(len(secs)):
            mem = col == s_
            if mem.sum() < 5:
                continue
            x = np.where(X["tradable"][:, mem], I["ret1"][:, mem], np.nan)
            cnt = np.sum(np.isfinite(x), axis=1)
            sr[:, s_] = np.where(cnt >= 5, np.nanmean(x, axis=1), np.nan)
        X["sector"] = col
        X["sector_ret1"] = np.where(col[None, :] >= 0, sr[:, np.maximum(col, 0)], np.nan)
    return m, X


def episodes(day_mask: np.ndarray, gap: int = 10) -> np.ndarray:
    """True 인 날들을 gap 거래일 이내면 같은 사건으로 묶는다. 반환: 날짜별 사건 번호 (없으면 -1)."""
    ep = np.full(len(day_mask), -1)
    cur, last = -1, -10 ** 9
    for t in np.where(day_mask)[0]:
        if t - last > gap:
            cur += 1
        ep[t] = cur
        last = t
    return ep


# ════════════════════════════════════════════════════════════
# 사건 조사
# ════════════════════════════════════════════════════════════
def forward_returns(m, entry: str, horizons=(1, 3, 5, 10, 21)) -> dict:
    c, o = m.c.astype(np.float64), m.o.astype(np.float64)
    if entry == "close":
        return {n: E._shift(c, -n) / c - 1 - COST_RT for n in horizons}
    nxo = E._shift(o, -1)
    return {n: E._shift(c, -n) / nxo - 1 - COST_RT for n in horizons}


def event_study(m, X, mask: np.ndarray, fwd: dict, t0: int, t1: int, ep_gap: int = 10, max_per_day: int | None = None,
                order: np.ndarray | None = None) -> dict:
    """mask: T×N 매수 사건. max_per_day 가 있으면 order(클수록 먼저) 상위만. 결과: 지평별 통계 + 에피소드 통계."""
    msk = mask.copy()
    msk[:t0] = False
    msk[t1:] = False
    if max_per_day:
        sc = np.where(msk, order, -np.inf)
        keep = np.zeros_like(msk)
        for t in np.where(msk.any(1))[0]:
            idx = np.argsort(-sc[t])[:max_per_day]
            idx = idx[np.isfinite(sc[t][idx])]
            keep[t, idx] = True
        msk = keep
    days = msk.any(1)
    ep = episodes(days, ep_gap)
    out = {"n_events": int(msk.sum()), "n_days": int(days.sum()), "n_episodes": int(ep.max() + 1) if days.any() else 0}
    tr = X["tradable"]
    for n, y in fwd.items():
        s = msk & np.isfinite(y)
        if s.sum() == 0:
            continue
        v = y[s]
        base = np.nanmean(np.where(tr & np.isfinite(y), y, np.nan), axis=1)
        exc = v - np.broadcast_to(base[:, None], y.shape)[s]
        # 날짜 평균 → 에피소드 평균
        day_mean = np.array([np.nanmean(np.where(s[t], y[t], np.nan)) for t in range(m.T)])
        ep_means = np.array([np.nanmean(day_mean[(ep == e) & np.isfinite(day_mean)]) for e in range(out["n_episodes"])])
        ep_means = ep_means[np.isfinite(ep_means)]
        loo = [np.delete(ep_means, i).mean() for i in range(len(ep_means))] if len(ep_means) > 1 else [np.nan]
        out[n] = {"mean": float(v.mean()), "median": float(np.median(v)), "win": float((v > 0).mean()),
                  "excess": float(exc.mean()), "ep_mean": float(ep_means.mean()) if len(ep_means) else np.nan,
                  "ep_median": float(np.median(ep_means)) if len(ep_means) else np.nan,
                  "ep_pos": float((ep_means > 0).mean()) if len(ep_means) else np.nan,
                  "ep_loo_min": float(np.nanmin(loo)), "n": int(s.sum())}
    return out


# ════════════════════════════════════════════════════════════
# 한 달 하이브리드 시뮬레이션
# ════════════════════════════════════════════════════════════
@dataclass
class Exit:
    hold: int = 5                 # 최대 보유 거래일 (진입일 = 1일차)
    tp: float = 0.0               # 장중 익절 (진입가 대비). 0 이면 없음
    stop: float = 0.0             # 장중 손절. 0 이면 없음
    at: str = "close"             # 보유기간 만료 시 'close' (그날 종가) / 'next_open' (다음 날 시가)
    ma5_recover: bool = False     # 종가가 5일선 위로 올라오면 종가 매도


@dataclass
class Hybrid:
    name: str
    base_top: list                          # 평소 전략 순위 (날짜별 종목 인덱스 배열)
    surge_exit: np.ndarray                  # 평소 보유 급등 매도 (T×N bool)
    # 시장 패닉
    mp_sig: np.ndarray | None = None        # 날짜 bool: 이 날 종가 정보로 패닉 진입
    mp_rank_day: np.ndarray | None = None   # 날짜 int: 후보 순위를 어느 날 것으로 쓰나 (보통 패닉일)
    mp_top: list | None = None              # 날짜별 패닉 후보 (종목 인덱스, 좋은 순)
    mp_entry: str = "open"                  # 'close' (신호일 종가) / 'open' (다음 날 시가)
    mp_exit: Exit = field(default_factory=Exit)
    mp_slots: int = 2                       # 패닉 모드에서 쓸 슬롯 수 (나머지는 평소 보유 유지)
    # 종목 패닉
    sp_top: list | None = None              # 날짜별 종목 패닉 후보
    sp_entry: str = "open"
    sp_exit: Exit = field(default_factory=Exit)
    sp_mode: str = "fill"                   # 'fill' 빈 슬롯만 / 'replace' 평소 보유 중 순위 낮은 것을 판다
    sp_max: int = 1                         # 동시에 들 수 있는 종목 패닉 포지션 수


def run_month_hybrid(m, X, cfg: Hybrid, s: int, k: int = 2, capital: float = E.CAPITAL):
    e = min(s + E.MONTH - 1, m.T - 1)
    cash = float(capital)
    slot = capital / k
    pos: dict = {}                # j -> dict(cost, ea, t0, kind, exit)
    pend_open: set = set()        # 다음 날 시가 매도 예약
    stats = {"mp": 0, "sp": 0, "base": 0}
    next_rebal = s
    mp_until = -1                 # 시장 패닉 포지션이 남아 있는 동안 평소 매수 중지

    def raw_px(t, j, px):
        return px * m.raw[t, j] / m.c[t, j] if m.c[t, j] > 0 else px

    def sell(j, t, px, extra=0.0):
        nonlocal cash
        p = pos.pop(j)
        net = px * (1 - E.slip(raw_px(t, j, px)) - extra) * (1 - E.FEE - E.TAX)
        cash += p["cost"] * net / p["ea"]
        pend_open.discard(j)

    def buy(j, t, kind, ex, at_close=False):
        nonlocal cash
        if j in pos or len(pos) >= k:
            return False
        px_adj = m.c[t, j] if at_close else m.o[t, j]
        cp = m.c[t - 1, j]
        if not (px_adj > 0) or not (cp > 0):
            return False
        if not at_close and px_adj >= cp * 1.295:
            return False          # 시가 상한가: 못 산다
        if at_close and px_adj >= cp * 1.295:
            return False          # 종가 상한가: 못 산다
        rp = raw_px(t, j, px_adj)
        px = rp * (1 + E.slip(rp))
        q = int(min(cash, slot) / (px * (1 + E.FEE)))
        if q < 1:
            return False
        cost = q * px * (1 + E.FEE)
        cash -= cost
        pos[j] = {"cost": cost, "ea": px_adj * (1 + E.slip(rp)) * (1 + E.FEE), "t0": t, "kind": kind, "exit": ex,
                  "close_entry": at_close}
        stats[kind] += 1
        return True

    def base_score_rank(sig):
        return {int(j): r for r, j in enumerate(cfg.base_top[sig])}

    for t in range(s, e + 1):
        sig = t - 1
        # ① 시가: 예약 매도
        for j in list(pend_open):
            if j in pos and m.o[t, j] > 0:
                sell(j, t, float(m.o[t, j]))
        # ② 시장 패닉 진입 (시가)
        if cfg.mp_sig is not None and cfg.mp_entry == "open" and cfg.mp_sig[sig] and t > mp_until:
            rd = int(cfg.mp_rank_day[sig])
            cands = cfg.mp_top[rd]
            if len(cands):
                # 평소 보유를 팔아 mp_slots 만큼 자리를 만든다 (순위 낮은 것부터)
                rk = base_score_rank(sig)
                base_hold = sorted([j for j in pos if pos[j]["kind"] != "mp"], key=lambda j: -rk.get(j, 10 ** 6))
                need = max(0, len(pos) + cfg.mp_slots - k)
                for j in base_hold[:need]:
                    if m.o[t, j] > 0:
                        sell(j, t, float(m.o[t, j]))
                bought = 0
                for j in cands:
                    if bought >= cfg.mp_slots:
                        break
                    if buy(int(j), t, "mp", cfg.mp_exit):
                        bought += 1
                if bought:
                    mp_until = t + cfg.mp_exit.hold - 1
        in_mp = any(p["kind"] == "mp" for p in pos.values())
        # ③ 종목 패닉 진입 (시가)
        if cfg.sp_top is not None and cfg.sp_entry == "open" and not in_mp:
            n_sp = sum(1 for p in pos.values() if p["kind"] == "sp")
            for j in cfg.sp_top[sig]:
                if n_sp >= cfg.sp_max:
                    break
                j = int(j)
                if j in pos:
                    continue
                if len(pos) >= k and cfg.sp_mode == "replace":
                    rk = base_score_rank(sig)
                    bh = sorted([x for x in pos if pos[x]["kind"] == "base"], key=lambda x: -rk.get(x, 10 ** 6))
                    if bh and m.o[t, bh[0]] > 0:
                        sell(bh[0], t, float(m.o[t, bh[0]]))
                if buy(j, t, "sp", cfg.sp_exit):
                    n_sp += 1
        # ④ 평소 전략 (시장 패닉 포지션이 없을 때)
        in_mp = any(p["kind"] == "mp" for p in pos.values())
        if not in_mp:
            if t >= next_rebal:
                keep = set(int(x) for x in cfg.base_top[sig][:k])
                for j in [x for x in pos if pos[x]["kind"] == "base" and x not in keep]:
                    if m.o[t, j] > 0:
                        sell(j, t, float(m.o[t, j]))
                next_rebal = t + 21
            for j in cfg.base_top[sig]:
                if len(pos) >= k:
                    break
                buy(int(j), t, "base", None)
        # ⑤ 장중·종가 청산
        for j in list(pos):
            p = pos[j]
            o_, h_, l_, c_ = m.o[t, j], m.h[t, j], m.l[t, j], m.c[t, j]
            if not (c_ > 0):
                continue
            if p["kind"] == "base":
                if cfg.surge_exit[t, j] and p["t0"] < t:
                    sell(j, t, float(c_))
                continue
            ex: Exit = p["exit"]
            started = p["t0"] < t or not p["close_entry"]
            if started and ex.stop:
                sl = p["ea"] * (1 - ex.stop)
                if p["t0"] < t and o_ <= sl:
                    sell(j, t, float(o_), 0.002); continue
                if l_ <= sl:
                    sell(j, t, float(sl), 0.002); continue
            if started and ex.tp:
                tpp = p["ea"] * (1 + ex.tp)
                if p["t0"] < t and o_ >= tpp:
                    sell(j, t, float(o_)); continue
                if h_ >= tpp:
                    sell(j, t, float(tpp)); continue
            held = t - p["t0"] + (0 if p["close_entry"] else 1)
            if ex.ma5_recover and p["t0"] < t and c_ > m.ind["ma5"][t, j]:
                sell(j, t, float(c_)); continue
            if held >= ex.hold:
                if ex.at == "close" or t == e:
                    sell(j, t, float(c_))
                else:
                    pend_open.add(j)
        if not any(p["kind"] == "mp" for p in pos.values()) and mp_until >= t:
            mp_until = t
            next_rebal = t + 1
        # ⑥ 종가 진입 (시장 패닉 / 종목 패닉)
        if t < e and cfg.mp_sig is not None and cfg.mp_entry == "close" and cfg.mp_sig[t] and t > mp_until:
            rd = int(cfg.mp_rank_day[t])
            cands = cfg.mp_top[rd]
            if len(cands):
                rk = base_score_rank(t)
                base_hold = sorted([j for j in pos if pos[j]["kind"] != "mp"], key=lambda j: -rk.get(j, 10 ** 6))
                need = max(0, len(pos) + cfg.mp_slots - k)
                for j in base_hold[:need]:
                    sell(j, t, float(m.c[t, j]))
                bought = 0
                for j in cands:
                    if bought >= cfg.mp_slots:
                        break
                    if buy(int(j), t, "mp", cfg.mp_exit, at_close=True):
                        bought += 1
                if bought:
                    mp_until = t + cfg.mp_exit.hold
        if t < e and cfg.sp_top is not None and cfg.sp_entry == "close" and not any(p["kind"] == "mp" for p in pos.values()):
            n_sp = sum(1 for p in pos.values() if p["kind"] == "sp")
            for j in cfg.sp_top[t]:
                if n_sp >= cfg.sp_max:
                    break
                j = int(j)
                if j in pos:
                    continue
                if len(pos) >= k and cfg.sp_mode == "replace":
                    rk = base_score_rank(t)
                    bh = sorted([x for x in pos if pos[x]["kind"] == "base"], key=lambda x: -rk.get(x, 10 ** 6))
                    if bh:
                        sell(bh[0], t, float(m.c[t, bh[0]]))
                if buy(j, t, "sp", cfg.sp_exit, at_close=True):
                    n_sp += 1
    # ⑦ 마지막 날 종가 전량
    for j in list(pos):
        c_ = m.c[e, j]
        if not (c_ > 0):
            back = np.where(m.c[:e + 1, j] > 0)[0]
            c_ = m.c[back[-1], j] if len(back) else pos[j]["ea"]
        sell(j, e, float(c_))
    return cash / capital - 1, stats


def month_stats(rets) -> dict:
    a = np.asarray(rets, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p10d": float((a <= -0.1).mean()),
            "p10u": float((a >= 0.1).mean()), "p_pos": float((a > 0).mean()), "n": len(a)}


def ranked_lists(score: np.ndarray, K: int = 12) -> list:
    """T×N 점수 (NaN = 후보 아님, 클수록 먼저) → 날짜별 상위 K 종목 인덱스."""
    sp = E.Spec("tmp", score, "signal").prepare(K)
    return sp.top
