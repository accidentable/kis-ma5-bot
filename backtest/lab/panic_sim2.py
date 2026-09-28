"""
backtest/lab/panic_sim2.py — 패닉 매수 연구 2차 시뮬레이터 (panic_sim 확장, 설계 검토 반영)

panic_sim.run_month_hybrid 대비 달라진 점
  - 거래정지일(거래대금 0) 에는 체결하지 않는다. 팔 수 없으면 슬롯을 계속 차지한다.
  - 하한가 마감일의 종가 매도는 다음 날 시가로 미룬다 (limit-down deferral).
  - 청산: FIX(보유일) · 5일선 회복(REC) · 진입일 저가 이탈(FAIL) · 시장 기준 익절(X-MKT) · 시장 손절(MKT-STOP)
          · 장중 익절/손절 · 급등(+10%) 종가 매도
  - 시장 패닉 모드: 전량(ALL-IN) 또는 1+1 분할(STAGED), 사건당 진입 횟수 제한(re-arm)
  - 대조군(R0-REFRESH): run_month(force_rebal=하이브리드가 평소 전략으로 돌아온 날들) — 매매 없이 같은 날 재정렬만
  - 종목 패닉: REPLACE1 (빈 슬롯 먼저, 없으면 평소 보유 중 순위 가장 낮은 것을 같은 동시호가에 판다),
              진입 '당일 종가' / '다음 날 시가' / '당일 시가(갭 확인 후, 추가 슬리피지)'
  - 합성 지수(KOSPI200 대용) 한 종목을 열 하나로 붙여 매매할 수 있다 (세금 없음, 5원 틱).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtest.lab import engine as E


@dataclass
class Exit:
    hold: int = 5                # 최대 보유 거래일 (진입일 = 1일차; 종가 진입은 다음 날이 1일차)
    at: str = "close"            # 보유 만료 청산 시점: 'close' | 'next_open'
    tp: float = 0.0              # 장중 익절 (진입가 대비)
    stop: float = 0.0            # 장중 손절
    rec: bool = False            # 종가 ≥ 5일선이면 종가 매도 (진입 다음 날부터)
    fail_low: bool = False       # 종가 < 진입일 저가면 종가 매도
    surge: bool = False          # 하루 +10%↑ 면 종가 매도
    mkt_tp: float = 0.0          # 시장지수 M 이 기준일 대비 +X 이상인 첫 종가에 매도
    mkt_stop: float = 0.0        # 시장지수 M 이 기준일 대비 −Y 이하인 종가에 매도
    rec_arr: np.ndarray | None = None   # REC 판정을 바꿔 끼울 T×N bool (종가 잡음 검사용). None 이면 c ≥ ma5
    defer_all: bool = False      # 종가에 정하는 모든 청산(보유 만료 · REC · FAIL · 시장 · 급등)을 다음 날 시가로 (강건성 검사용)
    defer_rules: bool = False    # 정확한 종가가 필요한 청산(REC · FAIL · 시장 · 급등)만 다음 날 시가로. 보유 만료는 그대로 종가


@dataclass
class Hybrid:
    name: str
    base_top: list
    surge_exit: np.ndarray
    # 시장 패닉
    mp_sig: np.ndarray | None = None          # 날짜 bool (신호가 확정되는 날)
    mp_rank_day: np.ndarray | None = None     # 신호일 → 후보 목록을 볼 날
    mp_anchor_day: np.ndarray | None = None   # 신호일 → 시장 기준일 (X-MKT / 손절 기준)
    mp_top: list | None = None                # 날짜별 후보 (인덱스는 rank_day)
    mp_entry: str = "open"                    # 'open' (신호 다음 날 시가) | 'close' (신호일 종가)
    mp_exit: Exit = field(default_factory=Exit)
    mp_mode: str = "allin"                    # 'allin' | 'staged'
    stage2_sig: np.ndarray | None = None      # staged: 이 날 종가 신호면 두 번째 슬롯 전환 (다음 날 시가)
    stage2_drop: float = 0.0                  # staged: M ≤ M_t0·(1−D) 인 종가도 두 번째 슬롯 신호
    stage2_window: int = 10
    mp_episode: np.ndarray | None = None      # 날짜 → 사건 번호 (재진입 제한용)
    mp_max_entries: int = 0                   # 사건당 최대 진입 횟수 (0 = 제한 없음)
    # 종목 패닉
    sp_top: list | None = None                # 날짜별 후보 (결정일 기준)
    sp_entry: str = "close"                   # 'close' | 'open' (다음 날) | 'open_same' (당일 시가)
    sp_exit: Exit = field(default_factory=Exit)
    sp_extra_slip: float = 0.0
    sp_max: int = 1                           # 동시에 들 수 있는 종목 패닉 포지션 수 (A18 단독 모드 = 2)
    idx_col: int = -1                         # 합성 지수 열 (세금 없음, 가용 현금 전부로 한 종목)
    legacy: bool = False                      # panic.py 재현용: 정지일 체결 허용 · 하한가 매도 미루기 없음


def run_month(m, cfg: Hybrid, s: int, k: int = 2, capital: float = E.CAPITAL, force_rebal=None, force_swap=None):
    """반환: (월 수익률, 정보 dict).

    force_rebal: 이 날들 시가에 평소 전략을 다시 정렬한다 (대조군 R0-REFRESH: 하이브리드가 실제로 재정렬한 날)
    force_swap:  이 날들 시가에 평소 보유 중 순위 가장 낮은 것이 상위 k 밖이면 1종목 교체 (트랙 A 대조군:
                 하이브리드가 종목 패닉 청산으로 빈 슬롯을 평소 종목으로 실제로 채운 날)
    info: mp_entries / sp_entries (신호일), rebal_days (패닉 청산 뒤 실제로 재정렬한 날), refill_days (종목 패닉 청산으로
          빈 슬롯을 평소 종목으로 실제로 채운 날), trades [(종류, 진입일, 청산일, 수익률)], halt_end (마지막 날 정지 보유 수),
          ret_alt (정지 보유를 정지 뒤 첫 거래 종가로 평가한 월 수익률)
    """
    force_rebal = force_rebal or ()
    force_swap = force_swap or ()
    I = m.ind
    M = I["mkt"]
    cnext = I.get("cnext")
    e = min(s + E.MONTH - 1, m.T - 1)
    cash = float(capital)
    slot = capital / k
    pos: dict = {}
    pend_open: set = set()
    info = {"mp_entries": [], "sp_entries": [], "halt_end": 0, "entries": 0, "rebal_days": [], "refill_days": [],
            "trades": []}
    next_rebal = s
    mp_active_mode = False
    post_mp = False          # 패닉 청산 뒤 재정렬이 예약됐고 아직 안 됨
    sp_freed = 0             # 종목 패닉 청산으로 비었고 아직 다른 것으로 안 채운 슬롯 수
    mp_freed = 0             # 분할 모드: 다른 패닉 포지션이 남은 채 한 슬롯이 청산돼 빈 슬롯 수
    stage = {}
    ep_count: dict = {}

    def traded(t, j):
        if cfg.legacy:
            return m.c[t, j] > 0
        return m.val[t, j] > 0 and m.c[t, j] > 0

    def raw_px(t, j, px):
        return px * m.raw[t, j] / m.c[t, j] if m.c[t, j] > 0 else px

    def slip_of(t, j, px):
        if j == cfg.idx_col:
            return max(0.0005, 2.5 / max(raw_px(t, j, px), 1))
        return E.slip(raw_px(t, j, px))

    def proceeds(j, t, px, extra=0.0):
        p = pos[j]
        tax = 0.0 if j == cfg.idx_col else E.TAX
        net = px * (1 - slip_of(t, j, px) - extra) * (1 - E.FEE - tax)
        return p["cost"] * net / p["ea"], net

    def sell(j, t, px, extra=0.0, at_open=False):
        nonlocal cash, sp_freed, mp_freed
        v, net = proceeds(j, t, px, extra)
        p = pos.pop(j)
        cash += v
        pend_open.discard(j)
        if p["kind"] != "base":
            info["trades"].append((p["kind"], p["t0"], t, net / p["ea"] - 1))
            if p["kind"] == "sp":
                sp_freed += 1
            elif any(q["kind"] == "mp" for q in pos.values()):
                mp_freed += 1

    def close_sellable(j, t):
        if not traded(t, j):
            return False
        lim = 0.15 if m.dates[t] < "20150615" else 0.30
        return cfg.legacy or not (m.c[t - 1, j] > 0 and m.c[t, j] <= m.c[t - 1, j] * (1 - lim + 0.005) and t < e)

    def try_sell_close(j, t):
        """종가 매도. 정지면 못 팔고, 하한가 마감이면 다음 날 시가로 미룬다."""
        if not traded(t, j):
            return False
        if not close_sellable(j, t):
            pend_open.add(j)
            return False
        sell(j, t, float(m.c[t, j]))
        return True

    def try_sell_open(j, t):
        if not traded(t, j) or not (m.o[t, j] > 0):
            return False
        sell(j, t, float(m.o[t, j]), at_open=True)
        return True

    def fill_qty(j, t, at, extra, budget_cash):
        """살 수 있으면 (수량, 체결가 raw, 조정가, 슬리피지). 못 사면 None."""
        if j in pos or not traded(t, j):
            return None
        px_adj = m.c[t, j] if at == "close" else m.o[t, j]
        cp = m.c[t - 1, j]
        if not (px_adj > 0) or not (cp > 0) or px_adj >= cp * 1.295:
            return None
        rp = raw_px(t, j, px_adj)
        sl = slip_of(t, j, px_adj) + extra
        px = rp * (1 + sl)
        budget = budget_cash if j == cfg.idx_col else min(budget_cash, slot)
        q = int(budget / (px * (1 + E.FEE)))
        return (q, px, px_adj, sl) if q >= 1 else None

    def buy(j, t, kind, ex=None, at="open", extra=0.0, anchor=-1):
        nonlocal cash
        if len(pos) >= k:
            return False
        f = fill_qty(j, t, at, extra, cash)
        if f is None:
            return False
        q, px, px_adj, sl = f
        cost = q * px * (1 + E.FEE)
        cash -= cost
        pos[j] = {"cost": cost, "ea": px_adj * (1 + sl) * (1 + E.FEE), "t0": t, "kind": kind, "exit": ex,
                  "close_entry": at == "close", "low0": float(m.l[t, j]), "anchor": anchor}
        info["entries"] += 1
        return True

    def rank_map(day):
        return {int(j): r for r, j in enumerate(cfg.base_top[day])}

    def worst_base(day, with_sp=False):
        """팔 순서: (with_sp 면 종목 패닉 포지션 먼저) 평소 보유 중 순위 낮은 것부터. 목록에 없으면 가장 낮음."""
        rk = rank_map(day)
        bh = sorted([x for x in pos if pos[x]["kind"] == "base"], key=lambda x: -rk.get(x, 10 ** 6))
        if with_sp:
            bh = [x for x in pos if pos[x]["kind"] == "sp"] + bh
        return bh

    def mp_open_entry(t, sig, n_slots):
        """시장 패닉 시가 진입: 평소 보유를 순위 낮은 것부터 팔아 n_slots 만들고 후보를 산다."""
        nonlocal sp_freed, post_mp
        rd = int(cfg.mp_rank_day[sig])
        anc = int(cfg.mp_anchor_day[sig]) if cfg.mp_anchor_day is not None else sig
        cands = [int(x) for x in cfg.mp_top[rd]]
        if not cands:
            return 0
        need = max(0, len(pos) + n_slots - k)
        for j in worst_base(sig, with_sp=True)[:need]:
            try_sell_open(j, t)
        got = 0
        for j in cands:
            if got >= n_slots:
                break
            if buy(j, t, "mp", cfg.mp_exit, "open", anchor=anc):
                got += 1
        if got:
            sp_freed = 0
            post_mp = False
        return got

    def sp_enter(t, dday, at, extra):
        """종목 패닉 REPLACE1: 후보를 순서대로 보며, 살 수 있는 첫 종목을 산다. 빈 슬롯이 없으면 평소 보유 중 순위 가장 낮은
        것을 같은 동시호가에 판다 — 팔 수 있고 그 돈으로 살 수 있을 때만."""
        nonlocal sp_freed
        n_sp = sum(1 for p in pos.values() if p["kind"] == "sp")
        rank_day = t if at == "close" else t - 1
        for j in [int(x) for x in cfg.sp_top[dday]]:
            if n_sp >= cfg.sp_max:
                break
            if j in pos:
                continue
            if len(pos) < k:
                if buy(j, t, "sp", cfg.sp_exit, at, extra=extra):
                    info["sp_entries"].append(dday)
                    n_sp += 1
                    sp_freed = max(0, sp_freed - 1)
                continue
            wb = worst_base(rank_day)
            if not wb:
                break
            w = wb[0]
            if at == "close":
                if not close_sellable(w, t):
                    break
                px_w = float(m.c[t, w])
            else:
                if not traded(t, w) or not (m.o[t, w] > 0):
                    break
                px_w = float(m.o[t, w])
            if fill_qty(j, t, at, extra, cash + proceeds(w, t, px_w)[0]) is None:
                continue
            sell(w, t, px_w, at_open=(at != "close"))
            if buy(j, t, "sp", cfg.sp_exit, at, extra=extra):
                info["sp_entries"].append(dday)
                n_sp += 1

    for t in range(s, e + 1):
        sig = t - 1
        # ① 시가: 예약 매도 · 대조군 강제 재정렬
        for j in list(pend_open):
            if j in pos:
                try_sell_open(j, t)
        if mp_active_mode and not any(p["kind"] == "mp" for p in pos.values()):
            next_rebal = t                   # 패닉 포지션이 시가에 정리됐으면 같은 시가에 재정렬
            post_mp = True
            mp_active_mode = False
            mp_freed = 0
        if t in force_rebal:
            next_rebal = min(next_rebal, t)
        if t in force_swap:
            keep = set(int(x) for x in cfg.base_top[sig][:k])
            wb = worst_base(sig)
            if wb and wb[0] not in keep:
                try_sell_open(wb[0], t)
        mp_held = any(p["kind"] == "mp" for p in pos.values())
        # ② 시장 패닉 (시가 진입). 분할 모드는 진행 중인 10일 창 안의 새 신호로 1단계를 다시 시작하지 않는다
        in_stage = cfg.mp_mode == "staged" and stage and sig <= stage["until"]
        if cfg.mp_sig is not None and cfg.mp_entry == "open" and cfg.mp_sig[sig] and not mp_held and not in_stage:
            epi = int(cfg.mp_episode[sig]) if cfg.mp_episode is not None else -1
            allowed = not cfg.mp_max_entries or ep_count.get(epi, 0) < cfg.mp_max_entries
            if allowed:
                n = 1 if cfg.mp_mode == "staged" else k
                if mp_open_entry(t, sig, n):
                    info["mp_entries"].append(sig)
                    ep_count[epi] = ep_count.get(epi, 0) + 1
                    mp_active_mode = True
                    if cfg.mp_mode == "staged":
                        stage = {"t0": sig, "until": sig + cfg.stage2_window, "done": False,
                                 "anchor": int(cfg.mp_anchor_day[sig]) if cfg.mp_anchor_day is not None else sig}
        # ②' staged 두 번째 슬롯 (전날 종가 신호 → 오늘 시가)
        if stage and not stage["done"] and stage.get("go") == t:
            if mp_open_entry(t, int(stage["sig2"]), 1):
                info["mp_entries"].append(int(stage["sig2"]))
                mp_active_mode = True
            stage["done"] = True
        mp_held = any(p["kind"] == "mp" for p in pos.values())
        # ③ 종목 패닉 (시가 진입: 다음 날 시가 / 당일 시가)
        if cfg.sp_top is not None and cfg.sp_entry in ("open", "open_same") and not mp_held:
            dday = sig if cfg.sp_entry == "open" else t
            sp_enter(t, dday, "open", cfg.sp_extra_slip if cfg.sp_entry == "open_same" else 0.0)
        # ④ 평소 전략 (빈 슬롯 수보다 많은 '청산으로 빈 슬롯' 기록은 이미 다른 진입이 쓴 것)
        sp_freed = min(sp_freed, k - len(pos))
        mp_freed = max(0, min(mp_freed, k - len(pos) - sp_freed))
        mp_held = any(p["kind"] == "mp" for p in pos.values())
        if not (mp_held and cfg.mp_mode == "allin"):
            if t >= next_rebal and not mp_held:
                keep = set(int(x) for x in cfg.base_top[sig][:k])
                for j in [x for x in pos if pos[x]["kind"] == "base" and x not in keep]:
                    try_sell_open(j, t)
                next_rebal = t + 21
                if post_mp:
                    info["rebal_days"].append(t)
                    post_mp = False
            for j in cfg.base_top[sig]:
                if len(pos) >= k:
                    break
                if buy(int(j), t, "base"):
                    if sp_freed > 0:
                        info["refill_days"].append(t)
                        sp_freed -= 1
                    elif mp_freed > 0:                 # 분할 모드: 한 슬롯만 끝나 그 자리를 새 순위로 채움
                        info["refill_days"].append(t)
                        mp_freed -= 1
        # ⑤ 장중·종가 청산
        for j in list(pos):
            if j not in pos:
                continue
            p = pos[j]
            if not traded(t, j):
                continue
            o_, h_, l_, c_ = m.o[t, j], m.h[t, j], m.l[t, j], m.c[t, j]
            if p["kind"] == "base":
                if cfg.surge_exit[t, j] and p["t0"] < t:
                    try_sell_close(j, t)
                continue
            ex: Exit = p["exit"]
            live = p["t0"] < t or not p["close_entry"]
            if live and ex.stop:
                slv = p["ea"] * (1 - ex.stop)
                if p["t0"] < t and o_ <= slv:
                    sell(j, t, float(o_), 0.002); continue
                if l_ <= slv:
                    sell(j, t, float(slv), 0.002); continue
            if live and ex.tp:
                tpv = p["ea"] * (1 + ex.tp)
                if p["t0"] < t and o_ >= tpv:
                    sell(j, t, float(o_)); continue
                if h_ >= tpv:
                    sell(j, t, float(tpv)); continue
            held = t - p["t0"] + (0 if p["close_entry"] else 1)
            after = p["t0"] < t
            reason = None
            if ex.surge and after and m.ind["ret1"][t, j] >= 0.10:
                reason = "surge"
            elif ex.rec and after and (ex.rec_arr[t, j] if ex.rec_arr is not None else c_ >= m.ind["ma5"][t, j]):
                reason = "rec"
            elif ex.fail_low and after and c_ < p["low0"]:
                reason = "fail"
            elif p["kind"] == "mp" and after and p["anchor"] >= 0 and (
                    (ex.mkt_tp and M[t] >= M[p["anchor"]] * (1 + ex.mkt_tp)) or
                    (ex.mkt_stop and M[t] <= M[p["anchor"]] * (1 - ex.mkt_stop))):
                reason = "mkt"
            elif held >= ex.hold:
                reason = "hold"
            if reason is None:
                continue
            if ((reason == "hold" and ex.at == "next_open") or ex.defer_all
                    or (ex.defer_rules and reason != "hold")) and t < e:
                pend_open.add(j)
            else:
                try_sell_close(j, t)
        mp_held = any(p["kind"] == "mp" for p in pos.values())
        if mp_active_mode and not mp_held:
            next_rebal = t + 1               # 하이브리드는 패닉 포지션 정리 다음 날 평소 전략 재정렬
            post_mp = True
            mp_freed = 0
        mp_active_mode = mp_held
        # staged 두 번째 슬롯 신호 (오늘 종가 기준 → 내일 시가)
        if stage and not stage["done"] and stage["t0"] < t <= stage["until"] and "go" not in stage:
            c1 = cfg.stage2_sig is not None and cfg.stage2_sig[t]
            c2 = cfg.stage2_drop and M[t] <= M[stage["anchor"]] * (1 - cfg.stage2_drop)
            if c1 or c2:
                stage["go"] = t + 1
                stage["sig2"] = t
        # ⑥ 종가 진입
        if t < e and cfg.mp_sig is not None and cfg.mp_entry == "close" and cfg.mp_sig[t] and not mp_held:
            rd = int(cfg.mp_rank_day[t])
            anc = int(cfg.mp_anchor_day[t]) if cfg.mp_anchor_day is not None else t
            cands = [int(x) for x in cfg.mp_top[rd]]
            if cands:
                for j in worst_base(t, with_sp=True):
                    try_sell_close(j, t)
                got = 0
                for j in cands:
                    if got >= k:
                        break
                    if buy(j, t, "mp", cfg.mp_exit, "close", anchor=anc):
                        got += 1
                if got:
                    info["mp_entries"].append(t)
                    mp_active_mode = True
                    sp_freed = 0
                    post_mp = False
        mp_held = any(p["kind"] == "mp" for p in pos.values())
        if t < e and cfg.sp_top is not None and cfg.sp_entry == "close" and not mp_held:
            sp_enter(t, t, "close", 0.0)
    # ⑦ 마지막 날 종가 평가 (정지 종목은 마지막 체결가; ret_alt 는 정지 뒤 첫 거래 종가)
    alt_diff = 0.0
    for j in list(pos):
        c_ = m.c[e, j]
        halted = m.val[e, j] <= 0
        if halted:
            info["halt_end"] += 1
        if not (c_ > 0):
            back = np.where(m.c[:e + 1, j] > 0)[0]
            c_ = m.c[back[-1], j] if len(back) else pos[j]["ea"]
        if halted and cnext is not None and np.isfinite(cnext[e, j]) and cnext[e, j] > 0:
            v0 = proceeds(j, e, float(c_))[0]
            v1 = proceeds(j, e, float(cnext[e, j]))[0]
            alt_diff += v1 - v0
        sell(j, e, float(c_))
    info["ret_alt"] = (cash + alt_diff) / capital - 1
    return cash / capital - 1, info
