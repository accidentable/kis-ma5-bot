"""
core/kis/tick.py — 호가가격단위(tick size) 보정

2023-01-25 한국거래소 개편으로 코스피·코스닥 호가단위가 통일됐다.
지정가가 호가단위에 맞지 않으면 주문이 거부되므로 주문 직전 반드시 통과시킨다.
"""
from __future__ import annotations

# (상한가 미만, 호가단위) — 가격이 작은 구간부터
_TIERS: tuple[tuple[float, int], ...] = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
)
_TOP_TICK = 1_000  # 500,000원 이상


def tick_size(price: float) -> int:
    """해당 가격대의 호가단위."""
    for upper, tick in _TIERS:
        if price < upper:
            return tick
    return _TOP_TICK


def round_to_tick(price: float, mode: str = "nearest") -> int:
    """
    가격을 호가단위에 맞춘다.

    mode:
      nearest  가장 가까운 호가
      down     아래쪽 호가 (매도 지정가를 낮출 때)
      up       위쪽 호가 (매수 지정가를 올릴 때)

    구간 경계에서 반올림 방향에 따라 호가단위가 바뀔 수 있으므로,
    보정 후 그 가격대의 호가단위로 한 번 더 검증한다.
    """
    if price <= 0:
        return 0

    def _apply(p: float) -> int:
        t = tick_size(p)
        if mode == "down":
            return int(p // t) * t
        if mode == "up":
            return int(-(-p // t)) * t
        return int(round(p / t)) * t

    result = _apply(price)
    # 경계를 넘어가며 호가단위가 달라진 경우 재보정 (예: 19,995 → 20,000)
    if tick_size(result) != tick_size(price):
        t = tick_size(result)
        if mode == "down":
            result = int(result // t) * t
        elif mode == "up":
            result = int(-(-result // t)) * t
        else:
            result = int(round(result / t)) * t
    return int(result)


def offset_ticks(price: float, ticks: int) -> int:
    """
    현재가에서 N틱 떨어진 호가를 구한다. ticks 가 양수면 위, 음수면 아래.
    구간을 넘어가면 호가단위가 바뀌므로 한 틱씩 이동한다.
    """
    p = round_to_tick(price, "nearest")
    step = 1 if ticks >= 0 else -1
    for _ in range(abs(ticks)):
        t = tick_size(p if step > 0 else p - 1)
        p += step * t
        if p <= 0:
            return tick_size(1)
    return int(p)
