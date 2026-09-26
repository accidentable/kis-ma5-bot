"""
core/kis/quotes.py — 시세 조회

  get_price          현재가 (FHKST01010100)
  get_daily_candles  일봉 OHLCV (FHKST03010100) — 수정주가 기준
  is_open_day        국내 개장일 여부 (CTCA0903R)
  recent_open_days   최근 개장일 목록
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from core.kis import client

logger = logging.getLogger(__name__)

TR_PRICE = "FHKST01010100"
TR_DAILY_CHART = "FHKST03010100"
TR_HOLIDAY = "CTCA0903R"

_PATH_PRICE = "/uapi/domestic-stock/v1/quotations/inquire-price"
_PATH_DAILY = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
_PATH_HOLIDAY = "/uapi/domestic-stock/v1/quotations/chk-holiday"


def _f(v, default: float = 0.0) -> float:
    try:
        s = str(v).strip()
        return float(s) if s else default
    except (TypeError, ValueError):
        return default


def _i(v, default: int = 0) -> int:
    return int(_f(v, default))


def get_price(ticker: str, market: str = "J") -> dict:
    """
    현재가 조회.
    market: J=KRX, NX=NXT, UN=KRX+NXT 통합. 프리마켓(08:00~08:50)엔 NX/UN 이라야 가격이 나온다.
    반환: ticker, name, price, prev_close, change_pct, volume, value,
          is_halted(거래정지), is_watch(관리종목), upper_limit, lower_limit
    """
    data = client.get(
        _PATH_PRICE, TR_PRICE,
        {"FID_COND_MRKT_DIV_CODE": market, "FID_INPUT_ISCD": ticker},
    )
    o = data.get("output", {}) or {}
    return {
        "ticker": ticker,
        "market": market,
        "name": str(o.get("rprs_mrkt_kor_name", "")).strip(),
        "price": _f(o.get("stck_prpr")),
        "prev_close": _f(o.get("stck_sdpr")),
        "open": _f(o.get("stck_oprc")),
        "high": _f(o.get("stck_hgpr")),
        "low": _f(o.get("stck_lwpr")),
        "change_pct": _f(o.get("prdy_ctrt")),
        "volume": _i(o.get("acml_vol")),
        "value": _f(o.get("acml_tr_pbmn")),
        "upper_limit": _f(o.get("stck_mxpr")),
        "lower_limit": _f(o.get("stck_llam")),
        # iscd_stat_cls_code: 00 그외 / 51 관리종목 / 52 투자위험 / 53 투자경고 / 54 투자주의 / 57 증거금100% / 58 거래정지 / 59 단기과열
        "status_code": str(o.get("iscd_stat_cls_code", "")).strip(),
        "is_halted": str(o.get("iscd_stat_cls_code", "")).strip() == "58",
        "is_watch": str(o.get("iscd_stat_cls_code", "")).strip() in ("51", "52", "53", "59"),
    }


def get_premarket_price(ticker: str) -> float | None:
    """
    프리마켓 가격. config.PREMARKET_MARKET_CODES 순서로 시도해 0 이 아닌 첫 값을 돌려준다.
    전부 실패하면 None — KRX 코드(J)로는 본장 전에 전일 종가만 나오므로 대체하지 않는다.
    """
    import config
    for code in config.PREMARKET_MARKET_CODES:
        try:
            q = get_price(ticker, market=code)
            if q["price"] > 0:
                return q["price"]
            logger.debug("%s 프리마켓 가격 0 (%s)", ticker, code)
        except client.KisError as e:
            logger.debug("%s 프리마켓 조회 실패 (%s): %s", ticker, code, e)
        except Exception as e:
            logger.warning("%s 프리마켓 조회 오류 (%s): %s", ticker, code, e)
    return None


def get_daily_candles(ticker: str, days: int = 40) -> list[dict]:
    """
    일봉 OHLCV. 최신순이 아니라 **날짜 오름차순**으로 정렬해 반환한다.
    수정주가(FID_ORG_ADJ_PRC=0) 기준이라 액면분할이 있어도 이동평균이 깨지지 않는다.
    """
    end = date.today()
    # 주말·공휴일을 감안해 넉넉히 요청 (영업일 ≈ 달력일 × 0.68)
    start = end - timedelta(days=int(days * 1.8) + 15)

    data = client.get(
        _PATH_DAILY, TR_DAILY_CHART,
        {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        },
    )

    rows = data.get("output2") or []
    candles = []
    for r in rows:
        d = str(r.get("stck_bsop_date", "")).strip()
        close = _f(r.get("stck_clpr"))
        if not d or close <= 0:
            continue
        candles.append({
            "date": d,
            "open": _f(r.get("stck_oprc")),
            "high": _f(r.get("stck_hgpr")),
            "low": _f(r.get("stck_lwpr")),
            "close": close,
            "volume": _i(r.get("acml_vol")),
            "value": _f(r.get("acml_tr_pbmn")),
        })

    candles.sort(key=lambda c: c["date"])
    return candles[-days:] if days else candles


def _holiday_rows(base: date) -> list[dict]:
    return client.paginate(
        _PATH_HOLIDAY, TR_HOLIDAY,
        {"BASS_DT": base.strftime("%Y%m%d"), "CTX_AREA_FK": "", "CTX_AREA_NK": ""},
        output_key="output",
        max_pages=3,
    )


def is_open_day(d: date | None = None) -> bool:
    """해당 일자가 국내 증시 개장일인지."""
    d = d or date.today()
    try:
        for row in _holiday_rows(d):
            if str(row.get("bass_dt", "")).strip() == d.strftime("%Y%m%d"):
                return str(row.get("opnd_yn", "")).strip().upper() == "Y"
    except Exception as e:
        logger.warning("휴장일 조회 실패(%s) — 주말 여부로 대체 판단", e)
    return d.weekday() < 5


def recent_open_days(upto: date | None = None, count: int = 10) -> list[date]:
    """upto(포함) 이전의 개장일을 최신순으로 count 개 반환."""
    upto = upto or date.today()
    try:
        rows = _holiday_rows(upto - timedelta(days=count * 2 + 20))
        opened = sorted(
            (
                datetime.strptime(str(r["bass_dt"]).strip(), "%Y%m%d").date()
                for r in rows
                if str(r.get("opnd_yn", "")).strip().upper() == "Y" and str(r.get("bass_dt", "")).strip()
            ),
            reverse=True,
        )
        return [d for d in opened if d <= upto][:count]
    except Exception as e:
        logger.warning("개장일 목록 조회 실패(%s) — 주말 제외로 대체", e)
        out, cur = [], upto
        while len(out) < count:
            if cur.weekday() < 5:
                out.append(cur)
            cur -= timedelta(days=1)
        return out


def trading_days_between(start: date, end: date) -> int:
    """start 이후 end 까지 경과한 거래일 수 (start 당일은 0)."""
    if end <= start:
        return 0
    days = recent_open_days(end, count=40)
    return sum(1 for d in days if start < d <= end)
