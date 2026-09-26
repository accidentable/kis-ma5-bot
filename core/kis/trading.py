"""
core/kis/trading.py — 주문 / 계좌

2025년 개정된 TR_ID 를 쓴다. 구버전(TTTC0802U 등)은 폐기되어 거부된다.
  매수      TTTC0012U   (구 TTTC0802U)
  매도      TTTC0011U   (구 TTTC0801U)
  정정/취소 TTTC0013U   (구 TTTC0803U)
  미체결    TTTC0084R   (구 TTTC8036R)
주문 body 에 EXCG_ID_DVSN_CD(거래소ID구분코드)가 필수로 추가됐다.
"""
from __future__ import annotations

import logging
from datetime import date

import config
from core.kis import client
from core.kis.tick import round_to_tick

logger = logging.getLogger(__name__)

TR_BUY = "TTTC0012U"
TR_SELL = "TTTC0011U"
TR_REVISE_CANCEL = "TTTC0013U"
TR_BALANCE = "TTTC8434R"
TR_PENDING = "TTTC0084R"
TR_BUYABLE = "TTTC8908R"
TR_DAILY_CCLD = "TTTC0081R"

_PATH_ORDER = "/uapi/domestic-stock/v1/trading/order-cash"
_PATH_RVSECNCL = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
_PATH_BALANCE = "/uapi/domestic-stock/v1/trading/inquire-balance"
_PATH_PENDING = "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
_PATH_BUYABLE = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
_PATH_DAILY_CCLD = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"


def _f(v, default: float = 0.0) -> float:
    try:
        s = str(v).strip()
        return float(s) if s else default
    except (TypeError, ValueError):
        return default


def _i(v, default: int = 0) -> int:
    return int(_f(v, default))


def _acct() -> dict[str, str]:
    return {"CANO": config.CANO, "ACNT_PRDT_CD": config.ACNT_PRDT_CD}


# ══════════════════════════════════════════════════════════════
# 주문
# ══════════════════════════════════════════════════════════════
def place_order(ticker: str, qty: int, price: int, side: str, *, market: bool = False) -> dict:
    """
    현금 주문. side 는 'buy' | 'sell'.
    price 는 호출 전에 호가단위로 보정해 넘기는 것을 전제하지만 여기서도 한 번 더 맞춘다.
    """
    if qty <= 0:
        return {
            "success": False, "order_no": "", "org_no": "", "message": "수량 0",
            "ticker": ticker, "qty": qty, "price": price, "side": side,
        }

    price = 0 if market else round_to_tick(price, "nearest")
    body = {
        **_acct(),
        "PDNO": ticker,
        "ORD_DVSN": config.ORD_DVSN_MARKET if market else config.ORD_DVSN_LIMIT,
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(price)),
        "EXCG_ID_DVSN_CD": config.EXCG_ID_DVSN_CD,
        "SLL_TYPE": "01" if side == "sell" else "",
        "CNDT_PRIC": "",
    }

    label = "매수" if side == "buy" else "매도"
    kind = "시장가" if market else f"지정가 {price:,}원"

    if config.DRY_RUN:
        logger.info("[DRY_RUN] %s %s %d주 %s — 주문 안 보냄", label, ticker, qty, kind)
        return {
            "success": True, "order_no": "DRYRUN", "org_no": "", "message": "DRY_RUN",
            "ticker": ticker, "qty": qty, "price": price, "side": side, "dry_run": True,
        }

    logger.info("%s 주문 전송: %s %d주 %s", label, ticker, qty, kind)
    data = client.post(
        _PATH_ORDER,
        TR_BUY if side == "buy" else TR_SELL,
        body,
        use_hashkey=True,
        is_order=True,
    )
    out = data.get("output", {}) or {}
    result = {
        "success": True,
        "order_no": str(out.get("ODNO", "")).strip(),
        "org_no": str(out.get("KRX_FWDG_ORD_ORGNO", "")).strip(),
        "message": str(data.get("msg1", "")).strip(),
        "ticker": ticker, "qty": qty, "price": price, "side": side,
        "dry_run": False,
    }
    logger.info("%s 주문 접수: 주문번호 %s", label, result["order_no"])
    return result


def buy(ticker: str, qty: int, price: int, *, market: bool = False) -> dict:
    return place_order(ticker, qty, price, "buy", market=market)


def sell(ticker: str, qty: int, price: int, *, market: bool = False) -> dict:
    return place_order(ticker, qty, price, "sell", market=market)


def cancel_order(order_no: str, org_no: str, qty: int, *, order_dvsn: str = config.ORD_DVSN_LIMIT) -> dict:
    """미체결 주문 전량 취소."""
    if config.DRY_RUN:
        logger.info("[DRY_RUN] 주문 취소 %s — 전송 안 함", order_no)
        return {"success": True, "message": "DRY_RUN", "order_no": order_no}

    body = {
        **_acct(),
        "KRX_FWDG_ORD_ORGNO": org_no,
        "ORGN_ODNO": order_no,
        "ORD_DVSN": order_dvsn,
        "RVSE_CNCL_DVSN_CD": "02",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y",
        "EXCG_ID_DVSN_CD": config.EXCG_ID_DVSN_CD,
    }
    data = client.post(_PATH_RVSECNCL, TR_REVISE_CANCEL, body, use_hashkey=True, is_order=True)
    return {
        "success": True,
        "order_no": str((data.get("output") or {}).get("ODNO", "")).strip() or order_no,
        "message": str(data.get("msg1", "")).strip(),
    }


# ══════════════════════════════════════════════════════════════
# 계좌 조회
# ══════════════════════════════════════════════════════════════
def get_balance() -> dict:
    """주식 잔고. holdings[] / cash(예수금) / d2_cash / total_eval / net_asset"""
    params = {
        **_acct(),
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    data = client.get(_PATH_BALANCE, TR_BALANCE, params)

    holdings = []
    for row in data.get("output1") or []:
        qty = _i(row.get("hldg_qty"))
        if qty <= 0:
            continue
        holdings.append({
            "ticker": str(row.get("pdno", "")).strip(),
            "name": str(row.get("prdt_name", "")).strip(),
            "qty": qty,
            "sellable_qty": _i(row.get("ord_psbl_qty"), qty),
            "avg_price": _f(row.get("pchs_avg_pric")),
            "price": _f(row.get("prpr")),
            "eval_amount": _f(row.get("evlu_amt")),
            "pnl_amount": _f(row.get("evlu_pfls_amt")),
            "pnl_pct": _f(row.get("evlu_pfls_rt")),
        })

    o2 = data.get("output2") or [{}]
    o2 = o2[0] if isinstance(o2, list) else o2

    return {
        "holdings": holdings,
        "cash": _f(o2.get("dnca_tot_amt")),
        "d2_cash": _f(o2.get("prvs_rcdl_excc_amt")),
        "total_eval": _f(o2.get("tot_evlu_amt")),
        "net_asset": _f(o2.get("nass_amt")),
        "pnl_amount": _f(o2.get("evlu_pfls_smtl_amt")),
    }


def get_buyable(ticker: str, price: int) -> dict:
    """
    해당 종목/가격 기준 매수 가능 수량.
    미수(신용)를 쓰지 않으려고 '미수없는매수수량'(nrcvb_buy_qty)을 기준으로 삼는다.
    """
    params = {
        **_acct(),
        "PDNO": ticker,
        "ORD_UNPR": str(int(price)),
        "ORD_DVSN": config.ORD_DVSN_LIMIT,
        "CMA_EVLU_AMT_ICLD_YN": "N",
        "OVRS_ICLD_YN": "N",
    }
    data = client.get(_PATH_BUYABLE, TR_BUYABLE, params)
    o = data.get("output", {}) or {}
    return {
        "cash": _f(o.get("ord_psbl_cash")),
        "qty_no_margin": _i(o.get("nrcvb_buy_qty")),
        "amount_no_margin": _f(o.get("nrcvb_buy_amt")),
        "qty_max": _i(o.get("max_buy_qty")),
        "amount_max": _f(o.get("max_buy_amt")),
    }


def get_pending_orders() -> list[dict]:
    """정정·취소 가능한 미체결 주문."""
    params = {
        **_acct(),
        "INQR_DVSN_1": "0",
        "INQR_DVSN_2": "0",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    rows = client.paginate(_PATH_PENDING, TR_PENDING, params)

    orders = []
    for r in rows:
        remain = _i(r.get("psbl_qty"))
        if remain <= 0:
            continue
        orders.append({
            "order_no": str(r.get("odno", "")).strip(),
            "org_no": str(r.get("ord_gno_brno", "")).strip(),
            "ticker": str(r.get("pdno", "")).strip(),
            "name": str(r.get("prdt_name", "")).strip(),
            "side": "buy" if str(r.get("sll_buy_dvsn_cd", "")).strip() == "02" else "sell",
            "qty": _i(r.get("ord_qty")),
            "filled_qty": _i(r.get("tot_ccld_qty")),
            "remain_qty": remain,
            "price": _i(r.get("ord_unpr")),
            "order_time": str(r.get("ord_tmd", "")).strip(),
            "order_dvsn": str(r.get("ord_dvsn_cd", "")).strip() or config.ORD_DVSN_LIMIT,
        })
    return orders


def get_today_fills(target: date | None = None) -> list[dict]:
    """당일 체결 내역."""
    d = (target or date.today()).strftime("%Y%m%d")
    params = {
        **_acct(),
        "INQR_STRT_DT": d,
        "INQR_END_DT": d,
        "SLL_BUY_DVSN_CD": "00",
        "PDNO": "",
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "CCLD_DVSN": "01",
        "INQR_DVSN": "00",
        "INQR_DVSN_1": "",
        "INQR_DVSN_3": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    rows = client.paginate(_PATH_DAILY_CCLD, TR_DAILY_CCLD, params, output_key="output1")

    fills = []
    for r in rows:
        filled = _i(r.get("tot_ccld_qty"))
        if filled <= 0:
            continue
        fills.append({
            "order_no": str(r.get("odno", "")).strip(),
            "ticker": str(r.get("pdno", "")).strip(),
            "name": str(r.get("prdt_name", "")).strip(),
            "side": "buy" if str(r.get("sll_buy_dvsn_cd", "")).strip() == "02" else "sell",
            "qty": filled,
            "avg_price": _f(r.get("avg_prvs")),
            "amount": _f(r.get("tot_ccld_amt")),
            "time": str(r.get("ord_tmd", "")).strip(),
        })
    return fills
