"""
tests/test_mock.py — 모의투자(KIS_ENV=mock) 점검. 가짜 HTTP 응답으로 돌아서 API 키·네트워크 없이 된다.

    python tests/test_mock.py

확인하는 것: 모의 주소·키·계좌, 모의 TR_ID(V…), 토큰 캐시·상태 파일 분리, 모의 미지원 API 대체
(미체결 → 일별주문체결조회, 휴장일 → 일봉), cli.py check 의 환경 표시, 실전 모드로 되돌렸을 때 TR_ID.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import tempfile
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.update({
    "KIS_ENV": "mock",
    "KIS_APP_KEY": "REAL_KEY", "KIS_APP_SECRET": "REAL_SECRET", "KIS_ACCOUNT_NO": "1111111101",
    "KIS_MOCK_APP_KEY": "MOCK_KEY", "KIS_MOCK_APP_SECRET": "MOCK_SECRET", "KIS_MOCK_ACCOUNT_NO": "50123456-01",
    "TOKEN_CACHE": "file", "STATE_BACKEND": "local", "DRY_RUN": "false",
})
os.environ.pop("KIS_RATE_LIMIT_PER_SEC", None)

import config  # noqa: E402

config.DATA_DIR = tempfile.mkdtemp(prefix="mock-test-")

from core import state                         # noqa: E402
from core.kis import auth, client, quotes, trading  # noqa: E402

PASS, FAIL = [], []
CALLS: list[dict] = []
VTS = "https://openapivts.koreainvestment.com:29443"


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


class FakeResp:
    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status_code = status
        self.headers = {"tr_cont": data.pop("_tr_cont", "D")}
        self.text = str(data)

    def json(self):
        return dict(self._data)

    def raise_for_status(self):
        pass


TODAY = date.today().strftime("%Y%m%d")
# 미체결 대체 조회용 당일 주문 내역 (일별주문체결조회 output1 모양)
CCLD_ROWS = [
    # 그대로 남은 매수 주문 → 미체결
    {"odno": "0000000101", "orgn_odno": "", "ord_gno_brno": "00950", "pdno": "005930", "prdt_name": "삼성전자",
     "sll_buy_dvsn_cd": "02", "ord_qty": "10", "tot_ccld_qty": "3", "rmn_qty": "7", "ord_unpr": "69000",
     "ord_tmd": "090500", "ord_dvsn_cd": "00", "cncl_yn": "N", "ord_dvsn_name": "지정가"},
    # 다 체결 → 빠짐
    {"odno": "0000000102", "orgn_odno": "", "ord_gno_brno": "00950", "pdno": "000660", "prdt_name": "SK하이닉스",
     "sll_buy_dvsn_cd": "01", "ord_qty": "5", "tot_ccld_qty": "5", "rmn_qty": "0", "ord_unpr": "200000",
     "ord_tmd": "090600", "ord_dvsn_cd": "00", "cncl_yn": "N", "ord_dvsn_name": "지정가"},
    # 취소된 원주문 + 취소 주문 행 → 둘 다 빠짐
    {"odno": "0000000103", "orgn_odno": "", "ord_gno_brno": "00950", "pdno": "035420", "prdt_name": "NAVER",
     "sll_buy_dvsn_cd": "02", "ord_qty": "4", "tot_ccld_qty": "0", "rmn_qty": "4", "ord_unpr": "180000",
     "ord_tmd": "090700", "ord_dvsn_cd": "00", "cncl_yn": "N", "ord_dvsn_name": "지정가"},
    {"odno": "0000000104", "orgn_odno": "0000000103", "ord_gno_brno": "00950", "pdno": "035420", "prdt_name": "NAVER",
     "sll_buy_dvsn_cd": "02", "ord_qty": "4", "tot_ccld_qty": "0", "rmn_qty": "4", "ord_unpr": "0",
     "ord_tmd": "091000", "ord_dvsn_cd": "00", "cncl_yn": "Y", "ord_dvsn_name": "지정가취소"},
    # 정정: 원주문은 빠지고 정정 주문이 미체결로 남음
    {"odno": "0000000105", "orgn_odno": "", "ord_gno_brno": "00950", "pdno": "051910", "prdt_name": "LG화학",
     "sll_buy_dvsn_cd": "01", "ord_qty": "2", "tot_ccld_qty": "0", "rmn_qty": "2", "ord_unpr": "300000",
     "ord_tmd": "090800", "ord_dvsn_cd": "00", "cncl_yn": "N", "ord_dvsn_name": "지정가"},
    {"odno": "0000000106", "orgn_odno": "0000000105", "ord_gno_brno": "00950", "pdno": "051910", "prdt_name": "LG화학",
     "sll_buy_dvsn_cd": "01", "ord_qty": "2", "tot_ccld_qty": "0", "rmn_qty": "2", "ord_unpr": "295000",
     "ord_tmd": "091100", "ord_dvsn_cd": "00", "cncl_yn": "N", "ord_dvsn_name": "지정가정정"},
]


def _candles() -> list[dict]:
    """삼성전자 일봉: 최근 15일 중 평일만 (오늘 제외 — 장 시작 전 가정)."""
    out = []
    d = date.today() - timedelta(days=1)
    while len(out) < 10:
        if d.weekday() < 5:
            out.append({"stck_bsop_date": d.strftime("%Y%m%d"), "stck_clpr": "70000", "stck_oprc": "70000",
                        "stck_hgpr": "70000", "stck_lwpr": "70000", "acml_vol": "1", "acml_tr_pbmn": "1"})
        d -= timedelta(days=1)
    return out


def route(tr_id: str, params: dict | None, body: dict | None) -> dict:
    ok = {"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "정상처리 되었습니다."}
    if tr_id in ("VTTC0012U", "VTTC0011U", "VTTC0013U"):
        return {**ok, "output": {"ODNO": "0000009999", "KRX_FWDG_ORD_ORGNO": "00950"}}
    if tr_id == "VTTC8434R":
        return {**ok, "output1": [{"pdno": "005930", "prdt_name": "삼성전자", "hldg_qty": "10", "ord_psbl_qty": "10",
                                   "pchs_avg_pric": "65000", "prpr": "70000", "evlu_amt": "700000",
                                   "evlu_pfls_amt": "50000", "evlu_pfls_rt": "7.69"}],
                "output2": [{"dnca_tot_amt": "99000000", "prvs_rcdl_excc_amt": "99000000",
                             "tot_evlu_amt": "99700000", "nass_amt": "99700000", "evlu_pfls_smtl_amt": "50000"}]}
    if tr_id == "VTTC8908R":
        return {**ok, "output": {"ord_psbl_cash": "99000000", "nrcvb_buy_qty": "1414", "nrcvb_buy_amt": "98980000",
                                 "max_buy_qty": "1414", "max_buy_amt": "98980000"}}
    if tr_id == "VTTC0081R":
        rows = CCLD_ROWS if params and params.get("CCLD_DVSN") == "00" else []
        return {**ok, "output1": [dict(r) for r in rows], "output2": {}, "ctx_area_fk100": "", "ctx_area_nk100": ""}
    if tr_id == "FHKST01010100":
        return {**ok, "output": {"stck_prpr": "70000", "stck_sdpr": "69000", "prdy_ctrt": "1.45",
                                 "iscd_stat_cls_code": "00"}}
    if tr_id == "FHKST03010100":
        return {**ok, "output1": {}, "output2": _candles()}
    return {"rt_cd": "1", "msg_cd": "TEST404", "msg1": f"테스트에 없는 TR {tr_id}"}


class FakeSession:
    def _do(self, method, url, headers, params=None, json=None):
        CALLS.append({"method": method, "url": url, "tr_id": headers.get("tr_id"), "appkey": headers.get("appkey"),
                      "params": params, "body": json, "hashkey": headers.get("hashkey")})
        return FakeResp(route(headers.get("tr_id", ""), params, json))

    def get(self, url, headers=None, params=None, timeout=None):
        return self._do("GET", url, headers, params=params)

    def post(self, url, headers=None, json=None, timeout=None):
        return self._do("POST", url, headers, json=json)


AUTH_POSTS: list[dict] = []


def fake_auth_post(url, json=None, headers=None, timeout=None):
    AUTH_POSTS.append({"url": url, "body": json})
    if url.endswith("/oauth2/tokenP"):
        return FakeResp({"access_token": "MOCKTOKEN123456", "expires_in": 86400})
    if url.endswith("/uapi/hashkey"):
        return FakeResp({"HASH": "H" * 16})
    return FakeResp({}, status=404)


client._session = FakeSession()
auth.requests.post = fake_auth_post


def last(tr_id: str) -> dict:
    return [c for c in CALLS if c["tr_id"] == tr_id][-1]


def main() -> int:
    print("── 설정 ─────────────────────────────")
    check("KIS_ENV=mock → IS_MOCK", config.IS_MOCK and config.KIS_ENV == "mock")
    check("모의 주소", config.KIS_BASE_URL == VTS, config.KIS_BASE_URL)
    check("모의 키 사용", config.KIS_APP_KEY == "MOCK_KEY" and config.KIS_APP_SECRET == "MOCK_SECRET")
    check("모의 계좌 분해 (하이픈 제거)", (config.CANO, config.ACNT_PRDT_CD) == ("50123456", "01"))
    check("모의 유량 기본값 1.5/초", config.KIS_RATE_LIMIT_PER_SEC == 1.5, str(config.KIS_RATE_LIMIT_PER_SEC))
    check("SSM 토큰 경로 분리", config.SSM_TOKEN_PATH == "/ma5-bot/kis/token-mock")
    check("설정 검증 통과", config.validate() == [], str(config.validate()))
    check("요약에 모의투자 표시", "모의투자" in config.summary())

    print("── 토큰 ─────────────────────────────")
    tok = auth.get_token()
    check("토큰 발급 → 모의 주소", tok == "MOCKTOKEN123456" and AUTH_POSTS[-1]["url"] == f"{VTS}/oauth2/tokenP")
    check("토큰 발급에 모의 키", AUTH_POSTS[-1]["body"]["appkey"] == "MOCK_KEY")
    check("토큰 캐시 파일 분리", os.path.exists(os.path.join(config.DATA_DIR, "kis_token_mock.json"))
          and not os.path.exists(os.path.join(config.DATA_DIR, "kis_token.json")))

    print("── 주문 · 계좌 TR_ID ─────────────────")
    bal = trading.get_balance()
    c = last("VTTC8434R")
    check("잔고 VTTC8434R + 모의 주소", c["url"].startswith(VTS) and c["params"]["CANO"] == "50123456")
    check("잔고 파싱", bal["holdings"][0]["qty"] == 10 and bal["net_asset"] == 99_700_000)
    check("요청 헤더에 모의 키", c["appkey"] == "MOCK_KEY")

    b = trading.get_buyable("005930", 70000)
    check("매수가능 VTTC8908R", last("VTTC8908R") and b["qty_no_margin"] == 1414)

    r = trading.buy("005930", 3, 70050)
    c = last("VTTC0012U")
    check("매수 VTTC0012U", r["success"] and r["order_no"] == "0000009999" and c["body"]["ORD_QTY"] == "3")
    check("주문 hashkey 모의 주소", c["hashkey"] and AUTH_POSTS[-1]["url"] == f"{VTS}/uapi/hashkey")
    trading.sell("005930", 3, 70000)
    check("매도 VTTC0011U", last("VTTC0011U")["body"]["SLL_TYPE"] == "01")
    trading.cancel_order("0000000101", "00950", 7)
    check("취소 VTTC0013U", last("VTTC0013U")["body"]["ORGN_ODNO"] == "0000000101")

    fills = trading.get_today_fills()
    check("체결 조회 VTTC0081R (CCLD_DVSN=01)", last("VTTC0081R")["params"]["CCLD_DVSN"] == "01" and fills == [])
    check("실전 TR_ID 가 나간 적 없음", not [x for x in CALLS if str(x["tr_id"]).startswith("TTTC")])

    print("── 모의 미지원 API 대체 ─────────────────")
    try:
        trading.tr(trading.TR_PENDING)
        check("미지원 TR 은 막힘", False, "예외 없음")
    except NotImplementedError as e:
        check("미지원 TR 은 막힘", "모의투자" in str(e), str(e))

    pend = trading.get_pending_orders()
    c = last("VTTC0081R")
    nos = sorted(o["order_no"] for o in pend)
    check("미체결 → 일별주문체결조회 전체(CCLD_DVSN=00)", c["params"]["CCLD_DVSN"] == "00")
    check("미체결: 남은 주문 + 정정 주문만", nos == ["0000000101", "0000000106"], str(nos))
    o = {x["order_no"]: x for x in pend}["0000000101"]
    check("미체결 필드", (o["side"], o["remain_qty"], o["filled_qty"], o["org_no"], o["price"])
          == ("buy", 7, 3, "00950", 69000), str(o))
    check("정정 주문은 정정가", {x["order_no"]: x for x in pend}["0000000106"]["price"] == 295000)

    n_before = len(CALLS)
    today = date.today()
    opened = quotes.is_open_day(today)
    check("오늘 개장 = 평일 여부 (API 안 부름)", opened == (today.weekday() < 5) and len(CALLS) == n_before)
    days = quotes.recent_open_days(today, count=5)
    check("휴장일 조회(CTCA0903R) 안 부름", not [x for x in CALLS if x["tr_id"] == "CTCA0903R"])
    check("지난 개장일 = 일봉 날짜", len(days) == 5 and all(d.weekday() < 5 for d in days), str(days))
    check("개장일 판정에 쓴 일봉도 모의 주소", last("FHKST03010100")["url"].startswith(VTS))
    sat = today - timedelta(days=(today.weekday() - 5) % 7 or 7)
    check("지난 토요일은 휴장", quotes.is_open_day(sat) is False)
    past_weekday = days[-1]
    check("지난 개장일은 개장", quotes.is_open_day(past_weekday) is True)
    check("거래일 수 계산", quotes.trading_days_between(days[2], days[0]) == 2)

    print("── 상태 파일 분리 ───────────────────")
    state.save(state.load())
    check("상태 파일 state_mock.json", os.path.exists(os.path.join(config.DATA_DIR, "state_mock.json"))
          and not os.path.exists(os.path.join(config.DATA_DIR, "state.json")))

    print("── cli.py check ─────────────────────")
    import cli
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_check(None)
    out = buf.getvalue()
    check("check 성공", rc == 0, out[-300:] if rc else "")
    check("check 에 환경·주소 표시", "모의투자 (KIS_ENV=mock)" in out and VTS in out)
    check("check 에 모의 주문 경고", "모의투자 계좌로 주문" in out)

    print("── 설정 오류 ────────────────────────")
    os.environ["KIS_MOCK_APP_KEY"] = ""
    importlib.reload(config)
    check("모의 키 비면 KIS_MOCK_APP_KEY 로 알림", any("KIS_MOCK_APP_KEY" in p for p in config.validate()))
    os.environ["KIS_ENV"] = "paper"
    importlib.reload(config)
    check("KIS_ENV 오타 잡힘", any("KIS_ENV" in p for p in config.validate()))

    print("── 실전으로 되돌리면 ─────────────────")
    os.environ["KIS_ENV"] = "real"
    importlib.reload(config)
    check("실전 주소·키", config.KIS_BASE_URL == "https://openapi.koreainvestment.com:9443"
          and config.KIS_APP_KEY == "REAL_KEY" and config.CANO == "11111111")
    check("실전 TR_ID 그대로", trading.tr(trading.TR_BUY) == "TTTC0012U" and trading.tr(trading.TR_PENDING) == "TTTC0084R")
    check("실전 유량 기본값 2.5/초", config.KIS_RATE_LIMIT_PER_SEC == 2.5)
    check("실전 SSM 경로", config.SSM_TOKEN_PATH == "/ma5-bot/kis/token")

    print(f"\n{len(PASS)} 통과, {len(FAIL)} 실패")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
