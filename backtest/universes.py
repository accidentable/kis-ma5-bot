"""
backtest/universes.py — 백테스트용 유니버스 (KOSPI100 / KOSPI200 / KOSDAQ150)

한투 종목마스터에서 지수 편입 여부를 읽는다.
  KOSPI100   kospi_code.mst 의 KOSPI100 플래그
  KOSPI200   kospi_code.mst 의 KOSPI200섹터업종 값이 '0' 이 아닌 종목 (201종목 확인)
  KOSDAQ150  kosdaq_code.mst 의 KOSDAQ150지수여부 플래그

주의: 지금 시점의 구성종목으로 과거를 돌리므로 생존편향이 있다. 유니버스 간 상대 비교용이다.
관리종목·공매도과열 같은 일시 상태 플래그는 오늘 값이라 과거에 적용하면 틀리므로 쓰지 않는다.
"""
from __future__ import annotations

import io
import json
import logging
import os
import ssl
import urllib.request
import zipfile

from core import universe as live_universe

logger = logging.getLogger(__name__)

KOSDAQ_URL = "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip"

_KOSDAQ_TAIL = 222
_KOSDAQ_WIDTHS = [
    2, 1,
    4, 4, 4, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 9,
    5, 5, 1, 1, 1,
    2, 1, 1, 1, 2,
    2, 2, 3, 1, 3,
    12, 12, 8, 15, 21,
    2, 7, 1, 1, 1,
    1, 9, 9, 9, 5,
    9, 8, 9, 3, 1,
    1, 1,
]
_KOSDAQ_NAMES = [
    "증권그룹구분코드", "시가총액규모",
    "지수업종대분류", "지수업종중분류", "지수업종소분류", "벤처기업", "저유동성",
    "KRX", "ETP", "KRX100", "KRX자동차", "KRX반도체",
    "KRX바이오", "KRX은행", "SPAC", "KRX에너지화학", "KRX철강",
    "단기과열", "KRX미디어통신", "KRX건설", "투자주의환기", "KRX증권",
    "KRX선박", "KRX보험", "KRX운송", "KOSDAQ150", "기준가",
    "매매수량단위", "시간외수량단위", "거래정지", "정리매매", "관리종목",
    "시장경고", "경고예고", "불성실공시", "우회상장", "락구분",
    "액면변경", "증자구분", "증거금비율", "신용가능", "신용기간",
    "전일거래량", "액면가", "상장일자", "상장주수", "자본금",
    "결산월", "공모가", "우선주", "공매도과열", "이상급등",
    "KRX300", "매출액", "영업이익", "경상이익", "당기순이익",
    "ROE", "기준년월", "시가총액", "그룹사코드", "회사신용한도초과",
    "담보대출가능", "대주가능",
]
assert len(_KOSDAQ_WIDTHS) == len(_KOSDAQ_NAMES), (len(_KOSDAQ_WIDTHS), len(_KOSDAQ_NAMES))


def _download(url: str) -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".mst"))
        return z.read(name).decode("cp949", errors="replace")


def _parse(text: str, tail_total: int, widths: list[int], names: list[str]) -> list[dict]:
    tail_len = tail_total - 1
    rows = []
    for line in text.splitlines():
        if len(line) <= tail_len:
            continue
        head, tail = line[: len(line) - tail_len], line[-tail_len:]
        rec = {"종목코드": head[0:9].strip(), "종목명": head[21:].strip()}
        pos = 0
        for n, w in zip(names, widths):
            rec[n] = tail[pos: pos + w].strip()
            pos += w
        rows.append(rec)
    return rows


def _flag(v) -> bool:
    return str(v).strip().upper() in ("Y", "1")


def _marcap(v) -> float:
    try:
        return float(v or 0)
    except ValueError:
        return 0.0


def build(cache_path: str, need_all: bool = False) -> dict[str, list[dict]]:
    """{"KOSPI100": [...], "KOSPI200": [...], "KOSDAQ150": [...], "ALL": [...]} — 각 항목은 {ticker, name, market}.

    ALL 은 코스피·코스닥 보통주 전체 (ETF·ETN·리츠·스팩·우선주 제외). 생존편향을 줄이려고 쓴다 —
    지수 구성종목은 '최근에 오른 종목'이 들어와 있어서 과거로 돌리면 결과가 부풀려진다.
    그래도 상장폐지 종목은 마스터에 없어서 빠진다 (편향이 줄 뿐 사라지진 않는다)."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if not need_all or "ALL" in cached:
            return cached

    kospi = live_universe._parse_mst(live_universe._download_mst())
    kosdaq = _parse(_download(KOSDAQ_URL), _KOSDAQ_TAIL, _KOSDAQ_WIDTHS, _KOSDAQ_NAMES)

    def ok_code(c: str) -> bool:
        return len(c) == 6 and c.isalnum()

    k100, k200, q150, every = [], [], [], []
    for r in kospi:
        code = r["종목코드"]
        if not ok_code(code) or _flag(r.get("우선주")) or _flag(r.get("SPAC")):
            continue
        item = {"ticker": code, "name": r["종목명"], "market": "KOSPI", "marcap": _marcap(r.get("시가총액"))}
        if r.get("그룹코드") == "ST" and not _flag(r.get("ETP")) and code.endswith("0"):
            every.append(item)
        if _flag(r.get("KOSPI100")):
            k100.append(item)
        if str(r.get("KOSPI200섹터업종", "0")).strip() not in ("", "0"):
            k200.append(item)
    for r in kosdaq:
        code = r["종목코드"]
        if not ok_code(code) or _flag(r.get("SPAC")):
            continue
        item = {"ticker": code, "name": r["종목명"], "market": "KOSDAQ", "marcap": _marcap(r.get("시가총액"))}
        if _flag(r.get("KOSDAQ150")):
            q150.append(item)
        if (r.get("증권그룹구분코드") == "ST" and not _flag(r.get("ETP")) and not _flag(r.get("우선주"))
                and code.endswith("0")):
            every.append(item)

    out = {"KOSPI100": k100, "KOSPI200": k200, "KOSDAQ150": q150, "ALL": every}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    logger.info("유니버스: KOSPI100 %d / KOSPI200 %d / KOSDAQ150 %d / 전체 보통주 %d",
                len(k100), len(k200), len(q150), len(every))
    return out
