"""
core/universe.py — KOSPI100 유니버스

한국투자증권이 공개하는 종목마스터(kospi_code.mst)를 받아 파싱한다.
KOSPI100 편입 여부 플래그가 파일에 그대로 들어있어서, 지수 구성종목을
따로 스크래핑할 필요가 없다. 관리종목·거래정지·투자경고 같은 배제 조건도
같은 파일에서 읽는다.

마스터파일은 매 영업일 갱신되므로 하루 단위로 캐시한다.
"""
from __future__ import annotations

import io
import json
import logging
import os
import ssl
import urllib.request
import zipfile
from datetime import date
from typing import Optional

import config

logger = logging.getLogger(__name__)

MST_URL = "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip"

# 마스터파일 뒷부분 고정폭 레이아웃 (한투 '종목마스터정보(코스피).h' 기준)
_WIDTHS = [
    2, 1, 4, 4, 4,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 9, 5, 5, 1,
    1, 1, 2, 1, 1,
    1, 2, 2, 2, 3,
    1, 3, 12, 12, 8,
    15, 21, 2, 7, 1,
    1, 1, 1, 1, 9,
    9, 9, 5, 9, 8,
    9, 3, 1, 1, 1,
]
_NAMES = [
    "그룹코드", "시가총액규모", "지수업종대분류", "지수업종중분류", "지수업종소분류",
    "제조업", "저유동성", "지배구조지수종목", "KOSPI200섹터업종", "KOSPI100",
    "KOSPI50", "KRX", "ETP", "ELW발행", "KRX100",
    "KRX자동차", "KRX반도체", "KRX바이오", "KRX은행", "SPAC",
    "KRX에너지화학", "KRX철강", "단기과열", "KRX미디어통신", "KRX건설",
    "Non1", "KRX증권", "KRX선박", "KRX섹터_보험", "KRX섹터_운송",
    "SRI", "기준가", "매매수량단위", "시간외수량단위", "거래정지",
    "정리매매", "관리종목", "시장경고", "경고예고", "불성실공시",
    "우회상장", "락구분", "액면변경", "증자구분", "증거금비율",
    "신용가능", "신용기간", "전일거래량", "액면가", "상장일자",
    "상장주수", "자본금", "결산월", "공모가", "우선주",
    "공매도과열", "이상급등", "KRX300", "KOSPI", "매출액",
    "영업이익", "경상이익", "당기순이익", "ROE", "기준년월",
    "시가총액", "그룹사코드", "회사신용한도초과", "담보대출가능", "대주가능",
]
_TAIL = 228  # 뒷부분 고정폭 블록 길이 (개행 포함)


def _cache_path() -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return os.path.join(config.DATA_DIR, "universe.json")


def _download_mst() -> str:
    """마스터파일 zip 을 받아 .mst 본문을 문자열로 반환 (cp949)."""
    # 한투 다운로드 서버는 인증서 체인이 불완전할 때가 있어 별도 컨텍스트를 쓴다.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    logger.info("종목마스터 다운로드: %s", MST_URL)
    with urllib.request.urlopen(MST_URL, context=ctx, timeout=60) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".mst"))
        raw = z.read(name)

    return raw.decode("cp949", errors="replace")


def _parse_mst(text: str) -> list[dict]:
    """
    마스터파일 본문을 dict 목록으로 파싱한다.

    각 줄은 [가변길이 머리말][227자 고정폭 꼬리말] 구조다.
    머리말: 단축코드(9) + 표준코드(12) + 한글종목명(나머지)
    꼬리말: _WIDTHS 순서대로 _NAMES 필드
    """
    tail_len = _TAIL - 1  # splitlines 로 개행이 빠진 길이
    rows: list[dict] = []

    for line in text.splitlines():
        if len(line) <= tail_len:
            continue

        head = line[: len(line) - tail_len]
        tail = line[-tail_len:]

        rec = {
            "종목코드": head[0:9].strip(),
            "종목명": head[21:].strip(),
        }
        pos = 0
        for name, width in zip(_NAMES, _WIDTHS):
            rec[name] = tail[pos: pos + width].strip()
            pos += width
        rows.append(rec)

    return rows


def _is_flagged(value) -> bool:
    """마스터파일의 Y/N·1/0 플래그 해석."""
    s = str(value).strip().upper()
    return s in ("Y", "1")


def _build() -> list[dict]:
    """마스터파일에서 매매 대상 KOSPI100 종목 목록을 만든다."""
    rows = _parse_mst(_download_mst())

    total = len(rows)
    members = [r for r in rows if _is_flagged(r.get(config.UNIVERSE_FLAG))]
    picked = len(members)

    stocks: list[dict] = []
    excluded: dict[str, int] = {}

    for r in members:
        code = r["종목코드"]
        # 단축코드는 6자리 영숫자다. 숫자만으로 검사하면 분할·재상장 종목이
        # 통째로 빠진다 (예: 삼성에피스홀딩스 '0126Z0').
        if len(code) != 6 or not code.isalnum():
            continue

        hit = [f for f in config.EXCLUDE_FLAGS if _is_flagged(r.get(f))]
        if hit:
            for f in hit:
                excluded[f] = excluded.get(f, 0) + 1
            logger.info("유니버스 제외: %s(%s) — %s", r["종목명"], code, ", ".join(hit))
            continue

        try:
            marcap = float(r.get("시가총액") or 0)
        except ValueError:
            marcap = 0.0

        stocks.append({
            "ticker": code,
            "name": r["종목명"],
            "marcap": marcap,   # 단위: 억원
        })

    stocks.sort(key=lambda s: s["marcap"], reverse=True)
    logger.info(
        "유니버스 구성: 전체 %d종목 → %s %d종목 → 배제 후 %d종목 %s",
        total, config.UNIVERSE_FLAG, picked, len(stocks),
        f"(제외: {excluded})" if excluded else "",
    )
    return stocks


def get_universe(force: bool = False) -> list[dict]:
    """KOSPI100 유니버스. 당일 캐시가 있으면 재사용한다."""
    today = date.today().isoformat()
    path = _cache_path()

    if not force:
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("date") == today and cached.get("stocks"):
                return cached["stocks"]
        except (OSError, ValueError):
            pass

    try:
        stocks = _build()
    except Exception as e:
        logger.error("유니버스 생성 실패: %s", e)
        # 최후의 보루: 날짜가 지났어도 남아있는 캐시를 쓴다
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("stocks"):
                logger.warning("이전 캐시(%s)로 대체한다", cached.get("date"))
                return cached["stocks"]
        except (OSError, ValueError):
            pass
        raise

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"date": today, "stocks": stocks}, f, ensure_ascii=False)
    except OSError as e:
        logger.warning("유니버스 캐시 저장 실패: %s", e)

    return stocks


# ══════════════════════════════════════════════════════════════
# 시총 상위 N (코스피 + 코스닥) — 52주 신고가 근접 전략용
# ══════════════════════════════════════════════════════════════
KOSDAQ_URL = "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip"

# 코스닥 마스터 뒷부분 고정폭 레이아웃 (한투 '종목마스터정보(코스닥).h' 기준)
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
    "그룹코드", "시가총액규모",
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


def _download(url: str) -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    logger.info("종목마스터 다운로드: %s", url)
    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".mst"))
        return z.read(name).decode("cp949", errors="replace")


def _parse_kosdaq(text: str) -> list[dict]:
    tail_len = _KOSDAQ_TAIL - 1
    rows = []
    for line in text.splitlines():
        if len(line) <= tail_len:
            continue
        head, tail = line[: len(line) - tail_len], line[-tail_len:]
        rec = {"종목코드": head[0:9].strip(), "종목명": head[21:].strip()}
        pos = 0
        for n, w in zip(_KOSDAQ_NAMES, _KOSDAQ_WIDTHS):
            rec[n] = tail[pos: pos + w].strip()
            pos += w
        rows.append(rec)
    return rows


def _common_stock(r: dict) -> bool:
    """보통주만: 주식 그룹(ST), 6자리 코드 끝자리 0 (우선주 제외), ETF·스팩·우선주 플래그 없음."""
    code = r["종목코드"]
    if len(code) != 6 or not code.isalnum() or not code.endswith("0"):
        return False
    if r.get("그룹코드") != "ST":
        return False
    return not any(_is_flagged(r.get(f)) for f in ("ETP", "SPAC", "우선주"))


def build_large(top_n: int) -> list[dict]:
    """코스피 + 코스닥 보통주를 시가총액 순으로 top_n 개. 관리·정지·경고 등 배제 플래그 종목은 뺀다."""
    rows = [dict(r, market="KOSPI") for r in _parse_mst(_download_mst())]
    rows += [dict(r, market="KOSDAQ") for r in _parse_kosdaq(_download(KOSDAQ_URL))]
    stocks, excluded = [], 0
    for r in rows:
        if not _common_stock(r):
            continue
        if any(_is_flagged(r.get(f)) for f in config.EXCLUDE_FLAGS):
            excluded += 1
            continue
        try:
            marcap = float(r.get("시가총액") or 0)
        except ValueError:
            marcap = 0.0
        if marcap <= 0:
            continue
        stocks.append({"ticker": r["종목코드"], "name": r["종목명"], "market": r["market"], "marcap": marcap})
    stocks.sort(key=lambda s: s["marcap"], reverse=True)
    out = stocks[:top_n]
    logger.info("시총 상위 유니버스: 보통주 %d종목 (배제 %d) → 상위 %d", len(stocks), excluded, len(out))
    return out


def get_large_universe(top_n: int, force: bool = False) -> list[dict]:
    """시총 상위 top_n (코스피+코스닥). 하루 단위 캐시, 실패하면 지난 캐시로 대체."""
    today = date.today().isoformat()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, "universe_large.json")
    if not force:
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("date") == today and cached.get("top_n") == top_n and cached.get("stocks"):
                return cached["stocks"]
        except (OSError, ValueError):
            pass
    try:
        stocks = build_large(top_n)
    except Exception as e:
        logger.error("시총 상위 유니버스 생성 실패: %s", e)
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("stocks"):
                logger.warning("이전 캐시(%s)로 대체한다", cached.get("date"))
                return cached["stocks"]
        except (OSError, ValueError):
            pass
        raise
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"date": today, "top_n": top_n, "stocks": stocks}, f, ensure_ascii=False)
    except OSError as e:
        logger.warning("유니버스 캐시 저장 실패: %s", e)
    return stocks


def get_name(ticker: str, universe: Optional[list[dict]] = None) -> str:
    for s in universe or get_universe():
        if s["ticker"] == ticker:
            return s["name"]
    return ticker
