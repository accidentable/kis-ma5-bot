"""
config.py — 환경변수 로드 + 전략 파라미터

전략은 STRATEGY 로 고른다.
  near_high         52주 신고가 근접 로테이션
                    코스피+코스닥 시총 상위 200 중 '전일 종가 / 250일 최고가' 가 가장 높은 2종목을
                    반반 사서 21거래일 들고, 21거래일마다 다시 골라 교체한다. 손절·익절 없음.
                    근거: backtest/results/strategy_report5_*.md (KRX 전종목 16년, 상장폐지 포함)
  closebet          종가 베팅 (선택형): 강세 마감 테마주를 종가에 사서 다음 날 시가에 판다 — 검증 미통과
  contest (기본)    대회 모드: 20일 모멘텀 1위 1종목 집중 + 손절·추적 + 목표 락 — 한 달 +30% 확률 최대화 (contest_tail_20261002.md)
  ma5               예전 MA5 돌파 역발상 (5일선 아래 → 위 돌파 매수, 익절 | 5일선 이탈 | 3거래일)
"""
from __future__ import annotations

import os
import logging
from dotenv import load_dotenv

# 실전·모의 봇을 한 서버에서 따로 돌릴 때 systemd 가 ENV_FILE 로 설정 파일을 고른다 (없으면 .env).
load_dotenv(os.getenv("ENV_FILE") or None)

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════
# 한국투자증권 Open API — 실전(real) / 모의투자(mock)
# ══════════════════════════════════════════════════════════════
# KIS_ENV=mock 이면 KIS_MOCK_* 키·계좌와 모의투자 주소를 쓴다. 실전/모의 키는 서로 호환되지 않는다.
KIS_ENV: str = os.getenv("KIS_ENV", "real").strip().lower()
IS_MOCK: bool = KIS_ENV == "mock"

KIS_REAL_URL: str = "https://openapi.koreainvestment.com:9443"
KIS_MOCK_URL: str = "https://openapivts.koreainvestment.com:29443"

_KEY_PREFIX = "KIS_MOCK_" if IS_MOCK else "KIS_"
KIS_APP_KEY: str = os.getenv(f"{_KEY_PREFIX}APP_KEY", "").strip()
KIS_APP_SECRET: str = os.getenv(f"{_KEY_PREFIX}APP_SECRET", "").strip()
KIS_ACCOUNT_NO: str = os.getenv(f"{_KEY_PREFIX}ACCOUNT_NO", "").strip().replace("-", "")

KIS_BASE_URL: str = KIS_MOCK_URL if IS_MOCK else KIS_REAL_URL

# 계좌번호 분해: 앞 8자리 종합계좌, 뒤 2자리 상품코드
CANO: str = KIS_ACCOUNT_NO[:8]
ACNT_PRDT_CD: str = KIS_ACCOUNT_NO[8:10] if len(KIS_ACCOUNT_NO) >= 10 else "01"

# 2025년 NXT(대체거래소) 출범으로 주문 API 에 추가된 필수 필드.
#   KRX  한국거래소만
#   NXT  넥스트레이드만
#   SOR  최선주문집행 (두 거래소 중 유리한 쪽으로 자동 라우팅)
EXCG_ID_DVSN_CD: str = os.getenv("EXCG_ID_DVSN_CD", "KRX").strip().upper()

# 조회 API 유량제한. 공지상 실전은 초당 20건이지만 실제로는 훨씬 빡빡하게 걸린다.
# 모의투자는 초당 2건 (한투 공식 샘플도 호출마다 0.5초 쉰다).
# EGW00201 을 만나면 클라이언트가 스스로 더 낮춘다(core/kis/client.py).
KIS_RATE_LIMIT_PER_SEC: float = _env_float("KIS_RATE_LIMIT_PER_SEC", 1.5 if IS_MOCK else 2.5)
# 주문 API 는 초당 1건.
KIS_ORDER_INTERVAL_SEC: float = _env_float("KIS_ORDER_INTERVAL_SEC", 1.1)

# ══════════════════════════════════════════════════════════════
# 실행 옵션
# ══════════════════════════════════════════════════════════════
DRY_RUN: bool = _env_bool("DRY_RUN", True)
STATE_BACKEND: str = os.getenv("STATE_BACKEND", "local").strip().lower()
TOKEN_CACHE: str = os.getenv("TOKEN_CACHE", "file").strip().lower()

AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2").strip()
DYNAMODB_TABLE: str = os.getenv("DYNAMODB_TABLE", "ma5-bot-state").strip()
# 토큰·상태는 실전/모의를 따로 둔다 (모의 토큰으로 실전을 부르거나, 모의 보유를 실전 보유로 착각하지 않게).
SSM_TOKEN_PATH: str = "/ma5-bot/kis/token-mock" if IS_MOCK else "/ma5-bot/kis/token"

# Lambda 는 /tmp 만 쓰기 가능하다.
DATA_DIR: str = os.getenv("DATA_DIR", "/tmp/ma5-bot" if os.getenv("AWS_LAMBDA_FUNCTION_NAME") else "data")

# ══════════════════════════════════════════════════════════════
# 텔레그램
# ══════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_raw_ids = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
TELEGRAM_ALLOWED_CHAT_IDS: list[int] = [
    int(x.strip()) for x in _raw_ids.split(",") if x.strip().lstrip("-").isdigit()
]
# 알림 머리말. 실전·모의 봇을 같이 돌릴 때 어느 쪽 메시지인지 바로 보이게 한다. 비우면 머리말 없음.
BOT_LABEL: str = os.getenv("BOT_LABEL", "모의" if IS_MOCK else "실전").strip()

# ══════════════════════════════════════════════════════════════
# 전략 선택
# ══════════════════════════════════════════════════════════════
STRATEGY: str = os.getenv("STRATEGY", "contest").strip().lower()   # 2026-10 부터 기본은 대회 모드. 신고가 로테이션은 STRATEGY=near_high

# ── 52주 신고가 근접 로테이션 (STRATEGY=near_high) ─────────────
NH_UNIVERSE_TOP: int = _env_int("NH_UNIVERSE_TOP", 200)        # 코스피+코스닥 시총 상위 N
NH_SLOTS: int = _env_int("NH_SLOTS", 2)                        # 보유 종목 수 (계좌를 N 등분)
NH_HOLD_DAYS: int = _env_int("NH_HOLD_DAYS", 21)               # 교체 주기 (거래일)
NH_HIGH_LOOKBACK: int = _env_int("NH_HIGH_LOOKBACK", 250)      # 신고가 기준 기간 (거래일)
NH_MOM_DAYS: int = _env_int("NH_MOM_DAYS", 60)                 # 이 기간 수익률 > 0 인 종목만
NH_MIN_PRICE: float = _env_float("NH_MIN_PRICE", 1000)         # 주가 하한 (원)
NH_MIN_VALUE: float = _env_float("NH_MIN_VALUE", 1_000_000_000)  # 20일 평균 거래대금 하한 (원)
# 최근 20거래일 안에 하루 이 % 이상 오른 날이 있으면 뺀다 (급등 테마주 회피). 0 이면 끔.
# 16년 백테스트: 한 달 평균은 비슷하고, 한 달 −10% 이하 확률이 모든 구간에서 줄었다 (7→5, 8→5, 13→6, 16→5%).
NH_MAX_DAILY_GAIN_PCT: float = _env_float("NH_MAX_DAILY_GAIN_PCT", 10.0)
# 보유 종목이 하루 이 % 이상 오르면 그날 마감 작업(15:15)에서 판다 (급등 뒤 부진). 0 이면 끔.
# 빈 자리는 다음 날 순위로 채운다 (NH_REFILL). 16년 백테스트: 한 달 평균 −0.1/+0.9/+1.5/+0.4% → −0.1/+1.1/+1.7/+1.0%
NH_SURGE_EXIT_PCT: float = _env_float("NH_SURGE_EXIT_PCT", 10.0)
NH_CANDIDATES: int = _env_int("NH_CANDIDATES", 15)             # 순위표에 남길 후보 수 (비싸서 못 사면 다음 순위)
# 교체일이 아닌 날에도 빈 슬롯이 있으면 그날 순위로 채운다 (매수 실패·수동 매도 뒤 복구용)
NH_REFILL: bool = _env_bool("NH_REFILL", True)
NH_PREP_TIME: str = os.getenv("NH_PREP_TIME", "0820").strip()   # 순위 계산 (시총 200 × 일봉 250개, 3~5분)
NH_ENTRY_TIME: str = os.getenv("NH_ENTRY_TIME", "0905").strip()  # 교체 매도 → 매수
NH_BUY_CUTOFF: str = os.getenv("NH_BUY_CUTOFF", "1430").strip()  # 이 시각 이후엔 빈 슬롯 매수 재시도 안 함
# 손절 (%). 0 이면 없음 — 백테스트는 손절 없이 검증했다. 켜면 검증 밖의 규칙이 된다.
NH_STOP_LOSS_PCT: float = _env_float("NH_STOP_LOSS_PCT", 0.0)

# ── 패닉 모드 (near_high 위에 얹는 선택 기능, 기본 꺼짐) ───────────
# 장 마감 뒤 시장 평균 등락률이 −PANIC_MKT_DROP_PCT% 이하면, 다음 날 평소 보유를 모두 팔고 최근 5일 가장 많이 빠진
# 과매도주 PANIC_SLOTS 개를 같은 금액씩 사서 PANIC_HOLD_DAYS 거래일째 15:15 에 판다. 규칙은 core/panic.py.
# 백테스트 (1억, 5종목): 봇 대비 한 달 평균 2011~2019 +1.4%p, 2020~ +1.7%p (여러 설정을 본 뒤 고른 값 — 실제는 더 낮을 수 있다)
PANIC_ENABLED: bool = _env_bool("PANIC_ENABLED", False)
PANIC_MKT_DROP_PCT: float = _env_float("PANIC_MKT_DROP_PCT", 4.0)        # 급락일 기준 (시장 평균 등락률 %)
PANIC_SLOTS: int = _env_int("PANIC_SLOTS", 5)                           # 패닉 때 살 종목 수 (계좌를 N 등분)
PANIC_HOLD_DAYS: int = _env_int("PANIC_HOLD_DAYS", 5)                   # 보유 거래일 (산 날 = 1일째)
PANIC_UNIVERSE_TOP: int = _env_int("PANIC_UNIVERSE_TOP", 1500)          # 시장 평균 계산 범위 (시총 상위 N)
PANIC_MKT_MIN_VALUE: float = _env_float("PANIC_MKT_MIN_VALUE", 3_000_000_000)   # 시장 평균에 넣을 20일 거래대금 하한
PANIC_PICK_TOP: int = _env_int("PANIC_PICK_TOP", 1000)                  # 후보 시총 순위 상한
PANIC_PICK_MIN_VALUE: float = _env_float("PANIC_PICK_MIN_VALUE", 2_000_000_000)  # 후보 20일 거래대금 하한
PANIC_IBS_MAX: float = _env_float("PANIC_IBS_MAX", 0.7)                 # 고가 근처 마감 종목 제외 (종가 위치)
PANIC_VR_MAX: float = _env_float("PANIC_VR_MAX", 3.0)                   # 오늘 거래대금이 20일 평균의 N 배 이상이면 제외
PANIC_CANDIDATES: int = _env_int("PANIC_CANDIDATES", 15)                # 저장할 후보 수 (못 사면 다음 순위)
PANIC_SCAN_TIME: str = os.getenv("PANIC_SCAN_TIME", "1535").strip()     # 급락 판정 (시총 1500 × 일봉, 모의투자 약 10분)

# ── 대회 수상 조건 (수동 매매 /progress 가 비교한다) ───────────
# 예: 2026-10-12. 비우면 이번 달 1일. 대회 모드는 이 날부터 목표 락·조건용 매매를 한 기간으로 센다 (달이 바뀌어도 리셋 안 함)
CONTEST_START: str = os.getenv("CONTEST_START", "").strip()
CONTEST_MIN_AMOUNT: float = _env_float("CONTEST_MIN_AMOUNT", 500_000_000)   # 매매금액 (체결, 매수+매도)
CONTEST_MIN_DAYS: int = _env_int("CONTEST_MIN_DAYS", 5)                    # 매매일수
CONTEST_MIN_STOCKS: int = _env_int("CONTEST_MIN_STOCKS", 5)                # 매매종목수 (코스피200 · 코스닥150)

# ── 종가 베팅 (STRATEGY=closebet, 선택형 — 기본 아님) ───────────
# 당일 강세 마감 테마주(거래대금 상위)를 장마감 동시호가에 사서 다음 날 장전 동시호가에 판다.
# 16년 백테스트에서 기대값 0 근처 (IS +0.4%, VAL −0.8%, TEST +0.3% / 한 달). 검증을 통과하지 못했다.
CB_SLOTS: int = _env_int("CB_SLOTS", 2)
CB_RANK_TOP: int = _env_int("CB_RANK_TOP", 30)                 # 당일 거래대금 순위 N 위 안
CB_MIN_CHANGE_PCT: float = _env_float("CB_MIN_CHANGE_PCT", 5.0)   # 당일 등락률 하한
CB_MAX_CHANGE_PCT: float = _env_float("CB_MAX_CHANGE_PCT", 29.0)  # 이상은 상한가 근처라 못 산다
CB_MIN_IBS: float = _env_float("CB_MIN_IBS", 0.9)               # (현재가−저가)/(고가−저가) — 고가 근처 마감
CB_MIN_VALUE: float = _env_float("CB_MIN_VALUE", 5_000_000_000)   # 당일 거래대금 하한 (원)
CB_REGIME: bool = _env_bool("CB_REGIME", True)                  # 코스닥지수가 100일선 위일 때만
CB_BUY_TICKS: int = _env_int("CB_BUY_TICKS", 5)                 # 장마감 동시호가 매수 지정가 = 현재가 + N틱
CB_SELL_TIME: str = os.getenv("CB_SELL_TIME", "0845").strip()   # 장전 동시호가 매도
CB_CHECK_TIME: str = os.getenv("CB_CHECK_TIME", "0905").strip() # 안 팔린 것 현재가 매도
CB_BUY_TIME: str = os.getenv("CB_BUY_TIME", "1521").strip()     # 장마감 동시호가(15:20~15:30) 매수

# ── 대회 모드 (STRATEGY=contest) ─────────────────────────────
# 한 달 안에 +CT_LOCK_PCT% 를 한 번 찍을 확률을 노린다 (평균 수익이 아니다). 근거 backtest/results/contest_tail_20261002.md:
# 20일 모멘텀 1위 1종목 · 손절 10% · 추적 10% · 보유 10일. 락 30% → 2020~25 월 +30% 확률 29%, −20% 확률 22%.
# 29거래일 창(대회 길이) 락 50%: +50% 확률 24%, +30% 31%, −30% 14% (락 30% 는 +30% 39%, +50% 12%).
CT_SLOTS: int = _env_int("CT_SLOTS", 1)                          # 동시에 드는 종목 수 (1 이 꼬리 확률 최대, 2 면 −30% 위험 절반)
CT_LOOKBACK: int = _env_int("CT_LOOKBACK", 20)                   # 순위에 쓰는 수익률 기간 (거래일)
CT_TOP_PCT: float = _env_float("CT_TOP_PCT", 10.0)               # 대상 중 상위 몇 % 안에서 고르나
CT_STOP_PCT: float = _env_float("CT_STOP_PCT", 10.0)             # 손절 (종가 기준 → 다음 날 시가)
CT_TRAIL_PCT: float = _env_float("CT_TRAIL_PCT", 10.0)           # 추적 손절 (보유 중 최고 종가 대비)
CT_HOLD_DAYS: int = _env_int("CT_HOLD_DAYS", 10)                 # 최대 보유 거래일, 지나면 다음 날 시가 매도 후 1위로 교체
CT_LOCK_PCT: float = _env_float("CT_LOCK_PCT", 50.0)             # 월초 순자산 대비 이만큼 넘으면 전량 매도 후 월말까지 현금 (10/02 30→50: +50% 노림)
CT_CRASH_ENABLED: bool = _env_bool("CT_CRASH_ENABLED", True)     # 폭락일 급락주 전환
CT_CRASH_MKT_PCT: float = _env_float("CT_CRASH_MKT_PCT", 3.0)    # 대상 평균 등락 −N% 이하
CT_CRASH_SIGMA: float = _env_float("CT_CRASH_SIGMA", 3.0)        # 그리고 직전 20일 시장 변동성의 N 배 이하
CT_CRASH_DROP_PCT: float = _env_float("CT_CRASH_DROP_PCT", 7.0)  # 급락주 후보: 오늘 −N% 이하
CT_CRASH_VR_MAX: float = _env_float("CT_CRASH_VR_MAX", 3.0)      # 거래대금이 20일 평균의 N 배 이상이면 제외 (뉴스 의심)
CT_CRASH_IDIO_MULT: float = _env_float("CT_CRASH_IDIO_MULT", 3.0)  # 종목 낙폭이 시장 낙폭의 N 배 이상이면 제외 (혼자 빠짐)
CT_CRASH_HOLD: int = _env_int("CT_CRASH_HOLD", 5)                # 전환 보유 거래일
CT_CRASH_MIN_N: int = _env_int("CT_CRASH_MIN_N", 100)           # 시장 평균을 믿으려면 현재가가 잡힌 종목이 이만큼은 돼야
CT_FILLER_N: int = _env_int("CT_FILLER_N", 4)                    # 대회 '지수 종목 5개 거래' 용: 기간 첫 매수 때 다음 순위 N 종목 1주씩 (다음 날 매도). 0 = 끔
CT_FALLBACK_N: int = _env_int("CT_FALLBACK_N", 10)              # 고른 종목이 1주도 못 살 만큼 비싸면 다음 순위로 내려가는 최대 수 (소액 실전용)
CT_PAPER_CAP: float = _env_float("CT_PAPER_CAP", 100_000_000)   # DRY_RUN 일 때 수량·순자산 계산에 쓰는 종이 계좌 (실계좌 잔고 대신)
CT_BRIEF_N: int = _env_int("CT_BRIEF_N", 5)                      # 아침 브리핑에 보여줄 후보 수 (어제 종가 기준 모멘텀 상위)
CT_BUY_TICKS: int = _env_int("CT_BUY_TICKS", 5)                  # 장마감 동시호가 매수 지정가 = 현재가 + N틱 (상한가 이내)
CT_KOSPI_N: int = _env_int("CT_KOSPI_N", 200)                    # 대상: 코스피 시총 상위 N
CT_KOSDAQ_N: int = _env_int("CT_KOSDAQ_N", 150)                  #       코스닥 시총 상위 N
CT_UNIVERSE_PULL: int = _env_int("CT_UNIVERSE_PULL", 1500)       # 둘을 뽑기 위해 받는 시총 상위 수
CT_PREP_TIME: str = os.getenv("CT_PREP_TIME", "0820").strip()    # 일봉 캐시 (350종목, 2~3분)
CT_SELL_TIME: str = os.getenv("CT_SELL_TIME", "0845").strip()    # 장전 동시호가 매도
CT_CHECK_TIME: str = os.getenv("CT_CHECK_TIME", "0905").strip()  # 안 팔린 것 현재가 매도
CT_SCAN_TIME: str = os.getenv("CT_SCAN_TIME", "1505").strip()    # 판정 (현재가 350개, 2~3분)
CT_BUY_TIME: str = os.getenv("CT_BUY_TIME", "1520").strip()      # 장마감 동시호가 매수
CT_EVAL_TIME: str = os.getenv("CT_EVAL_TIME", "1540").strip()    # 종가 판정 · 리포트

# ══════════════════════════════════════════════════════════════
# 유니버스 (STRATEGY=ma5)
# ══════════════════════════════════════════════════════════════
# 한투 종목마스터(kospi_code.mst)의 KOSPI100 플래그를 그대로 쓴다.
UNIVERSE_FLAG: str = "KOSPI100"

# 마스터파일에서 걸러낼 종목 (플래그가 'Y' 또는 '1' 이면 제외)
EXCLUDE_FLAGS: tuple[str, ...] = (
    "관리종목", "거래정지", "정리매매", "시장경고", "경고예고",
    "불성실공시", "단기과열", "공매도과열", "이상급등", "우선주", "SPAC",
)

# 20일 평균 거래대금 하한 (원). 이 아래는 전액 매수 시 체결이 밀린다.
MIN_TRADING_VALUE: float = _env_float("MIN_TRADING_VALUE", 3_000_000_000)

# ══════════════════════════════════════════════════════════════
# 진입 조건
# ══════════════════════════════════════════════════════════════
MA_PERIOD: int = 5

# "5일선 아래에서 놀다가" 의 정의:
#   최근 BELOW_LOOKBACK 거래일 중 종가가 그날의 MA5 아래였던 날이
#   BELOW_MIN_DAYS 일 이상이어야 한다. (전일 종가 < 전일 MA5 는 별도 필수 조건)
BELOW_LOOKBACK: int = _env_int("BELOW_LOOKBACK", 5)
BELOW_MIN_DAYS: int = _env_int("BELOW_MIN_DAYS", 3)

# 09:05 시점에 전일 종가 대비 이미 이만큼 올라 있으면 추격으로 보고 스킵.
MAX_CHASE_PCT: float = _env_float("MAX_CHASE_PCT", 5.0)

# ── 중장기 추세 필터: 골든크로스 상태 ─────────────────────────
# 20일선이 60일선 위에 있는 종목만 산다. "종가 하나가 60일선 위" 보다 훨씬 안정적인
# 추세 확인이다 — 하락추세 중 하루 반등으로 60일선을 잠깐 넘는 종목을 걸러낸다.
USE_TREND_FILTER: bool = _env_bool("USE_TREND_FILTER", True)
TREND_MA_SHORT: int = _env_int("TREND_MA_SHORT", 20)
TREND_MA_LONG: int = _env_int("TREND_MA_LONG", 60)
# 종가 > 60일선 도 함께 요구할지. 켜면 깊게 눌린 후보가 빠진다.
TREND_REQUIRE_PRICE_ABOVE: bool = _env_bool("TREND_REQUIRE_PRICE_ABOVE", False)
# 골든크로스 발생일 탐색 범위 (참고 정보용. 랭킹엔 안 쓴다 — 횡보장에선 크로스가 노이즈라서)
CROSS_LOOKBACK: int = _env_int("CROSS_LOOKBACK", 30)

# 후보가 여러 개일 때 1종목을 고르는 가중치 (각 지표를 후보 내 백분위로 환산 후 가중합)
# "수렴 후 발산" 자리를 우선한다 — 이평선이 뭉치고 변동폭이 줄어든 곳에서 5일선을 뚫는 종목.
#   squeeze      5·20·60일선 수렴도 — 최고−최저 / 주가. 작을수록 ↑ (1순위)
#   contraction  ATR(5)/ATR(20) — 봉이 작아지며 조여드는 정도. 작을수록 ↑
#   depth        전일 5일 이격도가 낮을수록 ↑ — 얼마나 깊이 눌렸다 올라오는가
#   thrust       돌파 강도 — 현재가가 5일선을 얼마나 확실히 넘었는가
#   trend        20일선/60일선 이격 — 추세 강도
#   value        20일 평균 거래대금 — 전액 매수를 소화할 유동성
RANK_WEIGHTS: dict[str, float] = {
    "squeeze": _env_float("RANK_W_SQUEEZE", 1.0),
    "contraction": _env_float("RANK_W_CONTRACTION", 0.7),
    "depth": _env_float("RANK_W_DEPTH", 0.3),
    "thrust": _env_float("RANK_W_THRUST", 0.5),
    "trend": _env_float("RANK_W_TREND", 0.3),
    "value": _env_float("RANK_W_VALUE", 0.3),
}

# ══════════════════════════════════════════════════════════════
# 포지션 / 청산
# ══════════════════════════════════════════════════════════════
MAX_POSITIONS: int = _env_int("MAX_POSITIONS", 1)
POSITION_PCT: float = _env_float("POSITION_PCT", 100.0)

# 1순위 종목이 주가가 높아 1주도 못 사는 경우, 다음 순위로 내려가며 시도한다.
# 0 이면 무제한. 신호 품질이 낮은 하위 후보까지 내려가는 게 싫으면 작게 잡는다.
ENTRY_FALLBACK_MAX_RANK: int = _env_int("ENTRY_FALLBACK_MAX_RANK", 5)

# ── 장중 재진입 ────────────────────────────────────────────
# 09:05 스캔에서 1차 통과한 종목을 그날 캐시해두고, 슬롯이 비면 10분 감시 때마다
# 그 종목들 현재가만 조회해 돌파 여부를 다시 본다 (일봉 재조회 없음).
INTRADAY_REENTRY: bool = _env_bool("INTRADAY_REENTRY", True)
# 재진입 허용 시간대 (HHMM, KST). 09:05 첫 진입 전 개장 직후 잡음 구간은 피하고,
# 늦은 오후엔 사봐야 움직일 시간이 없다.
REENTRY_START: str = os.getenv("REENTRY_START", "0910").strip()
REENTRY_CUTOFF: str = os.getenv("REENTRY_CUTOFF", "1430").strip()
# 당일 매수·매도한 종목은 그날 다시 안 산다. 익절 직후 같은 판정으로 더 비싸게 되사는 걸 막는다.
NO_SAME_DAY_REENTRY: bool = _env_bool("NO_SAME_DAY_REENTRY", True)

# ── 하루 일정 (HHMM, KST) ─────────────────────────────────
PREP_TIME: str = os.getenv("PREP_TIME", "0840").strip()              # 일봉 받아 후보 계산·캐시
PREMARKET_ENTRY_TIME: str = os.getenv("PREMARKET_ENTRY_TIME", "0850").strip()  # 프리마켓 판정 → 분할매수
ENTRY_TIME: str = os.getenv("ENTRY_TIME", "0905").strip()            # 단일 진입 (분할이 없을 때 대체)

# ── 프리마켓 분할 진입 ───────────────────────────────────
# 08:50 에 프리마켓(NXT) 가격으로 돌파를 판정하고, 본장 전에 분할 매수를 걸어둔다.
# 09:00 동시호가에서 시가가 기준가 이하면 그 자리에서 채워지고, 아니면 대기하다 눌릴 때 채워진다.
PREMARKET_SPLIT_ENTRY: bool = _env_bool("PREMARKET_SPLIT_ENTRY", True)
# 프리마켓 가격 조회에 쓸 시장코드 순서. UN=KRX+NXT 통합, NX=NXT. 둘 다 못 받으면 그 종목은 건너뛴다.
PREMARKET_MARKET_CODES: list[str] = [
    x.strip().upper() for x in os.getenv("PREMARKET_MARKET_CODES", "UN,NX").split(",") if x.strip()
]
# 선주문 건수. 1 이면 프리마켓가 지정가 한 건으로 동시호가에 참여한다 (기본).
# 2 이상이면 아래로 SPLIT_STEP_ATR 간격으로 분할 — 수렴→발산(슈팅) 전략에선 권하지 않는다:
# 진짜 슈팅은 안 눌려서 1차만 채워지고, 실패한 돌파는 눌리며 전량 채워진다 (맞을 땐 작게, 틀릴 땐 크게).
SPLIT_TRANCHES: int = _env_int("SPLIT_TRANCHES", 1)
# ── 장 초반 눌림 대기 ────────────────────────────────────
# 수렴 구간은 이평선이 뭉쳐 있어 개장 직후 변동성이 크고, 특히 장 초반에 훅 빠지는 일이 잦다.
# 그 자리가 진입 기회였다는 관찰에 따라, SPLIT_CANCEL_AT(10:00) 전의 모든 진입은
# "기준가 + PREOPEN_OFFSET_ATR × ATR" 지정가를 걸어두고 눌림을 기다린다.
#   08:50 선주문   기준가 = 프리마켓가. 동시호가부터 유효하므로 시가가 그 아래면 시가에 체결
#   09:05 대체·재진입  기준가 = 그 시각 현재가
# SPLIT_CANCEL_AT 까지 안 빠지면 취소하고 현재가로 한 번에 산다. 그 이후 진입은 처음부터 현재가.
# ATR = 최근 14일 평균 하루 고저폭. 대형주는 주가의 2~3% 라 −0.5ATR ≈ −1.3%.
# 5일선 − 1ATR 인 급이탈선보다 항상 위에 걸리므로, 눌림에 사자마자 급이탈로 나가는 구조는 아니다.
PREOPEN_OFFSET_ATR: float = _env_float("PREOPEN_OFFSET_ATR", -0.5)
# 기준가에 곱하는 % 보정. ATR 오프셋과 함께 적용된다. 보통 0.
PREOPEN_CHASE_PCT: float = _env_float("PREOPEN_CHASE_PCT", 0.0)
# 분할 간격 (ATR 배수). SPLIT_TRANCHES ≥ 2 일 때만 의미.
SPLIT_STEP_ATR: float = _env_float("SPLIT_STEP_ATR", 0.5)
# 눌림 대기 마감. 이 시각에 안 채워진 대기 주문을 취소하고, 슬롯이 비면 같은 틱에 현재가로 산다.
# 보통 STOP_BLACKOUT_UNTIL 과 같게 둔다 — "장 초반 한 시간" 을 한 덩어리로 보는 것.
SPLIT_CANCEL_AT: str = os.getenv("SPLIT_CANCEL_AT", "1000").strip()

# ── 손절 발동 유예 ────────────────────────────────────────
# 이 시각 전엔 5일선 급이탈·손절을 보지 않는다 (익절은 본다). 개장 직후 변동성 구간을
# 분할매수로 받아내고 흔들림을 견디기 위한 것. 대신 갭다운은 이 시각까지 그대로 안고 간다.
STOP_BLACKOUT_UNTIL: str = os.getenv("STOP_BLACKOUT_UNTIL", "1000").strip()

TAKE_PROFIT_PCT: float = _env_float("TAKE_PROFIT_PCT", 3.0)
# 장중 익절에 주는 여유폭 (ATR 배수). 손절선을 5일선 아래로 내린 것과 대칭.
# 장중엔 +TAKE_PROFIT_PCT% 목표에서 이만큼 더 오른 자리에서만 판다. 종가엔 여유 없이 +3%.
# 0 이면 장중에도 +3% 그대로 (예전 동작).
TP_INTRADAY_ATR_BUFFER: float = _env_float("TP_INTRADAY_ATR_BUFFER", 0.5)
MAX_HOLD_TRADING_DAYS: int = _env_int("MAX_HOLD_TRADING_DAYS", 3)

# 5일선 이탈을 장중에도 감시할지, 종가(15:15)에만 확인할지.
EXIT_ON_INTRADAY_MA5_BREAK: bool = _env_bool("EXIT_ON_INTRADAY_MA5_BREAK", True)
# 장중 이탈 판정에 주는 여유폭 (ATR 배수). 5일선은 종가로 만드는 선이라 장중 틱과 그대로
# 비교하면 하루 ±1~2% 잡음이 전부 '이탈' 로 보인다. 1.0 이면 5일선보다 1×ATR 아래로
# 내려가야 장중 이탈로 친다 — 대형주 기준 2~3% 아래. 종가 판정은 여유 없이 5일선 그대로.
# 0 이면 장중에도 5일선을 살짝만 밑돌아도 판다 (예전 동작 — 진입선과 청산선이 같아 잦은 손절).
INTRADAY_MA5_ATR_BUFFER: float = _env_float("INTRADAY_MA5_ATR_BUFFER", 1.0)
# ATR 을 못 구했을 때 쓰는 대체 여유폭 (5일선 대비 %)
INTRADAY_MA5_PCT_BUFFER_FALLBACK: float = _env_float("INTRADAY_MA5_PCT_BUFFER_FALLBACK", 2.0)

# 손절: 사용자 선택에 따라 기본 비활성. 5일선 이탈이 유일한 청산 방어선이다.
USE_STOP_LOSS: bool = _env_bool("USE_STOP_LOSS", False)
STOP_LOSS_PCT: float = _env_float("STOP_LOSS_PCT", 3.0)

# ══════════════════════════════════════════════════════════════
# 주문
# ══════════════════════════════════════════════════════════════
# 즉시 체결을 노리는 지정가 주문. 현재가보다 N틱 위(매수)/아래(매도)에 낸다.
# 시장가를 쓰지 않는 이유: 전액 주문이라 호가가 얇으면 슬리피지를 그대로 맞는다.
ENTRY_LIMIT_TICKS: int = _env_int("ENTRY_LIMIT_TICKS", 2)
EXIT_LIMIT_TICKS: int = _env_int("EXIT_LIMIT_TICKS", 2)
# 익절은 급할 게 없으므로 더 얕게 낸다. 3% 목표에서 슬리피지 0.1~0.2% 는 무시 못 할 크기다.
TP_EXIT_LIMIT_TICKS: int = _env_int("TP_EXIT_LIMIT_TICKS", 1)

# 미체결 주문을 이 시각에 정리한다 (HHMM, KST)
CANCEL_PENDING_AT: str = os.getenv("CANCEL_PENDING_AT", "1515").strip()

ORD_DVSN_LIMIT: str = "00"   # 지정가
ORD_DVSN_MARKET: str = "01"  # 시장가


# ══════════════════════════════════════════════════════════════
def validate() -> list[str]:
    """설정 검증. 치명적 문제의 목록을 반환한다 (빈 리스트면 정상)."""
    problems: list[str] = []

    if KIS_ENV not in ("real", "mock"):
        problems.append(f"KIS_ENV 값이 잘못됨: {KIS_ENV} (real/mock)")
    if not KIS_APP_KEY:
        problems.append(f"{_KEY_PREFIX}APP_KEY 가 비어 있다")
    if not KIS_APP_SECRET:
        problems.append(f"{_KEY_PREFIX}APP_SECRET 가 비어 있다")
    if len(KIS_ACCOUNT_NO) != 10:
        problems.append(
            f"{_KEY_PREFIX}ACCOUNT_NO 는 숫자 10자리여야 한다 (종합계좌 8 + 상품코드 2). 현재 {len(KIS_ACCOUNT_NO)}자리"
        )
    if EXCG_ID_DVSN_CD not in ("KRX", "NXT", "SOR"):
        problems.append(f"EXCG_ID_DVSN_CD 값이 잘못됨: {EXCG_ID_DVSN_CD} (KRX/NXT/SOR)")
    if STATE_BACKEND not in ("local", "dynamodb"):
        problems.append(f"STATE_BACKEND 값이 잘못됨: {STATE_BACKEND}")
    if not 0 < POSITION_PCT <= 100:
        problems.append(f"POSITION_PCT 는 0 초과 100 이하: {POSITION_PCT}")
    if BELOW_MIN_DAYS > BELOW_LOOKBACK:
        problems.append("BELOW_MIN_DAYS 가 BELOW_LOOKBACK 보다 클 수 없다")
    if STRATEGY not in ("near_high", "ma5", "closebet", "contest"):
        problems.append(f"STRATEGY 값이 잘못됨: {STRATEGY} (near_high/closebet/contest/ma5)")
    if CT_SLOTS < 1 or CT_LOCK_PCT <= 0 or CT_LOOKBACK < 5:
        problems.append("CT_SLOTS ≥ 1, CT_LOCK_PCT > 0, CT_LOOKBACK ≥ 5")
    if NH_SLOTS < 1 or NH_HOLD_DAYS < 1:
        problems.append("NH_SLOTS, NH_HOLD_DAYS 는 1 이상")
    if PANIC_ENABLED and STRATEGY != "near_high":
        problems.append("PANIC_ENABLED 는 STRATEGY=near_high 에서만 동작한다")
    if PANIC_SLOTS < 1 or PANIC_HOLD_DAYS < 1:
        problems.append("PANIC_SLOTS, PANIC_HOLD_DAYS 는 1 이상")

    return problems


def summary() -> str:
    """현재 설정 요약 (텔레그램/로그용)."""
    env = "모의투자" if IS_MOCK else "실전"
    mode = f"{env} · " + ("주문 안 보냄(DRY_RUN)" if DRY_RUN else "주문 전송")
    if STRATEGY == "contest":
        return (
            f"🏆 대회 모드  ({env} · {'DRY_RUN · 주문 안 보냄' if DRY_RUN else '실주문'})\n"
            f"계좌 {CANO[:4]}****{ACNT_PRDT_CD} · {EXCG_ID_DVSN_CD}\n\n"
            f"목표  한 달 +{CT_LOCK_PCT:g}% 한 번 (평균 아님)\n"
            f"종목  코스피{CT_KOSPI_N}·코스닥{CT_KOSDAQ_N} 중\n"
            f"      {CT_LOOKBACK}일 수익률 상위 {CT_TOP_PCT:g}% 1위 {CT_SLOTS}종목\n\n"
            f"매수  {CT_BUY_TIME[:2]}:{CT_BUY_TIME[2:]} 종가 (동시호가)\n"
            f"매도  손절 −{CT_STOP_PCT:g}% / 고점 −{CT_TRAIL_PCT:g}% / {CT_HOLD_DAYS}일\n"
            f"      → 다음 날 {CT_SELL_TIME[:2]}:{CT_SELL_TIME[2:]} 시가\n"
            f"락    월초 +{CT_LOCK_PCT:g}% 넘으면 전량 매도"
            + (f"\n폭락  시장 −{CT_CRASH_MKT_PCT:g}%·{CT_CRASH_SIGMA:g}σ → 급락주 {CT_CRASH_HOLD}일" if CT_CRASH_ENABLED else "")
            + (f"\n조건  첫 매수 때 {CT_FILLER_N}종목 1주씩" if CT_FILLER_N else "")
        )
    if STRATEGY == "closebet":
        return (
            f"모드: {mode} | 계좌: {CANO[:4]}****{ACNT_PRDT_CD} | 거래소: {EXCG_ID_DVSN_CD}\n"
            f"전략: 종가 베팅 (테마주, 선택형 — 백테스트 기대값 0 근처)\n"
            f"매수: {CB_BUY_TIME[:2]}:{CB_BUY_TIME[2:]} 장마감 동시호가 — 거래대금 {CB_RANK_TOP}위 안, "
            f"+{CB_MIN_CHANGE_PCT:g}~{CB_MAX_CHANGE_PCT:g}%, IBS ≥ {CB_MIN_IBS:g}, 거래대금 {CB_MIN_VALUE / 1e8:,.0f}억↑"
            + (", 코스닥 100일선 위" if CB_REGIME else "") + f" | {CB_SLOTS}종목\n"
            f"매도: 다음 날 {CB_SELL_TIME[:2]}:{CB_SELL_TIME[2:]} 장전 동시호가 (시가), "
            f"{CB_CHECK_TIME[:2]}:{CB_CHECK_TIME[2:]} 남은 것 현재가 매도"
        )
    if STRATEGY == "near_high":
        stop = f"{NH_STOP_LOSS_PCT:g}%" if NH_STOP_LOSS_PCT > 0 else "없음"
        return (
            f"모드: {mode} | 계좌: {CANO[:4]}****{ACNT_PRDT_CD} | 거래소: {EXCG_ID_DVSN_CD}\n"
            f"전략: 52주 신고가 근접 로테이션\n"
            f"유니버스: 코스피+코스닥 시총 상위 {NH_UNIVERSE_TOP} (주가 {NH_MIN_PRICE:,.0f}원↑, "
            f"거래대금 {NH_MIN_VALUE / 1e8:,.0f}억↑, {NH_MOM_DAYS}일 수익률 > 0"
            + (f", 20일 내 +{NH_MAX_DAILY_GAIN_PCT:g}%↑ 급등일 없음" if NH_MAX_DAILY_GAIN_PCT > 0 else "") + ")\n"
            f"순위: 전일 종가 / {NH_HIGH_LOOKBACK}일 최고가 (높을수록 먼저)\n"
            f"보유: {NH_SLOTS}종목 균등 | {NH_HOLD_DAYS}거래일마다 교체 | 손절 {stop}"
            + (f" | 하루 +{NH_SURGE_EXIT_PCT:g}%↑ 급등 시 마감 때 매도" if NH_SURGE_EXIT_PCT > 0 else "") + "\n"
            f"일정: {NH_PREP_TIME[:2]}:{NH_PREP_TIME[2:]} 순위 계산 → {NH_ENTRY_TIME[:2]}:{NH_ENTRY_TIME[2:]} 교체 매매 "
            f"(빈 슬롯은 {NH_BUY_CUTOFF[:2]}:{NH_BUY_CUTOFF[2:]}까지 10분마다 재시도)"
            + (f"\n패닉 모드: 켜짐 — {PANIC_SCAN_TIME[:2]}:{PANIC_SCAN_TIME[2:]} 시장 평균 −{PANIC_MKT_DROP_PCT:g}%↓ 판정 → "
               f"다음 날 5일 최대 낙폭주 {PANIC_SLOTS}종목 (시총 {PANIC_PICK_TOP}위 안, 거래대금 "
               f"{PANIC_PICK_MIN_VALUE / 1e8:,.0f}억↑), {PANIC_HOLD_DAYS}거래일 보유" if PANIC_ENABLED else "\n패닉 모드: 꺼짐")
        )
    stop = f"{STOP_LOSS_PCT}%" if USE_STOP_LOSS else "없음"
    if INTRADAY_REENTRY:
        reentry = (f"장중 10분마다 ({REENTRY_START[:2]}:{REENTRY_START[2:]}"
                   f"~{REENTRY_CUTOFF[:2]}:{REENTRY_CUTOFF[2:]}, 당일 매매 종목 제외)")
    else:
        reentry = "09:05 1회"
    if PREMARKET_SPLIT_ENTRY:
        kind = (f"눌림 대기 1건 (기준가 {PREOPEN_OFFSET_ATR:+g}ATR)" if SPLIT_TRANCHES <= 1
                else f"분할 {SPLIT_TRANCHES}건 ({PREOPEN_OFFSET_ATR:+g}ATR부터 −{SPLIT_STEP_ATR:g}ATR 간격)")
        entry = (f"{PREMARKET_ENTRY_TIME[:2]}:{PREMARKET_ENTRY_TIME[2:]} 프리마켓 판정 → {kind} | "
                 f"{SPLIT_CANCEL_AT[:2]}:{SPLIT_CANCEL_AT[2:]}까지 안 빠지면 취소 후 현재가 매수")
    else:
        entry = f"{ENTRY_TIME[:2]}:{ENTRY_TIME[2:]} 단일 진입"
    blackout = f"{STOP_BLACKOUT_UNTIL[:2]}:{STOP_BLACKOUT_UNTIL[2:]}부터"
    return (
        f"모드: {mode} | 계좌: {CANO[:4]}****{ACNT_PRDT_CD} | 거래소: {EXCG_ID_DVSN_CD}\n"
        f"일정: {entry}\n"
        f"급이탈·손절 발동: {blackout} (그 전엔 익절만)\n"
        f"유니버스: {UNIVERSE_FLAG} | 보유: {MAX_POSITIONS}종목 × {POSITION_PCT:.0f}%\n"
        f"진입: {MA_PERIOD}일선 상향돌파 (직전 {BELOW_LOOKBACK}일 중 {BELOW_MIN_DAYS}일 이상 아래)\n"
        f"청산: 익절 +{TAKE_PROFIT_PCT:g}%(장중 +{TP_INTRADAY_ATR_BUFFER:g}ATR) | "
        f"{MA_PERIOD}일선 이탈(장중 −{INTRADAY_MA5_ATR_BUFFER:g}ATR) | {MAX_HOLD_TRADING_DAYS}거래일 | 손절 {stop}\n"
        f"추세: {TREND_MA_SHORT}일선 > {TREND_MA_LONG}일선 | 랭킹: 이평선 수렴 + 변동폭 축소 우선\n"
        f"재진입: {reentry}"
    )
