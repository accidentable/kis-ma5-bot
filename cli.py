"""
cli.py — 로컬 실행 진입점

  python cli.py check      설정/토큰/계좌 점검
  python cli.py chatid     텔레그램 봇 확인 + 내 chat ID 조회
  python cli.py account    계좌 상품코드 진단 (APBK1271 오류 시)
  python cli.py universe   KOSPI100 유니버스 확인
  python cli.py scan       시그널 스캔 / near_high 는 오늘 순위 (주문 없음)
  python cli.py prep       개장 전 준비 1회 (near_high: 순위 계산 / closebet: 시가 매도)
  (closebet 은 entry = 종가 매수, monitor = 시가 미체결 점검, close = 리포트)
  python cli.py entry      09:05 진입 작업 1회 실행
  python cli.py monitor    장중 감시 1회 실행
  python cli.py close      마감 정리 1회 실행
  python cli.py status     보유 현황
  python cli.py serve      스케줄러 상주 실행 (람다 대신 로컬로 돌릴 때)

--force 를 붙이면 휴장일 체크를 건너뛴다 (주말 테스트용).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date


def setup_logging(level: int = logging.INFO) -> None:
    # 윈도우 콘솔 기본 코드페이지(cp949)에서 한글·기호가 깨지는 걸 막는다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def cmd_check(args) -> int:
    import config
    from core.kis import auth, quotes, trading

    print("═" * 60)
    print(config.summary())
    print("═" * 60)

    problems = config.validate()
    if problems:
        print("\n❌ 설정 문제:")
        for p in problems:
            print("   -", p)
        return 1
    print("\n✅ 설정 형식 정상")

    try:
        token = auth.get_token()
        print(f"✅ 토큰 발급 정상 (…{token[-8:]})")
    except Exception as e:
        print(f"❌ 토큰 발급 실패: {e}")
        return 1

    try:
        q = quotes.get_price("005930")
        print(f"✅ 시세 조회 정상 (삼성전자 {q['price']:,.0f}원 {q['change_pct']:+.2f}%)")
    except Exception as e:
        print(f"❌ 시세 조회 실패: {e}")
        return 1

    try:
        bal = trading.get_balance()
        print(f"✅ 잔고 조회 정상 — 순자산 {bal['net_asset']:,.0f}원 / 예수금 {bal['cash']:,.0f}원")
        for h in bal["holdings"]:
            print(f"     · {h['name']}({h['ticker']}) {h['qty']:,}주 {h['pnl_pct']:+.2f}%")
    except Exception as e:
        print(f"❌ 잔고 조회 실패: {e}")
        return 1

    try:
        opened = quotes.is_open_day(date.today())
        print(f"✅ 휴장일 조회 정상 — 오늘({date.today()}) 개장: {'예' if opened else '아니오'}")
    except Exception as e:
        print(f"⚠️ 휴장일 조회 실패(주말 판정으로 대체됨): {e}")

    if config.DRY_RUN:
        print("\n⚠️ DRY_RUN=true — 주문은 전송되지 않는다. 실매매하려면 .env 에서 false 로 바꿔라.")
    else:
        print("\n🔴 DRY_RUN=false — 실제 주문이 나간다.")
    return 0


def cmd_chatid(args) -> int:
    """
    텔레그램 봇 확인 + 내 chat ID 조회.
    봇에게 아무 메시지나 먼저 보낸 뒤 실행해야 한다 (봇은 먼저 말을 걸 수 없다).
    """
    import requests

    import config

    if not config.TELEGRAM_BOT_TOKEN:
        print("❌ .env 의 TELEGRAM_BOT_TOKEN 이 비어 있다.")
        print("   텔레그램에서 @BotFather 에게 /newbot 을 보내 토큰을 받아라.")
        return 1

    base = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

    try:
        me = requests.get(f"{base}/getMe", timeout=10).json()
    except requests.RequestException as e:
        print(f"❌ 텔레그램 연결 실패: {e}")
        return 1

    if not me.get("ok"):
        print(f"❌ 토큰이 잘못됐다: {me.get('description')}")
        return 1

    bot = me["result"]
    print(f"✅ 봇 확인: {bot.get('first_name')} (@{bot.get('username')})")

    info = requests.get(f"{base}/getWebhookInfo", timeout=10).json()
    hook_url = (info.get("result") or {}).get("url", "")
    if hook_url:
        print(f"ℹ️ 현재 등록된 웹훅: {hook_url}")
        if args.reset_webhook:
            r = requests.post(f"{base}/deleteWebhook", timeout=10).json()
            print(f"   → 해제 {'성공' if r.get('ok') else '실패: ' + str(r.get('description'))}")
        else:
            print("   웹훅이 활성이면 getUpdates 가 막힌다.")
            print("   해제하려면: python cli.py chatid --reset-webhook")
            print("   (나중에 infra/deploy.py webhook 이 새 주소로 다시 등록한다)")

    updates = requests.get(f"{base}/getUpdates", timeout=10).json()
    if not updates.get("ok"):
        print(f"❌ getUpdates 실패: {updates.get('description')}")
        return 1

    found: dict[int, str] = {}
    for u in updates.get("result", []):
        msg = u.get("message") or u.get("edited_message") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            label = chat.get("username") or chat.get("first_name") or chat.get("title") or ""
            found[int(chat["id"])] = f"{label} ({chat.get('type')})"

    if not found:
        print(f"\n⚠️ 받은 메시지가 없다. 텔레그램에서 @{bot.get('username')} 을 찾아")
        print("   아무 메시지나 보낸 다음 이 명령을 다시 실행해라.")
        return 1

    print("\n찾은 chat ID:")
    for cid, label in found.items():
        print(f"   {cid}   {label}")
    print("\n.env 에 아래 줄을 넣어라:")
    print(f"TELEGRAM_ALLOWED_CHAT_IDS={','.join(str(c) for c in found)}")
    return 0


def cmd_account(args) -> int:
    """
    계좌 진단 — 상품코드(뒤 2자리) 후보를 돌려가며 잔고 조회가 되는 조합을 찾는다.
    'APBK1271 해당계좌 정보가 없습니다' 가 뜰 때 쓴다. 조회만 하므로 주문은 나가지 않는다.
    """
    import config
    from core.kis import client, trading

    if len(config.KIS_ACCOUNT_NO) < 8:
        print(f"❌ KIS_ACCOUNT_NO 가 너무 짧다 ({len(config.KIS_ACCOUNT_NO)}자리)")
        return 1

    cano = config.CANO
    current = config.ACNT_PRDT_CD
    # 01 위탁(가장 흔함) 을 먼저, 그 다음 실제 설정값, 나머지 순
    candidates = ["01", current, "02", "03", "10", "11", "12", "21", "22", "29"]
    seen, ordered = set(), []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    print(f"종합계좌번호(앞 8자리): {cano}")
    print(f"현재 설정된 상품코드   : {current}")
    print(f"\n상품코드 {len(ordered)}개를 조회해본다 (주문 아님)...\n")

    hits = []
    for code in ordered:
        config.ACNT_PRDT_CD = code
        label = f"  {cano}-{code}"
        try:
            bal = trading.get_balance()
            hits.append(code)
            print(f"{label}  ✅ 성공 — 예수금 {bal['cash']:,.0f}원 / 순자산 {bal['net_asset']:,.0f}원 "
                  f"/ 보유 {len(bal['holdings'])}종목")
        except client.KisError as e:
            print(f"{label}  ✗ {e.msg_cd} {e.msg}")
        except Exception as e:
            print(f"{label}  ✗ {type(e).__name__}: {e}")

    config.ACNT_PRDT_CD = current

    print()
    if not hits:
        print("❌ 맞는 상품코드를 못 찾았다. 아래를 확인해라:")
        print("   1) 한투 앱/홈페이지에서 계좌번호 전체(########-##)를 다시 확인")
        print("   2) 그 계좌가 APP_KEY 를 발급받을 때 지정한 계좌와 같은지")
        print("   3) 실전 앱키인지 (모의투자 키는 실전 URL에서 계좌를 못 찾는다)")
        return 1

    if hits[0] == current and len(hits) == 1:
        print(f"✅ 현재 설정({current})이 맞다. 다른 원인을 봐야 한다.")
        return 0

    print(f"✅ 사용 가능한 상품코드: {', '.join(hits)}")
    print(f"\n.env 의 KIS_ACCOUNT_NO 를 아래로 바꿔라:")
    print(f"KIS_ACCOUNT_NO={cano}{hits[0]}")
    return 0


def cmd_universe(args) -> int:
    import config
    from core import universe
    if config.STRATEGY == "near_high":
        stocks = universe.get_large_universe(config.NH_UNIVERSE_TOP, force=args.force)
        print(f"코스피+코스닥 시총 상위 {len(stocks)}종목")
        for i, s in enumerate(stocks, 1):
            print(f"{i:3d}. {s['ticker']} {s['name']:<16} {s.get('market', ''):<6} 시총 {s['marcap']:>12,.0f}억")
        return 0
    stocks = universe.get_universe(force=args.force)
    print(f"KOSPI100 매매대상 {len(stocks)}종목")
    for i, s in enumerate(stocks, 1):
        print(f"{i:3d}. {s['ticker']} {s['name']:<16} 시총 {s['marcap']:>12,.0f}억")
    return 0


def cmd_scan(args) -> int:
    """수동 스캔. 항상 일봉을 새로 받는다 — 당일 캐시가 옛 코드로 만들어졌을 수 있다."""
    import config
    if config.STRATEGY == "closebet":
        from jobs import closebet
        cands, stats = closebet.candidates()
        print(closebet.regime_on()[1])
        print(f"거래대금 순위 {stats.get('rank')} → 후보 {len(cands)} | 탈락 {stats.get('rejects')}")
        for c in cands:
            print(f"  {c['name']}({c['ticker']}) {c['price']:,.0f}원 {c['change_pct']:+.1f}% IBS {c['ibs']:.2f} 거래대금 {c['value'] / 1e8:,.0f}억")
        return 0
    if config.STRATEGY == "near_high":
        from jobs import rotation
        ranked, stats = rotation.build_ranking()
        print(rotation.format_ranking(ranked, stats, limit=20))
        return 0
    from jobs import scan as scan_job
    ranked, stats = scan_job.scan(use_cache=False)
    print(scan_job.format_result(ranked, stats, limit=20))
    return 0


def cmd_prep(args) -> int:
    import config
    if config.STRATEGY == "closebet":
        from jobs import closebet
        print(closebet.sell_open(force=args.force))
        return 0
    if config.STRATEGY == "near_high":
        from jobs import rotation
        print(rotation.prep(force=args.force))
    else:
        from jobs import prep
        print(prep.run(force=args.force))
    return 0


def cmd_entry(args) -> int:
    import config
    if config.STRATEGY == "closebet":
        from jobs import closebet
        print(closebet.buy_close(force=args.force))
        return 0
    if config.STRATEGY == "near_high":
        from jobs import rotation
        print(rotation.entry(force=args.force))
        return 0
    from jobs import entry
    print(entry.run(force=args.force))
    return 0


def cmd_monitor(args) -> int:
    import config
    if config.STRATEGY == "closebet":
        from jobs import closebet
        print(closebet.check_open(force=args.force))
        return 0
    if config.STRATEGY == "near_high":
        from jobs import rotation
        print(rotation.monitor(force=args.force))
        return 0
    from jobs import monitor
    print(monitor.run(force=args.force))
    return 0


def cmd_close(args) -> int:
    import config
    if config.STRATEGY == "closebet":
        from jobs import closebet
        print(closebet.report(force=args.force))
        return 0
    if config.STRATEGY == "near_high":
        from jobs import rotation
        print(rotation.close(force=args.force))
        return 0
    from jobs import close
    print(close.run(force=args.force))
    return 0


def cmd_status(args) -> int:
    from core import commands
    import config
    cid = config.TELEGRAM_ALLOWED_CHAT_IDS[0] if config.TELEGRAM_ALLOWED_CHAT_IDS else 0
    print(commands._cmd_status() if not cid else commands.handle("/status", cid))
    return 0


def cmd_serve(args) -> int:
    """
    상주 실행 — 스케줄러 + 텔레그램 롱폴링.
    VM(네이버 클라우드 등) 한 대에서 이것만 띄우면 봇 전체가 돈다.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    import config
    from core import notify, poller

    log = logging.getLogger("serve")
    sched = BlockingScheduler(timezone="Asia/Seoul")

    def _wrap(name, fn):
        def _run():
            try:
                log.info("작업 시작: %s", name)
                fn()
            except Exception as e:
                log.exception("작업 실패: %s", name)
                notify.send_error(f"{name} 작업 실패", e)
        return _run

    def _hm(hhmm: str) -> tuple[int, int]:
        return int(hhmm[:2]), int(hhmm[2:])

    if config.STRATEGY == "closebet":
        from jobs import closebet
        for job_id, name, fn, hhmm in (("cb_sell", "시가 매도", closebet.sell_open, config.CB_SELL_TIME),
                                        ("cb_check", "시가 미체결 점검", closebet.check_open, config.CB_CHECK_TIME),
                                        ("cb_buy", "종가 매수", closebet.buy_close, config.CB_BUY_TIME),
                                        ("cb_report", "마감 리포트", closebet.report, "1540")):
            h, mi = _hm(hhmm)
            sched.add_job(_wrap(name, fn), CronTrigger(day_of_week="mon-fri", hour=h, minute=mi),
                          id=job_id, replace_existing=True)
    elif config.STRATEGY == "near_high":
        from jobs import rotation
        for job_id, name, fn, hhmm in (("prep", "순위 계산", rotation.prep, config.NH_PREP_TIME),
                                        ("entry", "교체 매매", rotation.entry, config.NH_ENTRY_TIME)):
            h, mi = _hm(hhmm)
            sched.add_job(_wrap(name, fn), CronTrigger(day_of_week="mon-fri", hour=h, minute=mi),
                          id=job_id, replace_existing=True)
        sched.add_job(_wrap("감시", rotation.monitor),
                      CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/10"),
                      id="monitor", replace_existing=True)
        sched.add_job(_wrap("마감", rotation.close), CronTrigger(day_of_week="mon-fri", hour=15, minute=15),
                      id="close", replace_existing=True)
    else:
        _schedule_ma5(sched, _wrap, _hm)

    # 텔레그램 명령 수신 (롱폴링) — 공개 엔드포인트가 필요 없다
    notify.set_commands()
    _, stop_poller = poller.start_thread()

    notify.send("🤖 봇 시작\n" + config.summary())
    log.info("스케줄러 시작 — Ctrl+C 로 종료")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("종료 신호 수신")
    finally:
        stop_poller.set()
        notify.send("🛑 봇 종료")
    return 0


def _schedule_ma5(sched, _wrap, _hm) -> None:
    """예전 MA5 돌파 전략 일정 (STRATEGY=ma5)."""
    from apscheduler.triggers.cron import CronTrigger

    import config
    from jobs import close, entry, monitor, prep

    ph, pm = _hm(config.PREP_TIME)
    sched.add_job(_wrap("준비", prep.run), CronTrigger(day_of_week="mon-fri", hour=ph, minute=pm),
                  id="prep", replace_existing=True)
    if config.PREMARKET_SPLIT_ENTRY:
        qh, qm = _hm(config.PREMARKET_ENTRY_TIME)
        sched.add_job(_wrap("프리마켓 분할진입", prep.premarket_entry),
                      CronTrigger(day_of_week="mon-fri", hour=qh, minute=qm),
                      id="premarket", replace_existing=True)
    eh, em = _hm(config.ENTRY_TIME)
    sched.add_job(_wrap("진입", entry.run), CronTrigger(day_of_week="mon-fri", hour=eh, minute=em),
                  id="entry", replace_existing=True)
    sched.add_job(_wrap("감시", monitor.run),
                  CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/10"),
                  id="monitor", replace_existing=True)
    sched.add_job(_wrap("마감", close.run), CronTrigger(day_of_week="mon-fri", hour=15, minute=15),
                  id="close", replace_existing=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="한투 OpenAPI 자동매매 봇 (STRATEGY=near_high | closebet | ma5)")
    parser.add_argument("--force", action="store_true", help="휴장일 체크를 건너뛴다")
    parser.add_argument("--reset-webhook", action="store_true",
                        help="chatid 실행 시 기존 텔레그램 웹훅을 해제한다")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn, help_text in [
        ("check", cmd_check, "설정/토큰/계좌 점검"),
        ("chatid", cmd_chatid, "텔레그램 봇 확인 + 내 chat ID 조회"),
        ("account", cmd_account, "계좌 상품코드 진단"),
        ("universe", cmd_universe, "유니버스 확인"),
        ("scan", cmd_scan, "시그널 스캔 / 순위 (주문 없음)"),
        ("prep", cmd_prep, "개장 전 준비 1회"),
        ("entry", cmd_entry, "진입 작업 1회"),
        ("monitor", cmd_monitor, "장중 감시 1회"),
        ("close", cmd_close, "마감 정리 1회"),
        ("status", cmd_status, "보유 현황"),
        ("serve", cmd_serve, "스케줄러 상주 실행"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
