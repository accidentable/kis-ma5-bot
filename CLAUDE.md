# CLAUDE.md — kis-ma5-bot (서버 작업용)

한국투자증권(KIS) Open API 자동매매 봇. 브랜치: backtest-universe-iofw7v.
목표: 다음 모의투자 대회에서 수상 조건을 채우면서 수익 내기.

## 말투 · 진행 방식
- 한국어 반말로 짧게 답해.
- 중간에 질문하지 말고 애매한 건 합리적으로 판단해서 진행해. 판단한 내용은 마지막에 정리해서 알려줘.
- 코드를 고치면 테스트 통과를 확인한 다음 커밋하고 push 해.
  테스트: python tests/test_rotation.py, tests/test_panic.py, tests/test_closebet.py, tests/test_manual.py
  (tests/test_pipeline.py 의 2개 실패는 예전부터 있던 거라 무시)
- 커밋 메시지는 영어로 무엇을 왜 바꿨는지 쓰고, 모델 이름은 넣지 마.

## 서버 환경
- 나는 `claude` 계정으로 돌고, 작업 폴더는 ~/kis-lab 이야. root 권한은 없어.
- 실제 봇은 /opt/ma5-bot (root 소유, systemd 서비스 ma5-bot). 거기는 직접 고치지 마.
- 디스크: 기본 디스크(/)는 10GB 라 거의 꽉 차 있어. 큰 파일은 전부 /data (20GB) 에 둬.
  - ~/kis-lab/data/lab 은 /data/lab 으로 가는 심볼릭 링크야. 일부러 한 거니까 복구하지 마.
  - data/lab/stock_master.csv.gz 는 skip-worktree 처리해둠. git status 에 삭제로 나와도 커밋하지 마.
  - marcap 원본은 /data/marcap.
- 메모리 7.8GB, 스왑은 /data 에 있어. 같은 서버에 다른 docker 앱이 돌고 있으니 docker 는 건드리지 마.

## 권한 · 배포 규칙
- 할 수 있는 sudo 명령은 이것뿐이야 (비밀번호 없이):
  - 배포: sudo /usr/local/bin/ma5-deploy [브랜치]  (git pull → pip → 테스트 → cli check → 재시작)
  - 봇: sudo systemctl start|stop|restart|status ma5-bot
  - 로그: sudo tail -n 50 /opt/ma5-bot/bot.log
- **배포와 봇 켜기·끄기·재시작은 내가 명시적으로 시킬 때만 해.**
- 배포는 GitHub 에 push 된 코드 기준이야. 배포 전에 push 됐는지 확인해.
- /opt/ma5-bot/.env (KIS 키 · 계좌 · DRY_RUN) 는 읽을 수도 고칠 수도 없어. 바꿔야 하면 무엇을 바꿀지 나한테 알려줘.
- 실제 주문이 나갈 수 있는 명령(cli.py buy/sell 등)은 실행하지 마.

## 대회 조건 (다음 모의투자 대회)
- 투자원금 1억, 개별 종목만 (ETF 안 됨)
- 매매금액 5억 이상 (체결 기준, 매수+매도 합계로 보고 있음)
- 매매일수 5영업일 이상, 코스피200·코스닥150 종목 5개 이상, OpenAPI 로 5회 이상 거래
- 내 목표: 4~5일에 한 번 사고팔기, 한 달 +10% 정도

## 현재 작업 (우선순위 순)
1. **모의투자 지원.** 지금 코드는 실전 전용이야 (실전 주소, TTTC TR_ID).
   - .env 에 KIS_ENV=real|mock 을 읽게 해. mock 이면 KIS_MOCK_APP_KEY / KIS_MOCK_APP_SECRET / KIS_MOCK_ACCOUNT_NO 를 써
     (서버 .env 에는 이미 이 칸이 들어가 있고, 지금은 KIS_ENV=real).
   - 주소: real https://openapi.koreainvestment.com:9443 / mock https://openapivts.koreainvestment.com:29443
   - 주문·잔고·매수가능·체결·미체결 TR_ID 를 모의용(V…)으로. KIS 공식 문서나 open-trading-api 샘플로 확인해.
     모의에서 지원 안 하는 API 는 대체하거나 명확한 에러로 막아. 유량 제한도 환경별로 다르게.
   - 토큰 캐시는 real/mock 을 분리하고, cli.py check 가 현재 모드랑 주소를 보여주게 해.
   - 가짜 응답으로 mock 테스트를 추가하고, .env.example 과 README 도 고쳐.
2. **새 전략 연구.** 백테스트 데이터가 없으면 python -m backtest.lab.setup_data /data/marcap 로 만들어.

## 지금까지 연구 결과 (자세한 건 backtest/results/lab_*.md)
- 대상 종목: 코스피 시총 100 + 코스닥 시총 150 (매일 전날 순위, 상장폐지 포함). 정의는 backtest/lab/panic3/move_timing.py 의 U.
- 52주 신고가 로테이션 (5종목, 21일 주기) + 패닉 모드 (시장 −4% 다음 날 과매도 5종목 5일 보유) 가 지금까지 제일 나았어. 둘 다 봇에 구현돼 있어.
- 하락 종목은 D+1 시가, 상승 종목은 당일 종가 매수가 가장 나았어.
- 이틀 합계 −15% 이하 종목을 D+1 시가에 사면 5일 평균 +3.8%. 하지만 수익은 시장·섹터가 같이 무너진 날에서만 나와
  (시장 −6% 이하 +9.8%, 섹터 −8% 이하 +11.3%, 종목 혼자 빠지면 −0.2%). 하한가 낀 종목은 −4% 라 제외.
  하락 추세였고 52주 고점에서 멀수록 반등이 컸어. 익절보다 5~10일 보유가 나았어.
- 칼날 잡기, 장중 눌림/반등, 섹터 강세 매수는 전부 손실이었어.
- 업종은 2018년 스냅샷만 있어 (그 뒤 상장 종목은 '미분류').
