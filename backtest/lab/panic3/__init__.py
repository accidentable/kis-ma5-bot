"""
backtest/lab/panic3 — 패닉 매수 후속 탐색 (사전 등록 연구 panic2 이후, 결과를 보면서 한 탐색이라 검증 강도는 낮다)

  since2020.py            2020 이후만 본 panic2 격자
  wide_universe.py        종목 범위 확장 (시총 500/1000, 코스피+코스닥 유동, 코스닥)
  dip_selection.py        시장 −2~−3% · −3~−4% · −4%↓ 날 어떤 종목이 5 · 10일 더 오르나 (특징 19개, 시총 1000위)
  capital_1e8_slots.py    1억 모의투자 · 종목 1~5개 대회 시뮬레이션 (봇 패닉 모드의 근거)
  sameday_vs_nextopen.py  당일 종가 진입 vs 다음 날 시가 진입
  intraday_path.py        급락일 전후 가격 경로 · 지정가 진입

실행: PYTHONPATH=. python backtest/lab/panic3/<파일> (지표 캐시 data/lab/panic2 필요: python -m backtest.lab.panic2_feat)
"""
