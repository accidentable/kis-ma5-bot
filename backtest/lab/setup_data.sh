#!/usr/bin/env bash
#
# backtest/lab/setup_data.sh — 백테스트 데이터(data/lab, 약 3GB)를 새 PC 에서 다시 만든다. git 에는 안 올라가 있다.
#
#   bash backtest/lab/setup_data.sh [marcap 받을 폴더]      (기본 ../marcap)
#
# 1. FinanceData/marcap (KRX 전 종목 일별, 상장폐지 포함) 2010~2026 parquet 만 받는다 (약 1GB)
# 2. data/lab/market.npz        수정주가 행렬           (build_marcap)
# 3. data/lab/market_label.npz  날짜별 코스피/코스닥 구분 (build_labels)
# 4. data/lab/panic2/*.npy      지표 캐시 (panic2_feat, 약 2.8GB · 수 분)
# marcap 은 매일 갱신되므로 다시 만들면 최근 날짜가 늘어나 결과 숫자가 조금 달라질 수 있다.
# data/lab/stock_master.csv.gz (2018 업종 스냅샷) 은 저장소에 들어 있다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MARCAP="${1:-$ROOT/../marcap}"
PY="${PYTHON:-python}"

"$PY" -m pip install -q -r requirements-lab.txt

if [ ! -d "$MARCAP/.git" ]; then
  echo "== marcap 받기 → $MARCAP"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/FinanceData/marcap.git "$MARCAP"
  git -C "$MARCAP" sparse-checkout set --no-cone $(for y in $(seq 2010 2026); do printf '/data/marcap-%s.parquet ' "$y"; done)
else
  echo "== marcap 갱신"
  git -C "$MARCAP" pull --depth 1 || true
fi

mkdir -p data/lab
echo "== market.npz"
"$PY" -m backtest.lab.build_marcap "$MARCAP" 2010
echo "== market_label.npz"
"$PY" -m backtest.lab.build_labels "$MARCAP"
echo "== 지표 캐시 (panic2)"
"$PY" -m backtest.lab.panic2_feat
echo "완료. 예) python -m backtest.lab.panic3.drop15_context"
