"""
backtest/lab/setup_data.py — 백테스트 데이터(data/lab, 약 3GB)를 새 PC 에서 다시 만든다 (윈도우 · 맥 · 리눅스 공통).

    python -m backtest.lab.setup_data [marcap 받을 폴더]      (기본: 저장소 옆 ../marcap)

setup_data.sh 와 같은 일을 한다. git 이 설치돼 있어야 한다.
  1. FinanceData/marcap 2010~2026 parquet 만 받는다 (약 1GB)
  2. market.npz → market_label.npz → panic2 지표 캐시 (수 분)
marcap 은 매일 갱신되므로 다시 만들면 최근 날짜가 늘어나 결과 숫자가 조금 달라질 수 있다.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(*cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def main() -> int:
    marcap = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "marcap"))
    py = sys.executable
    run(py, "-m", "pip", "install", "-q", "-r", "requirements-lab.txt")
    if not os.path.isdir(os.path.join(marcap, ".git")):
        print(f"== marcap 받기 → {marcap}")
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "https://github.com/FinanceData/marcap.git", marcap)
        run("git", "-C", marcap, "sparse-checkout", "set", "--no-cone",
            *[f"/data/marcap-{y}.parquet" for y in range(2010, 2027)])
    else:
        print("== marcap 갱신")
        run("git", "-C", marcap, "pull", "--depth", "1", check=False)
    os.makedirs(os.path.join(ROOT, "data", "lab"), exist_ok=True)
    run(py, "-m", "backtest.lab.build_marcap", marcap, "2010")
    run(py, "-m", "backtest.lab.build_labels", marcap)
    run(py, "-m", "backtest.lab.panic2_feat")
    print("완료. 예) python -m backtest.lab.panic3.drop15_context")
    return 0


if __name__ == "__main__":
    sys.exit(main())
