"""
backtest/lab/build_marcap.py — FinanceData/marcap (KRX 전종목 일별, 상장폐지 포함) → 수정주가 행렬

  git clone --depth 1 --filter=blob:none --sparse https://github.com/FinanceData/marcap.git <dir>
  git -C <dir> sparse-checkout set --no-cone /data/marcap-2010.parquet ... /data/marcap-2026.parquet
  python -m backtest.lab.build_marcap <dir> [시작연도]

결과: data/lab/market.npz (날짜 × 종목 행렬). 레포에는 올리지 않는다 (수백 MB).

수정주가: KRX 등락(Changes)은 기준가 대비라서, 기준가 = 종가 − 등락. 전일 종가와 기준가가 다르면
그날 권리락·분할·병합 같은 조정이 있었다는 뜻이고, 그 비율을 과거 가격 전체에 곱한다.
대상: 코스피·코스닥 보통주 (코드 끝자리 0, 스팩·리츠 제외). 코넥스 제외.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "lab", "market.npz")


def main() -> int:
    src = sys.argv[1]
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2010
    frames = []
    for y in range(y0, 2100):
        p = os.path.join(src, "data", f"marcap-{y}.parquet")
        if not os.path.exists(p):
            break
        df = pd.read_parquet(p, columns=["Date", "Code", "Name", "Market", "Open", "High", "Low", "Close",
                                         "Changes", "Volume", "Amount", "Marcap"])
        frames.append(df)
        print(y, len(df), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
    df = df[df["Code"].str.len().eq(6) & df["Code"].str.endswith("0")]
    df = df[~df["Name"].str.contains("스팩|리츠|SPAC", regex=True)]
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)

    # 조정 비율: 기준가 / 전일 종가 (조정 없으면 1)
    base = df["Close"] - df["Changes"]
    prev = df.groupby("Code")["Close"].shift(1)
    f = (base / prev).where(prev.notna() & (prev > 0) & (base > 0), 1.0)
    f = f.where((f - 1).abs() > 0.001, 1.0)          # 호가 반올림 수준의 차이는 무시
    df["f"] = f
    # 과거 가격에 곱할 누적 계수: 그 날 이후 모든 조정의 곱
    df["cum"] = df.groupby("Code")["f"].transform(lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1.0))
    for c in ("Open", "High", "Low", "Close"):
        df["a" + c] = df[c].where(df[c] > 0, df["Close"]) * df["cum"]

    dates = np.array(sorted(df["Date"].unique()))
    codes = np.array(sorted(df["Code"].unique()))
    di = pd.Index(dates).get_indexer(df["Date"])
    ci = pd.Index(codes).get_indexer(df["Code"])
    T, N = len(dates), len(codes)

    def mat(col, dtype=np.float32):
        m = np.full((T, N), np.nan, dtype=dtype)
        m[di, ci] = df[col].to_numpy(dtype)
        return m

    names = df.groupby("Code")["Name"].last().reindex(codes).to_numpy()
    market = df.groupby("Code")["Market"].last().reindex(codes).to_numpy()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT, dates=np.array([pd.Timestamp(d).strftime("%Y%m%d") for d in dates]), codes=codes,
        names=names, market=market,
        o=mat("aOpen"), h=mat("aHigh"), l=mat("aLow"), c=mat("aClose"),
        raw=mat("Close"), val=mat("Amount"), cap=mat("Marcap"), vol=mat("Volume"))
    print(f"저장: {OUT} ({T}일 × {N}종목, {os.path.getsize(OUT) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
