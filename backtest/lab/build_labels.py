"""
backtest/lab/build_labels.py — 날짜별(point-in-time) 시장 구분 행렬 (0 없음 / 1 KOSPI / 2 KOSDAQ·KOSDAQ GLOBAL)

  python -m backtest.lab.build_labels <marcap 폴더>   → data/lab/market_label.npz

build_marcap 은 종목별 '마지막' 시장 구분만 남겨서, 코스닥 → 코스피 이전 종목(셀트리온 2018, 카카오 2017 등)을
과거에도 코스피로 본다. 패닉 연구에서 KOSPI/KOSDAQ 을 나누는 규칙은 이 행렬을 쓴다.
"""
import os
import sys

import numpy as np
import pandas as pd

from backtest.lab import engine as E


def main():
    src = sys.argv[1]
    z = np.load(E.NPZ, allow_pickle=True)
    dates, codes = z["dates"], z["codes"]
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: j for j, c in enumerate(codes)}
    lab = np.zeros((len(dates), len(codes)), dtype=np.int8)
    for y in range(int(dates[0][:4]), int(dates[-1][:4]) + 1):
        p = os.path.join(src, "data", f"marcap-{y}.parquet")
        df = pd.read_parquet(p, columns=["Date", "Code", "Market"])
        df = df[df["Code"].isin(ci)]
        d = df["Date"].dt.strftime("%Y%m%d").map(di)
        c = df["Code"].map(ci)
        v = df["Market"].map({"KOSPI": 1, "KOSDAQ": 2, "KOSDAQ GLOBAL": 2}).fillna(0).astype(np.int8)
        ok = d.notna() & c.notna()
        lab[d[ok].astype(int).to_numpy(), c[ok].astype(int).to_numpy()] = v[ok].to_numpy()
        print(y, int(ok.sum()), flush=True)
    out = os.path.join(os.path.dirname(E.NPZ), "market_label.npz")
    np.savez_compressed(out, label=lab)
    moved = int(((lab == 2).any(0) & (lab == 1).any(0)).sum())
    print("저장:", out, "코스닥↔코스피 이전 종목:", moved)


if __name__ == "__main__":
    main()
