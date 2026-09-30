#!/usr/bin/env python3
"""Generate a Markowitz portfolio QP with a factor risk model (QPS / MPS with QUADOBJ).

  minimise   0.5*gamma*( sum_i D_i x_i^2 + sum_k f_k^2 ) - mu'x
  subject to f_k - sum_i F_ik x_i = 0          (factor exposures)
             sum_i x_i = 1                      (fully invested)
             sum_{i in sector s} x_i <= 0.25    (sector limits)
             0 <= x_i <= 0.05                   (position limits)

usage: python bench/gen_portfolio.py --assets 20000 --factors 30 -o port.mps
"""
import argparse, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=int, default=2000)
    ap.add_argument("--factors", type=int, default=20)
    ap.add_argument("--sectors", type=int, default=12)
    ap.add_argument("--density", type=float, default=0.3, help="fraction of factors each asset loads on")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--out", default="portfolio.mps")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    N, K, S = a.assets, a.factors, a.sectors
    with open(a.out, "w") as f:
        f.write(f"NAME PORT_N{N}_K{K}\nROWS\n N OBJ\n E BUDGET\n")
        for k in range(K):
            f.write(f" E FAC{k}\n")
        for s in range(S):
            f.write(f" L SEC{s}\n")
        f.write("COLUMNS\n")
        diag = {}
        for i in range(N):
            x = f"X{i}"
            mu = rnd.uniform(0.02, 0.15)
            f.write(f"    {x} OBJ {-mu:.6g}\n    {x} BUDGET 1\n    {x} SEC{i % S} 1\n")
            for k in range(K):
                if rnd.random() < a.density:
                    f.write(f"    {x} FAC{k} {-rnd.gauss(0, 0.3):.6g}\n")
            diag[x] = a.gamma * rnd.uniform(0.01, 0.09)
        for k in range(K):
            y = f"F{k}"
            f.write(f"    {y} FAC{k} 1\n")
            diag[y] = a.gamma
        f.write("RHS\n    RHS BUDGET 1\n")
        for s in range(S):
            f.write(f"    RHS SEC{s} 0.25\n")
        f.write("BOUNDS\n")
        for i in range(N):
            f.write(f" UP BND X{i} 0.05\n")
        for k in range(K):
            f.write(f" FR BND F{k}\n")
        f.write("QUADOBJ\n")
        for v, d in diag.items():
            f.write(f"    {v} {v} {d:.6g}\n")
        f.write("ENDATA\n")
    print(f"wrote {a.out}: {1 + K + S} rows, {N + K} cols")


if __name__ == "__main__":
    main()
