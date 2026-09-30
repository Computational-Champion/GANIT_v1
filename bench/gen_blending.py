#!/usr/bin/env python3
"""Generate a synthetic multi-period refinery crude-blending LP in MPS format.

Model (always feasible and bounded):
  sets   crudes c, products p, quality specs k, periods t
  vars   buy[c,t]   crude purchase           0 <= buy <= supply[c,t]
         inv[c,t]   crude tank inventory     0 <= inv <= tank_cap[c]
         x[c,p,t]   crude c blended into p   >= 0
         pinv[p,t]  product inventory        0 <= pinv <= ptank[p]
         sell[p,t]  product sales            0 <= sell <= demand[p,t]
  rows   crude balance    inv[c,t-1] + buy[c,t] - sum_p x[c,p,t] - inv[c,t] = 0
         product balance  pinv[p,t-1] + sum_c x[c,p,t] - sell[p,t] - pinv[p,t] = 0
         quality          sum_c (q[c,k] - qmax[p,k]) x[c,p,t] <= 0
         CDU capacity     sum_{c,p} x[c,p,t] <= cap
  obj    minimise  sum crude cost + holding cost - sales revenue

usage:
  python bench/gen_blending.py --crudes 40 --products 12 --specs 4 --periods 365 -o blend_big.mps
Size: ~C*P*T columns and ~(C + P + P*K + 1)*T rows.
"""
import argparse, random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crudes", type=int, default=20)
    ap.add_argument("--products", type=int, default=8)
    ap.add_argument("--specs", type=int, default=3)
    ap.add_argument("--periods", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quad", action="store_true",
                    help="convex QP: purchase cost rises with volume (0.5*q*buy^2, supply curve)")
    ap.add_argument("-o", "--out", default="blending.mps")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    C, P, K, T = a.crudes, a.products, a.specs, a.periods

    q = [[rnd.uniform(0.05, 1.0) for _ in range(K)] for _ in range(C)]
    q[0] = [0.05] * K  # one light sweet crude keeps every spec satisfiable
    qmax = [[rnd.uniform(0.35, 0.8) for _ in range(K)] for _ in range(P)]
    cost = [rnd.uniform(40, 90) for _ in range(C)]
    price = [rnd.uniform(70, 130) for _ in range(P)]
    cap = 1000.0 * P

    rows = []           # (name, type, rhs)
    cols = defaultdict(list)  # col -> [(row, val)]
    obj = {}
    bounds = {}         # col -> upper bound
    quad = {}           # col -> diagonal Hessian entry

    def row(name, typ, rhs=0.0):
        rows.append((name, typ, rhs))
        return name

    col_order = []

    def add(col, r, v):
        if col not in cols:
            col_order.append(col)
        cols[col].append((r, v))

    for t in range(T):
        cap_r = row(f"CAP_{t}", "L", cap)
        cb = [row(f"CB_{c}_{t}", "E") for c in range(C)]
        pb = [row(f"PB_{p}_{t}", "E") for p in range(P)]
        qr = [[row(f"Q_{p}_{k}_{t}", "L") for k in range(K)] for p in range(P)]
        for c in range(C):
            b = f"BUY_{c}_{t}"
            add(b, cb[c], 1.0)
            obj[b] = cost[c] * rnd.uniform(0.9, 1.1)
            bounds[b] = rnd.uniform(200, 1500)
            if a.quad:
                quad[b] = rnd.uniform(0.01, 0.05)
            iv = f"INV_{c}_{t}"
            add(iv, cb[c], -1.0)
            if t + 1 < T:
                add(iv, f"CB_{c}_{t+1}", 1.0)
            obj[iv] = 0.2
            bounds[iv] = 5000.0
            for p in range(P):
                x = f"X_{c}_{p}_{t}"
                add(x, cb[c], -1.0)
                add(x, pb[p], 1.0)
                add(x, cap_r, 1.0)
                for k in range(K):
                    coef = q[c][k] - qmax[p][k]
                    if abs(coef) > 1e-12:
                        add(x, qr[p][k], coef)
        for p in range(P):
            s = f"SELL_{p}_{t}"
            add(s, pb[p], -1.0)
            obj[s] = -price[p] * rnd.uniform(0.95, 1.05)
            bounds[s] = rnd.uniform(300, 1200)
            pi = f"PINV_{p}_{t}"
            add(pi, pb[p], -1.0)
            if t + 1 < T:
                add(pi, f"PB_{p}_{t+1}", 1.0)
            obj[pi] = 0.5
            bounds[pi] = 3000.0

    with open(a.out, "w") as f:
        f.write(f"NAME {'QBLEND' if a.quad else 'BLEND'}_C{C}_P{P}_K{K}_T{T}\nROWS\n N COST\n")
        for name, typ, _ in rows:
            f.write(f" {typ} {name}\n")
        f.write("COLUMNS\n")
        for col in col_order:
            if obj.get(col, 0.0):
                f.write(f"    {col} COST {obj[col]:.6g}\n")
            for r, v in cols[col]:
                f.write(f"    {col} {r} {v:.6g}\n")
        f.write("RHS\n")
        for name, _, rhs in rows:
            if rhs:
                f.write(f"    RHS {name} {rhs:.6g}\n")
        f.write("BOUNDS\n")
        for col in col_order:
            if col in bounds:
                f.write(f" UP BND {col} {bounds[col]:.6g}\n")
        if quad:
            f.write("QUADOBJ\n")
            for col in col_order:
                if col in quad:
                    f.write(f"    {col} {col} {quad[col]:.6g}\n")
        f.write("ENDATA\n")
    nnz = sum(len(v) for v in cols.values())
    print(f"wrote {a.out}: {len(rows)} rows, {len(col_order)} cols, {nnz} nnz")


if __name__ == "__main__":
    main()
