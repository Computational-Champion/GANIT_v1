# GANIT — Indigenous Optimization Solver (SIH26119)

GANIT solves **LP, MILP and convex QP** problems. All of it is written from scratch: no existing solver library, and no cuSPARSE or cuBLAS. The MPS/QPS reader, scaling, sparse kernels, dual simplex, branch-and-cut and GPU PDHG are all our own code.

| Problem | Algorithm | Device | Files |
|---|---|---|---|
| **LP** (large) | Restarted PDHG, adaptive restarts, primal-weight balancing, infeasibility certificates | GPU (CUDA) or CPU | `pdhg_impl.hpp`, `backend_cuda.cu`, `backend_cpu.cpp` |
| **LP** (exact vertex) | Bounded dual simplex: explicit basis inverse with kernel re-inversion, exact dual steepest edge, Harris ratio test, bound flipping, artificial boxing | CPU | `simplex.cpp` |
| **MILP** | Branch-and-cut: root Gomory mixed-integer cuts, reliability branching (pseudocosts + strong branching), best-bound search with depth-first plunging, rounding and diving heuristics, warm-started node LPs | CPU | `mip.cpp` |
| **QP** (convex) | PDHG with quadratic gradient and the Condat–Vũ step condition τ(‖Q‖/2 + σ‖A‖²) < 1 | GPU or CPU | same as PDHG |

## Build

You need Linux or WSL2, CMake 3.18 or later, and a C++17 compiler. The GPU backend also needs CUDA 11.8+ and a GPU with compute capability 6.0 or higher.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
# CPU-only build:  add -DGANIT_CUDA=OFF
```

## Run

The problem type is detected from the file automatically.

```bash
./build/ganit model.mps                     # LP -> GPU PDHG
./build/ganit model.mps --method simplex    # LP -> exact dual simplex
./build/ganit milp.mps --gap 1e-4           # MILP -> branch-and-cut
./build/ganit milp.mps --relax              # LP relaxation of a MILP
./build/ganit qp.mps --tol 1e-6             # convex QP (QUADOBJ/QMATRIX) -> PDHG
```

Common options:

- `--device gpu|cpu`: where PDHG runs
- `--tol X`: PDHG tolerance
- `--time S`: time limit in seconds
- `--cuts N`: number of cut rounds
- `--sol file`: write the solution to a file
- `--json`: print a machine-readable summary

## Benchmark against 7 reference solvers

```bash
python3 -m pip install highspy pyscipopt            # HiGHS, SCIP
sudo apt install coinor-cbc glpk-utils              # COIN-OR CBC, GLPK
python3 -m pip install gurobipy cplex xpress        # commercial (size-limited without a license)

python3 bench/bench.py --ganit build/ganit --device gpu \
    --refs highs,cbc,glpk,scip instances/*.mps instances/mip/*.mps instances/qp/*.mps
python3 bench/bench.py --ganit build/ganit --refs gurobi,cplex,xpress instances/*.mps
```

The script gives every solver the same settings: a MIP gap of 1e-4, the same time limit, and times that exclude file reading. CBC and GLPK do not support QP, so they are reported as N/A for QP files.

### Generators for industrial-scale test models

```bash
python3 bench/gen_blending.py --crudes 60 --products 16 --specs 5 --periods 365 -o blend_big.mps          # LP
python3 bench/gen_blending.py --quad --crudes 60 --products 16 --specs 5 --periods 365 -o qblend_big.mps  # QP
python3 bench/gen_portfolio.py --assets 100000 --factors 50 -o port_big.mps                              # QP
```

## Verified results (development machine, CPU)

### Open-source references (MIP gap 1e-4 for all solvers)

| Instance | Type | GANIT (s) | HiGHS (s) | CBC (s) | GLPK (s) | SCIP (s) | Objective agreement |
|---|---|---:|---:|---:|---:|---:|---|
| afiro | LP | 0.001 | 0.002 | 1.351 | 0.187 | 0.002 | 6.8e-05 (PDHG tol 1e-4) |
| bgetam | LP | 0.038 | 0.007 | 0.008 | 0.012 | 0.003 | all agree: infeasible |
| 25fv47 | LP | 0.196 | 0.127 | 0.187 | 0.202 | 0.534 | 1.4e-04 (PDHG tol 1e-4) |
| flugpl | MILP | 0.020 | 0.122 | 0.050 | 0.009 | 0.018 | exact |
| egout | MILP | 0.033 | 0.016 | 0.020 | 0.009 | 0.010 | exact |
| lseu | MILP | 0.533 | 0.270 | 0.254 | 0.401 | 0.551 | exact |
| p0548 | MILP | 4.174 | 0.087 | 0.100 | 26.41 | 0.104 | exact |
| dcmulti | MILP | 0.335 | 1.575 | 0.881 | 0.528 | 3.729 | 3.2e-05 (within the 1e-4 gap) |
| port_small | QP | 0.018 | 0.048 | N/A | N/A | 2.470 | 1.6e-06 / 2.4e-06 |

### Commercial references (size-limited free editions)

| Instance | Type | Agreement with Gurobi 13 / CPLEX 22 / Xpress 9.9 |
|---|---|---|
| Netlib LPs (afiro, adlittle, etamacro, stair, shell, 25fv47) | LP | within the PDHG tolerance of 1e-4; all agree on bgetam infeasible |
| port_tiny (portfolio) | QP | GANIT −0.1370938992 · Gurobi −0.1370939056 · CPLEX −0.1370939050 |

### Dual simplex (exact vertex solutions) vs HiGHS

afiro, adlittle, stair, shell, 25fv47 and greenbea match to about 1e-13 (greenbea: −72,555,248.13). etamacro matches to 5e-8.

### Large models (GPU PDHG vs HiGHS interior point, tolerance 1e-4)

| Model | NNZ | GANIT GPU | HiGHS IPM |
|---|---:|---:|---:|
| blend_med | 316K | 0.21 s | 8.8 s |
| blend_big | 2.9M | 0.80 s | 172 s |

## Honest limitations (roadmap)

- **Dense basis inverse:** the dual simplex keeps an explicit basis inverse, which suits models up to a few thousand rows. A sparse LU factorization with Forrest–Tomlin updates is the next step.
- **MILP cuts:** only Gomory cuts are implemented so far. Adding MIR, cover and flow-cover cuts, conflict analysis and presolve is what separates GANIT from SCIP/HiGHS on harder MIPLIB instances. In our tests, bell5 and gesa2 hit the time limit with a gap of 0.06–0.3%.
- **MIQP:** not supported yet.
- **PDHG accuracy:** PDHG is fast to 1e-4–1e-6 relative accuracy. Crossover to an exact vertex solution is planned.
- **MILP on the GPU:** the tree search runs on the CPU. Batched GPU strong branching, where many child LPs are solved concurrently with PDHG, is the planned GPU novelty for MILP.
- **Unboundedness:** there is no dual-infeasibility (unboundedness) detection in PDHG yet.

## Tests

`tests/test_warm.cpp` checks that warm-started re-solves after branching bound changes match cold solves. Build it with:

```bash
g++ -O2 -std=c++17 -Iinclude -Isrc tests/test_warm.cpp src/lp.cpp src/mps.cpp src/scaling.cpp src/simplex.cpp -o test_warm
```
