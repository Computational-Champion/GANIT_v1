#!/usr/bin/env python3
"""Benchmark GANIT against reference solvers:
   open source: HiGHS, COIN-OR CBC, GLPK, SCIP      commercial: Gurobi, CPLEX, FICO Xpress

Problem type is detected per file from GANIT's output:
   lp   -> every solver solves the LP
   milp -> every solver solves the full MILP (GANIT branch-and-cut); use --relax for LP relaxations
   qp   -> convex QP (CBC and GLPK do not support QP and are reported as N/A)

usage:
  python3 bench/bench.py --ganit build/ganit --device gpu --tol 1e-4 \
      --refs highs,cbc,glpk,scip instances/*.mps instances/mip/*.mps blend_*.mps

  --method barrier|simplex|default    algorithm for the reference solvers
                                      (barrier = interior point, the fastest
                                       exact method on large LPs)
  --crossover on|off                  barrier crossover (off = faster, non-vertex)
  --threads N                         threads for reference solvers (0 = all)

Open-source references:
  pip install highspy pyscipopt
  sudo apt install coinor-cbc glpk-utils        (provides the cbc and glpsol programs)
Commercial references (need a license for large models):
  pip install gurobipy cplex xpress
  Gurobi: free academic license  https://www.gurobi.com/academia/
  CPLEX : IBM Academic Initiative (free)
  Xpress: FICO academic license; pip community edition is size-limited
Solver-reported times exclude file reading for every solver, including GANIT.
Writes a Markdown table to stdout and results.csv.
"""
import argparse, csv, json, math, os, subprocess, sys, time

NAN = float("nan")


# --------------------------------------------------------------------------- GANIT
def run_ganit(binary, path, device, tol, tlim, relax=False, gap=1e-4):
    cmd = [binary, path, "--device", device, "--tol", str(tol), "--time", str(tlim), "--json", "--quiet",
           "--gap", str(gap)]
    if relax:
        cmd.append("--relax")
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            d = json.loads(line)
            obj = d["pobj"] if abs(d["pobj"]) < 1e300 else NAN
            return {"status": d["status"], "obj": obj, "time": d["solve_s"] + d["setup_s"],
                    "iters": d.get("nodes", d["iters"]), "rows": d["rows"], "cols": d["cols"], "nnz": d["nnz"],
                    "problem": d.get("problem", "lp")}
    return {"status": "CRASH", "obj": NAN, "time": NAN, "iters": "", "rows": "", "cols": "", "nnz": "",
            "problem": "lp", "err": (p.stderr or "")[-300:]}


# --------------------------------------------------------------------------- HiGHS
def run_highs(path, a):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(a.time))
    if a.ptype == "lp":
        h.setOptionValue("solve_relaxation", True)
    if a.ptype == "milp":
        h.setOptionValue("mip_rel_gap", a.gap)
    elif a.ptype == "qp":
        pass  # HiGHS picks its QP solver
    elif a.method == "barrier":
        h.setOptionValue("solver", "ipm")
        h.setOptionValue("run_crossover", "on" if a.crossover == "on" else "off")
    elif a.method == "simplex":
        h.setOptionValue("solver", "simplex")
    if a.threads:
        h.setOptionValue("threads", a.threads)
    h.readModel(path)
    t0 = time.time()
    h.run()
    t = time.time() - t0
    st = h.modelStatusToString(h.getModelStatus())
    return {"status": st, "obj": h.getInfo().objective_function_value if st == "Optimal" else NAN, "time": t}


# --------------------------------------------------------------------------- Gurobi
def run_gurobi(path, a):
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.read(path, env=env)
    if m.IsMIP and a.ptype != "milp":
        m = m.relax()
    m.Params.TimeLimit = a.time
    if a.ptype == "milp":
        m.Params.MIPGap = a.gap
    elif a.method == "barrier":
        m.Params.Method = 2
        m.Params.Crossover = -1 if a.crossover == "on" else 0
    elif a.method == "simplex":
        m.Params.Method = 1  # dual simplex
    if a.threads:
        m.Params.Threads = a.threads
    m.optimize()
    names = {2: "Optimal", 3: "Infeasible", 4: "Inf_or_Unbd", 5: "Unbounded", 9: "TimeLimit", 13: "Suboptimal"}
    st = names.get(m.Status, f"status{m.Status}")
    obj = m.ObjVal if m.Status == 2 else NAN
    return {"status": st, "obj": obj, "time": m.Runtime}


# --------------------------------------------------------------------------- CPLEX
def run_cplex(path, a):
    import cplex
    c = cplex.Cplex()
    for s in (c.set_log_stream, c.set_results_stream, c.set_warning_stream, c.set_error_stream):
        s(None)
    c.read(path)
    pt = c.get_problem_type()
    if a.ptype == "lp" and pt != c.problem_type.LP:
        c.set_problem_type(c.problem_type.LP)
    elif a.ptype == "qp" and pt not in (c.problem_type.QP,):
        c.set_problem_type(c.problem_type.QP)
    c.parameters.timelimit.set(a.time)
    if a.ptype == "milp":
        c.parameters.mip.tolerances.mipgap.set(a.gap)
    elif a.ptype == "qp":
        pass
    elif a.method == "barrier":
        c.parameters.lpmethod.set(c.parameters.lpmethod.values.barrier)
        if a.crossover == "off":
            c.parameters.barrier.crossover.set(c.parameters.barrier.crossover.values.none)
    elif a.method == "simplex":
        c.parameters.lpmethod.set(c.parameters.lpmethod.values.dual)
    if a.threads:
        c.parameters.threads.set(a.threads)
    t0 = c.get_time()
    c.solve()
    t = c.get_time() - t0
    s = c.solution
    code = s.get_status()
    optimal = code in (s.status.optimal, s.status.optimal_face_unbounded,
                       s.status.MIP_optimal, s.status.optimal_tolerance) or \
        (a.crossover == "off" and code == getattr(s.status, "num_best", -1))
    if optimal:
        st = "Optimal"
    elif code in (s.status.infeasible, s.status.MIP_infeasible, getattr(s.status, "optimal_infeasible", -1)):
        st = "Infeasible"
    else:
        st = s.get_status_string()
    return {"status": st, "obj": s.get_objective_value() if optimal else NAN, "time": t}


# --------------------------------------------------------------------------- Xpress
def run_xpress(path, a):
    import warnings
    import xpress as xp
    warnings.filterwarnings("ignore", module="xpress")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    lic = os.environ.get("XPRESS_LICENSE")  # path to a full license file, if you have one
    if not getattr(run_xpress, "_init", False):
        try:
            if lic:
                xp.init(lic)
        except Exception:
            pass
        run_xpress._init = True
    p = xp.problem()
    p.setControl("outputlog", 0)
    (p.readProb if hasattr(p, "readProb") else p.read)(path)
    p.setControl("timelimit", int(max(1, a.time)))
    if a.threads:
        p.setControl("threads", a.threads)
    if a.ptype == "milp":
        p.setControl("miprelstop", a.gap)
        t0 = time.time()
        (p.mipOptimize if hasattr(p, "mipOptimize") else p.mipoptimize)()
        t = time.time() - t0
        ms = p.getAttrib("mipstatus")
        names = {6: "Optimal", 5: "Infeasible", 4: "Feasible", 3: "Unfinished"}
        st = names.get(ms, f"mipstatus{ms}")
        return {"status": st, "obj": p.getAttrib("mipobjval") if ms == 6 else NAN, "time": t}
    alg = ""
    if a.method == "barrier" and a.ptype == "lp":
        alg = "b"
        p.setControl("crossover", 1 if a.crossover == "on" else 0)
    elif a.method == "simplex":
        alg = "d"
    t0 = time.time()
    (p.lpOptimize if hasattr(p, "lpOptimize") else p.lpoptimize)(alg)  # LP (relaxation for MIP)
    t = time.time() - t0
    lpstat = p.getAttrib("lpstatus")
    names = {1: "Optimal", 2: "Infeasible", 3: "Cutoff", 4: "Unfinished", 5: "Unbounded", 6: "CutoffInDual"}
    st = names.get(lpstat, f"lpstatus{lpstat}")
    return {"status": st, "obj": p.getAttrib("lpobjval") if lpstat == 1 else NAN, "time": t}


# --------------------------------------------------------------------------- COIN-OR CBC (CLI)
def run_cbc(path, a):
    import re, shutil
    exe = os.environ.get("CBC_PATH") or shutil.which("cbc")
    if not exe:
        raise RuntimeError("cbc not found (sudo apt install coinor-cbc, or set CBC_PATH)")
    if a.ptype == "qp":
        return {"status": "N/A", "obj": NAN, "time": NAN}
    cmd = [exe, path, "-sec", str(int(a.time)), "-ratio", str(a.gap)]
    if a.threads:
        cmd += ["-threads", str(a.threads)]
    cmd += (["-solve"] if a.ptype == "milp" else ["-initialSolve"]) + ["-quit"]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.time + 60).stdout
    t = time.time() - t0
    low = out.lower()
    if "infeasible" in low and ("problem proven infeasible" in low or "primal infeasible" in low or "result - problem infeasible" in low):
        return {"status": "Infeasible", "obj": NAN, "time": t}
    m = re.search(r"Objective value:\s*([-+0-9.eE]+)", out) or re.search(r"objective value\s*([-+0-9.eE]+)", out)
    optimal = ("Optimal solution found" in out) or ("Optimal - objective value" in out)
    if optimal and m:
        return {"status": "Optimal", "obj": float(m.group(1)), "time": t}
    status = "TimeLimit" if "time limit" in low or "stopped on time" in low else "Unknown"
    return {"status": status, "obj": NAN, "time": t}


# --------------------------------------------------------------------------- GLPK (CLI)
def run_glpk(path, a):
    import re, shutil, tempfile
    exe = os.environ.get("GLPSOL_PATH") or shutil.which("glpsol")
    if not exe:
        raise RuntimeError("glpsol not found (sudo apt install glpk-utils, or set GLPSOL_PATH)")
    if a.ptype == "qp":
        return {"status": "N/A", "obj": NAN, "time": NAN}
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as fh:
        outfile = fh.name
    cmd = [exe, "--freemps", path, "--tmlim", str(int(a.time)), "-o", outfile]
    if a.ptype == "milp":
        cmd += ["--mipgap", str(a.gap)]
    else:
        cmd += ["--nomip"]
    t0 = time.time()
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=a.time + 60)
    t = time.time() - t0
    log = (run.stdout or "").upper()
    txt = open(outfile).read() if os.path.exists(outfile) else ""
    os.unlink(outfile)
    ms = re.search(r"Status:\s+(.+)", txt)
    mo = re.search(r"Objective:\s+\S+\s*=\s*([-+0-9.eE]+)", txt)
    status_txt = ms.group(1).strip() if ms else "Unknown"
    if "INFEAS" in status_txt or "EMPTY" in status_txt or "NO PRIMAL FEASIBLE" in log \
            or "PROBLEM HAS NO" in log:
        return {"status": "Infeasible", "obj": NAN, "time": t}
    # GLPK reports INTEGER NON-OPTIMAL when it stops because the requested MIP gap was reached
    gap_reached = status_txt == "INTEGER NON-OPTIMAL" and "RELATIVE MIP GAP TOLERANCE REACHED" in log
    if (status_txt in ("OPTIMAL", "INTEGER OPTIMAL") or gap_reached) and mo:
        return {"status": "Optimal", "obj": float(mo.group(1)), "time": t}
    return {"status": status_txt.title()[:12], "obj": NAN, "time": t}


# --------------------------------------------------------------------------- SCIP
def run_scip(path, a):
    import pyscipopt
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(path)
    m.setParam("limits/time", float(a.time))
    if a.ptype == "milp":
        m.setParam("limits/gap", a.gap)
    elif a.ptype == "lp":
        for v in m.getVars():
            if v.vtype() != "CONTINUOUS":
                m.chgVarType(v, "C")
    if a.threads:
        try:
            m.setParam("parallel/maxnthreads", a.threads)
        except Exception:
            pass
    m.optimize()
    st = m.getStatus()
    names = {"optimal": "Optimal", "gaplimit": "Optimal", "infeasible": "Infeasible",
             "unbounded": "Unbounded", "inforunbd": "Inf_or_Unbd", "timelimit": "TimeLimit"}
    status = names.get(st, st)
    obj = m.getObjVal() if status == "Optimal" else NAN
    return {"status": status, "obj": obj, "time": m.getSolvingTime()}


REFS = {"highs": run_highs, "cbc": run_cbc, "glpk": run_glpk, "scip": run_scip,
        "gurobi": run_gurobi, "cplex": run_cplex, "xpress": run_xpress}
LABEL = {"highs": "HiGHS", "cbc": "CBC", "glpk": "GLPK", "scip": "SCIP",
         "gurobi": "Gurobi", "cplex": "CPLEX", "xpress": "Xpress"}


def safe(fn, path, a):
    try:
        return fn(path, a)
    except Exception as e:  # license limits, missing packages, ...
        msg = str(e).replace("\n", " ")
        return {"status": "ERR", "obj": NAN, "time": NAN, "err": msg[:120]}


def rel_diff(x, ref):
    if not (isinstance(x, float) and isinstance(ref, float)) or math.isnan(x) or math.isnan(ref):
        return NAN
    return abs(x - ref) / (1.0 + abs(ref))


def fmt_t(t):
    return "—" if t != t else (f"{t:.3f}" if t < 100 else f"{t:.1f}")


def fmt_e(v):
    return "—" if v != v else f"{v:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ganit", default="build/ganit")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--refs", default="highs,cbc,glpk,scip",
                    help="comma list: highs,cbc,glpk,scip,gurobi,cplex,xpress")
    ap.add_argument("--relax", action="store_true", help="solve LP relaxations of MILP files")
    ap.add_argument("--gap", type=float, default=1e-4, help="relative MIP gap for all solvers")
    ap.add_argument("--method", default="barrier", choices=["barrier", "simplex", "default"])
    ap.add_argument("--crossover", default="on", choices=["on", "off"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--csv", default="results.csv")
    # backwards compatibility with the first version of this script
    ap.add_argument("--highs-solver", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.highs_solver in ("ipm",):
        a.method = "barrier"
    elif a.highs_solver == "simplex":
        a.method = "simplex"
    refs = [r.strip().lower() for r in a.refs.split(",") if r.strip()]
    for r in refs:
        if r not in REFS:
            sys.exit(f"unknown reference solver '{r}' (choose from {', '.join(REFS)})")

    rows = []
    for f in a.files:
        name = os.path.basename(f).split(".")[0]
        g = run_ganit(a.ganit, f, a.device, a.tol, a.time, a.relax, a.gap)
        a.ptype = g["problem"]
        rec = {"instance": name, "type": a.ptype.upper(), "rows": g["rows"], "cols": g["cols"], "nnz": g["nnz"],
               "ganit_status": g["status"], "ganit_obj": g["obj"], "ganit_time": g["time"],
               "ganit_iters": g["iters"]}
        msg = [f"{name:12s} {a.ptype:4s} GANIT {g['status']:>17s} {fmt_t(g['time']):>9s}s"]
        for r in refs:
            res = safe(REFS[r], f, a)
            rec[f"{r}_status"] = res["status"]
            rec[f"{r}_obj"] = res["obj"]
            rec[f"{r}_time"] = res["time"]
            rec[f"{r}_relobj"] = rel_diff(g["obj"], res["obj"])
            if "err" in res:
                rec[f"{r}_err"] = res["err"]
            msg.append(f"{LABEL[r]} {res['status']:>10s} {fmt_t(res['time']):>9s}s")
            if "err" in res:
                msg.append(f"[{res['err'][:60]}]")
        print(" | ".join(msg), file=sys.stderr)
        rows.append(rec)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- time table
    hdr = ["Instance", "Type", "NNZ", "GANIT (s)"] + [f"{LABEL[r]} (s)" for r in refs]
    print(f"\n**Solve time** (GANIT: LP/QP via PDHG on {a.device} at tol {a.tol:g}, MILP via branch-and-cut; "
          f"MIP gap {a.gap:g} for all solvers)\n")
    print("| " + " | ".join(hdr) + " |")
    print("|---|---|---:|" + "---:|" * (1 + len(refs)))
    for r in rows:
        cells = [r["instance"], r["type"], str(r["nnz"]), fmt_t(r["ganit_time"])]
        for s in refs:
            st = r[f"{s}_status"]
            cells.append(fmt_t(r[f"{s}_time"]) if st in ("Optimal", "Infeasible") else st)
            
        print("| " + " | ".join(cells) + " |")

    # ---- accuracy table
    print("\n**Objective agreement** (|obj_GANIT − obj_ref| / (1 + |obj_ref|))\n")
    hdr = ["Instance", "Type", "GANIT status", "GANIT objective"] + [f"vs {LABEL[r]}" for r in refs]
    print("| " + " | ".join(hdr) + " |")
    print("|---|---|---|---:|" + "---:|" * len(refs))
    for r in rows:
        infeas = r["ganit_status"] in ("PRIMAL_INFEASIBLE", "INFEASIBLE")
        gobj = r["ganit_obj"]
        cells = [r["instance"], r["type"], r["ganit_status"], "—" if infeas or gobj != gobj else f"{gobj:.10g}"]
        for s in refs:
            st = r[f"{s}_status"]
            if infeas:
                cells.append("agree (infeasible)" if st in ("Infeasible", "Inf_or_Unbd") else f"ref: {st}")
            else:
                cells.append(fmt_e(r[f"{s}_relobj"]) if st == "Optimal" else st)
        print("| " + " | ".join(cells) + " |")

    solved = sum(1 for r in rows if r["ganit_status"] in ("OPTIMAL", "PRIMAL_INFEASIBLE", "INFEASIBLE"))
    print(f"\nGANIT correctly terminated on {solved}/{len(rows)} instances (tolerance {a.tol:g}).")
    errs = {k: v for r in rows for k, v in r.items() if k.endswith("_err")}
    if errs:
        print("Some reference runs failed (often license size limits); see results.csv.")


if __name__ == "__main__":
    main()
