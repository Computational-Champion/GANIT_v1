// GANIT MILP: LP-based branch-and-cut on top of the GANIT dual simplex.
//
//  * root: Gomory mixed-integer (GMI) cutting planes from the optimal tableau
//  * primal heuristics: simple rounding, fractional diving
//  * branching: reliability branching (pseudocosts initialised by strong
//    branching with an iteration limit; product score)
//  * node selection: best-bound with depth-first plunging
//  * warm start: children re-solve from the parent's optimal basis
//  * termination on relative gap (default 1e-4, same as commercial defaults)
#pragma once
#include <string>
#include <vector>

#include "ganit/lp.hpp"

namespace ganit {

struct MipOptions {
    double time_limit = 3600.0;
    double rel_gap = 1e-4;
    double abs_gap = 1e-6;
    double int_tol = 1e-6;
    double feas_tol = 1e-6;
    long node_limit = 100000000;
    int cut_rounds = 10;
    int max_cuts_per_round = 100;
    int sb_candidates = 8;      // strong-branching candidates per node
    int sb_iter_limit = 60;     // dual simplex iterations per strong-branch LP
    int reliability = 4;        // pseudocost observations before trusting them
    int verbose = 1;
    double log_interval = 2.0;
};

enum class MipStatus { Optimal, Infeasible, Unbounded, TimeLimit, NodeLimit, Numerical };
const char* to_string(MipStatus s);

struct MipResult {
    MipStatus status = MipStatus::Numerical;
    double objective = 0;     // best integer solution (original sense), NaN if none
    double bound = 0;         // best proven bound (original sense)
    double gap = 0;
    long nodes = 0;
    long lp_iterations = 0;
    int cuts = 0;
    double root_lp = 0;       // LP relaxation value before cuts
    double root_bound = 0;    // after cuts
    double seconds = 0;
    std::vector<double> x;
};

MipResult solve_mip(const LP& lp, const MipOptions& opt);

}  // namespace ganit
