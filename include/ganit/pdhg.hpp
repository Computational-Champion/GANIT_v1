// GANIT restarted PDHG (primal-dual hybrid gradient) LP solver - public API.
#pragma once
#include <string>
#include <vector>

#include "ganit/lp.hpp"

namespace ganit {

struct Options {
    double tol = 1e-4;          // relative KKT tolerance (primal, dual, gap)
    long max_iter = 10000000;
    double time_limit = 3600.0; // seconds
    int check_every = 64;       // evaluate KKT every N iterations
    int ruiz_iters = 10;
    bool pock_chambolle = true;
    int verbose = 1;
    double log_interval = 1.0;  // seconds between progress lines
    double infeas_tol = 1e-8;   // relative tolerance for infeasibility certificates
};

enum class Status { Optimal, PrimalInfeasible, IterationLimit, TimeLimit, NumericalError };
const char* to_string(Status s);

struct Result {
    Status status = Status::NumericalError;
    std::vector<double> x;  // primal solution (original space)
    std::vector<double> y;  // row duals (original space, reduced cost = c - A^T y)
    double pobj = 0, dobj = 0;
    double rel_primal = 0, rel_dual = 0, rel_gap = 0;
    long iterations = 0;
    int restarts = 0;
    double setup_seconds = 0, solve_seconds = 0;
    std::string device;
};

Result solve_pdhg_cpu(const LP& lp, const Options& opt);
#ifdef GANIT_HAVE_CUDA
Result solve_pdhg_gpu(const LP& lp, const Options& opt);
#endif

}  // namespace ganit
