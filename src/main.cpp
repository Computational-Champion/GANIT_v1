// ganit - command-line driver
//   ganit model.mps [--device gpu|cpu] [--tol 1e-4] [--time 600] [--iters N]
//                   [--quiet] [--json] [--sol solution.txt]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>

#include "ganit/lp.hpp"
#include "ganit/pdhg.hpp"
#include "simplex.hpp"
#include "mip.hpp"

using namespace ganit;

static void usage() {
    std::printf(
        "GANIT LP solver (restarted PDHG)\n"
        "usage: ganit <model.mps[.gz]> [options]\n"
        "  --method pdhg|simplex  LP algorithm (default pdhg; simplex = exact vertex)\n"
        "  --device gpu|cpu   compute device for pdhg (default: gpu if built with CUDA)\n"
        "  --tol X            relative KKT tolerance (default 1e-4)\n"
        "  --time S           time limit in seconds (default 3600)\n"
        "  --iters N          iteration limit\n"
        "  --sol FILE         write primal solution\n"
        "  --relax            solve the LP relaxation of a MILP\n"
        "  --gap X            MILP relative gap (default 1e-4)\n"
        "  --json             print one JSON summary line at the end\n"
        "  --quiet            no progress log\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string path, device, solfile;
#ifdef GANIT_HAVE_CUDA
    device = "gpu";
#else
    device = "cpu";
#endif
    Options opt;
    bool json = false;
    std::string method = "pdhg";
    bool relax = false;
    double gap = 1e-4;
    int cut_rounds = 10;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&]() -> std::string {
            if (a + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(1); }
            return argv[++a];
        };
        if (s == "--device") device = next();
        else if (s == "--method") method = next();
        else if (s == "--relax") relax = true;
        else if (s == "--gap") gap = std::stod(next());
        else if (s == "--cuts") cut_rounds = std::stoi(next());
        else if (s == "--tol") opt.tol = std::stod(next());
        else if (s == "--time") opt.time_limit = std::stod(next());
        else if (s == "--iters") opt.max_iter = std::stol(next());
        else if (s == "--sol") solfile = next();
        else if (s == "--json") json = true;
        else if (s == "--quiet") opt.verbose = 0;
        else if (s == "-h" || s == "--help") { usage(); return 0; }
        else if (!s.empty() && s[0] == '-') { std::fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
        else path = s;
    }
    if (path.empty()) { usage(); return 1; }

    try {
        auto t0 = std::chrono::steady_clock::now();
        LP lp = read_mps(path);
        double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (opt.verbose) {
            std::printf("Read %s: %d rows, %d cols, %lld nnz in %.2fs\n", lp.name.c_str(), lp.m,
                        lp.n, (long long)lp.A.nnz(), read_s);
            if (lp.num_int() > 0 && relax)
                std::printf("Note: %d integer variables relaxed (LP relaxation solved)\n", lp.num_int());
        }
        if (lp.has_q() && opt.verbose)
            std::printf("Quadratic objective: %lld Hessian nonzeros (convex QP via PDHG)\n", (long long)lp.Q.nnz());
        if (lp.has_q() && lp.num_int() > 0 && !relax) {
            std::fprintf(stderr, "MIQP is not supported yet (roadmap); use --relax for the continuous QP\n");
            return 1;
        }
        if (lp.has_q() && method == "simplex") {
            std::fprintf(stderr, "simplex handles LP only; QP uses --method pdhg\n");
            return 1;
        }
        if (lp.num_int() > 0 && !relax) {
            MipOptions mo;
            mo.time_limit = opt.time_limit;
            mo.rel_gap = gap;
            mo.cut_rounds = cut_rounds;
            mo.verbose = opt.verbose;
            MipResult r = solve_mip(lp, mo);
            if (json)
                std::printf("{\"model\":\"%s\",\"device\":\"cpu-bnc\",\"problem\":\"milp\",\"status\":\"%s\","
                            "\"pobj\":%.12e,\"dobj\":%.12e,\"gap\":%.3e,\"nodes\":%ld,\"cuts\":%d,"
                            "\"iters\":%ld,\"restarts\":0,\"read_s\":%.4f,\"setup_s\":0,\"solve_s\":%.4f,"
                            "\"rows\":%d,\"cols\":%d,\"nnz\":%lld}\n",
                            lp.name.c_str(), to_string(r.status), std::isfinite(r.objective) ? r.objective : 1e308,
                            std::isfinite(r.bound) ? r.bound : 1e308, std::isfinite(r.gap) ? r.gap : 1e308, r.nodes, r.cuts,
                            r.lp_iterations, read_s, r.seconds, lp.m, lp.n, (long long)lp.A.nnz());
            if (!solfile.empty() && !r.x.empty()) {
                std::ofstream out(solfile);
                out.precision(17);
                out << "# status " << to_string(r.status) << "\n# objective " << r.objective << "\n";
                for (int j = 0; j < lp.n; ++j) out << lp.col_names[j] << " " << r.x[j] << "\n";
            }
            return r.status == MipStatus::Optimal ? 0 : 2;
        }
        if (method == "simplex") {
            auto t1 = std::chrono::steady_clock::now();
            SimplexOptions so;
            so.verbose = opt.verbose > 1;
            DualSimplex ds(lp, so);
            LpStatus st = ds.solve(opt.max_iter, kInf, opt.time_limit);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            const char* sname = st == LpStatus::Optimal ? "OPTIMAL"
                              : st == LpStatus::Infeasible ? "PRIMAL_INFEASIBLE" : to_string(st);
            if (opt.verbose)
                std::printf("GANIT dual simplex | status %s | obj %.10e | iters %ld | %.3fs\n", sname,
                            ds.objective(), ds.iterations(), sec);
            if (json)
                std::printf("{\"model\":\"%s\",\"device\":\"cpu-simplex\",\"status\":\"%s\",\"pobj\":%.12e,"
                            "\"dobj\":%.12e,\"iters\":%ld,\"restarts\":0,\"read_s\":%.4f,\"setup_s\":0,"
                            "\"solve_s\":%.4f,\"rows\":%d,\"cols\":%d,\"nnz\":%lld}\n",
                            lp.name.c_str(), sname, ds.objective(), ds.objective(), ds.iterations(), read_s,
                            sec, lp.m, lp.n, (long long)lp.A.nnz());
            if (!solfile.empty()) {
                std::ofstream out(solfile);
                out.precision(17);
                out << "# status " << sname << "\n# objective " << ds.objective() << "\n";
                for (int j = 0; j < lp.n; ++j) out << lp.col_names[j] << " " << ds.primal(j) << "\n";
            }
            return st == LpStatus::Optimal ? 0 : 2;
        }
        Result r;
        if (device == "gpu") {
#ifdef GANIT_HAVE_CUDA
            r = solve_pdhg_gpu(lp, opt);
#else
            std::fprintf(stderr, "built without CUDA; use --device cpu\n");
            return 1;
#endif
        } else {
            r = solve_pdhg_cpu(lp, opt);
        }
        if (!solfile.empty()) {
            std::ofstream out(solfile);
            out.precision(17);
            out << "# status " << to_string(r.status) << "\n# objective " << r.pobj << "\n";
            for (int j = 0; j < lp.n; ++j) out << lp.col_names[j] << " " << r.x[j] << "\n";
        }
        if (json) {
            std::printf(
                "{\"model\":\"%s\",\"device\":\"%s\",\"problem\":\"%s\",\"status\":\"%s\",\"pobj\":%.12e,"
                "\"dobj\":%.12e,\"rel_p\":%.3e,\"rel_d\":%.3e,\"rel_gap\":%.3e,\"iters\":%ld,"
                "\"restarts\":%d,\"read_s\":%.4f,\"setup_s\":%.4f,\"solve_s\":%.4f,"
                "\"rows\":%d,\"cols\":%d,\"nnz\":%lld}\n",
                lp.name.c_str(), r.device.c_str(), lp.has_q() ? "qp" : "lp", to_string(r.status), r.pobj, r.dobj,
                r.rel_primal, r.rel_dual, r.rel_gap, r.iterations, r.restarts, read_s,
                r.setup_seconds, r.solve_seconds, lp.m, lp.n, (long long)lp.A.nnz());
        }
        return r.status == Status::Optimal ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
