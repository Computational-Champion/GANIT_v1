// Restarted-average PDHG for LP, written once and instantiated for each
// compute backend (CPU / CUDA). The backend only provides fused vector
// kernels; every algorithmic decision lives here.
//
// Iteration (scaled space, y = row duals, reduced cost c - A^T y):
//   x+   = proj_[l,u]( x - tau (c - A^T y) )
//   xbar = 2 x+ - x
//   q    = y - sigma A xbar
//   y+   = q + clamp(-q, sigma*lo, sigma*hi)      (prox of the row-bound set)
// with tau = eta/w, sigma = eta*w, eta < 1/||A||_2, w = primal weight.
//
// Convex QP (objective c'x + 0.5 x'Qx): the primal step uses the gradient
// c + Qx - A^T y (linearised / Condat-Vu PDHG), with the step condition
//   tau * (||Q||/2 + sigma ||A||^2) < 1.
// The dual objective becomes  -0.5 x'Qx + (row terms) + (bound terms of z),
// z = c + Qx - A^T y.
//
// Restarts follow the adaptive restart scheme of PDLP (Applegate et al. 2021):
// candidate = current or average iterate (lower KKT error); restart on
// sufficient decay, necessary decay without progress, or artificial restart.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include "ganit/pdhg.hpp"
#include "scaling.hpp"

namespace ganit {

struct KKT {
    double pobj = 0, dobj = 0, rel_p = 0, rel_d = 0, rel_gap = 0, err = 0;
    bool finite() const {
        return std::isfinite(pobj) && std::isfinite(dobj) && std::isfinite(rel_p) &&
               std::isfinite(rel_d);
    }
    bool converged(double tol) const { return rel_p <= tol && rel_d <= tol && rel_gap <= tol; }
};

template <class Backend>
Result run_pdhg(const LP& lp, const Options& opt, const char* device_name) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    auto elapsed = [&](Clock::time_point t0) {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    };

    Result res;
    res.device = device_name;

    ScaledLP s = scale_lp(lp, opt.ruiz_iters, opt.pock_chambolle);
    const double normA = estimate_norm(s);
    const double eta = 0.998 / normA;
    double w;
    {
        double cn = norm2(s.c), bn = finite_bound_norm(s.lo, s.hi);
        w = (cn > 1e-10 && bn > 1e-10) ? cn / bn : 1.0;
    }

    Backend be(s);
    using Vec = typename Backend::Vec;
    const int m = s.m, n = s.n;

    std::vector<double> x0h(n);
    for (int j = 0; j < n; ++j) x0h[j] = std::min(std::max(0.0, s.l[j]), s.u[j]);

    Vec x = be.from_host(x0h), xn = be.zeros(n), xbar = be.zeros(n);
    Vec y = be.zeros(m), yn = be.zeros(m);
    Vec aty = be.zeros(n), axbar = be.zeros(m);
    Vec sumx = be.zeros(n), sumy = be.zeros(m), xa = be.zeros(n), ya = be.zeros(m);
    Vec xr = be.from_host(x0h), yr = be.zeros(m);          // last restart point
    Vec ax_tmp = be.zeros(m), aty_tmp = be.zeros(n);       // for KKT evaluation
    Vec ray = be.zeros(m);                                 // dual ray candidate
    const bool hasq = s.has_q();
    Vec qx = be.zeros(hasq ? n : 0), qx_tmp = be.zeros(hasq ? n : 0);
    auto step_sizes = [&](double w_, double& tau_, double& sigma_) {
        sigma_ = eta * w_;
        tau_ = eta / w_;
        if (hasq) tau_ = std::min(tau_, 0.998 / (0.5 * s.normQ + sigma_ * normA * normA));
    };
    const double obj_scale = s.cscale * s.bscale;

    // Primal infeasibility test (Farkas): dual ray r = y - y_restart with
    // A^T r inside the homogeneous dual cone and positive dual objective.
    auto primal_infeasible = [&]() {
        be.sub(y, yr, ray);
        be.spmv_AT(ray, aty_tmp);
        double rd2 = 0, vobj = 0;
        be.dual_res(aty_tmp, nullptr, rd2, vobj, false);
        double dray = (be.row_dual_obj(ray) + vobj) * obj_scale;
        if (!(dray > 0)) return false;
        return std::sqrt(rd2) * (1.0 + s.bnorm) / dray <= opt.infeas_tol;
    };

    auto eval = [&](const Vec& xv, const Vec& yv) {
        KKT k;
        be.spmv_A(xv, ax_tmp);
        double rp2 = be.primal_res2(ax_tmp);
        be.spmv_AT(yv, aty_tmp);
        double xqx = 0.0;
        if (hasq) { be.spmv_Q(xv, qx_tmp); xqx = be.dot(xv, qx_tmp); }
        double rd2 = 0, vobj = 0;
        be.dual_res(aty_tmp, hasq ? &qx_tmp : nullptr, rd2, vobj);
        double robj = be.row_dual_obj(yv);
        k.pobj = (be.dot_c(xv) + 0.5 * xqx) * obj_scale + s.obj_const;
        k.dobj = (robj + vobj - 0.5 * xqx) * obj_scale + s.obj_const;
        k.rel_p = std::sqrt(rp2) / (1.0 + s.bnorm);
        k.rel_d = std::sqrt(rd2) / (1.0 + s.cnorm);
        k.rel_gap = std::fabs(k.pobj - k.dobj) / (1.0 + std::fabs(k.pobj) + std::fabs(k.dobj));
        k.err = std::sqrt(k.rel_p * k.rel_p + k.rel_d * k.rel_d + k.rel_gap * k.rel_gap);
        return k;
    };

    res.setup_seconds = elapsed(t_start);
    if (opt.verbose) {
        std::printf("GANIT PDHG | device %s | %s | rows %d cols %d nnz %lld | ||A||~%.3e | tol %.1e\n",
                    device_name, hasq ? "QP" : "LP", m, n, (long long)s.A.nnz(), normA, opt.tol);
        if (hasq) std::printf("Hessian nnz %lld | ||Q||~%.3e\n", (long long)s.Q.nnz(), s.normQ);
        std::printf("%9s %9s %15s %15s %9s %9s %9s %9s\n", "iter", "time", "primal obj",
                    "dual obj", "rel_p", "rel_d", "rel_gap", "weight");
    }

    const auto t_solve = Clock::now();
    KKT k_restart = eval(x, y);
    double k_prev_cand = std::numeric_limits<double>::infinity();
    long it = 0, it_since = 0;
    double last_log = -1e9;
    KKT best;
    bool best_is_avg = false;
    Status status = Status::IterationLimit;

    while (true) {
        double tau, sigma;
        step_sizes(w, tau, sigma);
        be.spmv_AT(y, aty);
        if (hasq) be.spmv_Q(x, qx);
        be.primal_step(x, aty, hasq ? &qx : nullptr, tau, xn, xbar, sumx);
        be.spmv_A(xbar, axbar);
        be.dual_step(y, axbar, sigma, yn, sumy);
        std::swap(x, xn);
        std::swap(y, yn);
        ++it;
        ++it_since;

        if (it % opt.check_every != 0) continue;

        KKT kc = eval(x, y);
        const double inv = 1.0 / static_cast<double>(it_since);
        be.scale_into(sumx, inv, xa);
        be.scale_into(sumy, inv, ya);
        KKT ka = eval(xa, ya);
        const double t = elapsed(t_solve);

        if (!kc.finite() && !ka.finite()) { status = Status::NumericalError; best = kc; break; }
        const bool avg_better = ka.finite() && (!kc.finite() || ka.err < kc.err);
        KKT kcand = avg_better ? ka : kc;

        if (opt.verbose && (t - last_log >= opt.log_interval)) {
            std::printf("%9ld %8.2fs %15.8e %15.8e %9.2e %9.2e %9.2e %9.2e\n", it, t,
                        kcand.pobj, kcand.dobj, kcand.rel_p, kcand.rel_d, kcand.rel_gap, w);
            last_log = t;
        }
        if (kc.converged(opt.tol)) { status = Status::Optimal; best = kc; best_is_avg = false; break; }
        if (ka.converged(opt.tol)) { status = Status::Optimal; best = ka; best_is_avg = true; break; }
        if (it >= opt.max_iter) { status = Status::IterationLimit; best = kcand; best_is_avg = avg_better; break; }
        if (t >= opt.time_limit) { status = Status::TimeLimit; best = kcand; best_is_avg = avg_better; break; }
        if (it_since >= 4 * opt.check_every && primal_infeasible()) {
            status = Status::PrimalInfeasible; best = kcand; best_is_avg = avg_better; break;
        }

        // ---- adaptive restart ----
        const bool sufficient = kcand.err <= 0.2 * k_restart.err;
        const bool necessary = kcand.err <= 0.8 * k_restart.err && kcand.err > k_prev_cand;
        const bool artificial = it_since >= 0.36 * static_cast<double>(it);
        k_prev_cand = kcand.err;
        if (sufficient || necessary || artificial) {
            if (avg_better) { be.copy(xa, x); be.copy(ya, y); }
            // Primal weight update (PDLP, smoothing theta = 0.5).
            double dx = std::sqrt(be.diff2(x, xr)), dy = std::sqrt(be.diff2(y, yr));
            if (dx > 1e-10 && dy > 1e-10)
                w = std::exp(0.5 * std::log(dy / dx) + 0.5 * std::log(w));
            be.copy(x, xr);
            be.copy(y, yr);
            be.fill(sumx, 0.0);
            be.fill(sumy, 0.0);
            it_since = 0;
            k_restart = kcand;
            k_prev_cand = std::numeric_limits<double>::infinity();
            ++res.restarts;
        }
    }

    res.solve_seconds = elapsed(t_solve);
    res.status = status;
    res.iterations = it;
    std::vector<double> xs = be.to_host(best_is_avg ? xa : x);
    std::vector<double> ys = be.to_host(best_is_avg ? ya : y);
    const double sign = s.maximize ? -1.0 : 1.0;
    res.x.resize(n);
    res.y.resize(m);
    for (int j = 0; j < n; ++j) res.x[j] = xs[j] * s.C[j] * s.bscale;
    for (int i = 0; i < m; ++i) res.y[i] = sign * ys[i] * s.R[i] * s.cscale;
    res.pobj = sign * best.pobj;
    res.dobj = sign * best.dobj;
    res.rel_primal = best.rel_p;
    res.rel_dual = best.rel_d;
    res.rel_gap = best.rel_gap;
    if (opt.verbose) {
        std::printf("Status %s | iters %ld | restarts %d | solve %.3fs (setup %.3fs)\n",
                    to_string(status), it, res.restarts, res.solve_seconds, res.setup_seconds);
        std::printf("Primal obj %.10e | Dual obj %.10e\n", res.pobj, res.dobj);
    }
    return res;
}

}  // namespace ganit
