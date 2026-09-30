// CPU backend: reference implementation of the fused PDHG kernels.
// Mirrors backend_cuda.cu one-to-one so results can be cross-checked.
#include <algorithm>
#include <cmath>
#include <vector>

#include "pdhg_impl.hpp"

namespace ganit {

const char* to_string(Status s) {
    switch (s) {
        case Status::Optimal: return "OPTIMAL";
        case Status::PrimalInfeasible: return "PRIMAL_INFEASIBLE";
        case Status::IterationLimit: return "ITERATION_LIMIT";
        case Status::TimeLimit: return "TIME_LIMIT";
        default: return "NUMERICAL_ERROR";
    }
}

namespace {

struct CpuBackend {
    using Vec = std::vector<double>;
    const ScaledLP& s;
    explicit CpuBackend(const ScaledLP& lp) : s(lp) {}

    Vec zeros(int n) { return Vec(n, 0.0); }
    Vec from_host(const std::vector<double>& h) { return h; }
    std::vector<double> to_host(const Vec& v) { return v; }
    void copy(const Vec& src, Vec& dst) { dst = src; }
    void fill(Vec& v, double a) { std::fill(v.begin(), v.end(), a); }

    static void spmv(const Csr& a, const Vec& x, Vec& y) {
#pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < a.rows; ++i) {
            double t = 0.0;
            for (int p = a.ptr[i]; p < a.ptr[i + 1]; ++p) t += a.val[p] * x[a.idx[p]];
            y[i] = t;
        }
    }
    void spmv_A(const Vec& x, Vec& ax) { spmv(s.A, x, ax); }
    void spmv_AT(const Vec& y, Vec& aty) { spmv(s.AT, y, aty); }
    void spmv_Q(const Vec& x, Vec& qx) { spmv(s.Q, x, qx); }
    double dot(const Vec& a, const Vec& b) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
        return acc;
    }

    void primal_step(const Vec& x, const Vec& aty, const Vec* qx, double tau, Vec& xn, Vec& xbar, Vec& sumx) {
        const double* q = qx ? qx->data() : nullptr;
#pragma omp parallel for
        for (int j = 0; j < s.n; ++j) {
            double g = s.c[j] - aty[j] + (q ? q[j] : 0.0);
            double v = x[j] - tau * g;
            v = std::min(std::max(v, s.l[j]), s.u[j]);
            xn[j] = v;
            xbar[j] = 2.0 * v - x[j];
            sumx[j] += v;
        }
    }
    void dual_step(const Vec& y, const Vec& axbar, double sigma, Vec& yn, Vec& sumy) {
#pragma omp parallel for
        for (int i = 0; i < s.m; ++i) {
            double q = y[i] - sigma * axbar[i];
            double t = std::min(std::max(-q, sigma * s.lo[i]), sigma * s.hi[i]);
            double v = q + t;
            yn[i] = v;
            sumy[i] += v;
        }
    }
    void sub(const Vec& a, const Vec& b, Vec& d) {
#pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    }
    void scale_into(const Vec& src, double a, Vec& dst) {
#pragma omp parallel for
        for (size_t i = 0; i < src.size(); ++i) dst[i] = a * src[i];
    }

    // ||(Ax - proj_[lo,hi](Ax)) / R||^2   (original-space primal residual)
    double primal_res2(const Vec& ax) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (int i = 0; i < s.m; ++i) {
            double p = std::min(std::max(ax[i], s.lo[i]), s.hi[i]);
            double r = (ax[i] - p) / s.Rres[i];
            acc += r * r;
        }
        return acc;
    }
    // Dual residual (original space) and bound part of the dual objective.
    // with_c = false evaluates the homogeneous dual (for infeasibility rays).
    void dual_res(const Vec& aty, const Vec* qx, double& rd2, double& vobj, bool with_c = true) {
        double a2 = 0.0, ob = 0.0;
        const double cw = with_c ? 1.0 : 0.0;
        const double* q = qx ? qx->data() : nullptr;
#pragma omp parallel for reduction(+ : a2, ob)
        for (int j = 0; j < s.n; ++j) {
            double z = cw * s.c[j] - aty[j] + (q ? q[j] : 0.0);
            bool lf = std::isfinite(s.l[j]), uf = std::isfinite(s.u[j]);
            double r = 0.0;
            if (z > 0) { if (lf) ob += z * s.l[j]; else r = z; }
            else if (z < 0) { if (uf) ob += z * s.u[j]; else r = z; }
            r /= s.Cres[j];
            a2 += r * r;
        }
        rd2 = a2;
        vobj = ob;
    }
    double row_dual_obj(const Vec& y) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (int i = 0; i < s.m; ++i) {
            if (y[i] > 0 && std::isfinite(s.lo[i])) acc += y[i] * s.lo[i];
            else if (y[i] < 0 && std::isfinite(s.hi[i])) acc += y[i] * s.hi[i];
        }
        return acc;
    }
    double dot_c(const Vec& x) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (int j = 0; j < s.n; ++j) acc += s.c[j] * x[j];
        return acc;
    }
    double diff2(const Vec& a, const Vec& b) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; acc += d * d; }
        return acc;
    }
};

}  // namespace

Result solve_pdhg_cpu(const LP& lp, const Options& opt) {
    return run_pdhg<CpuBackend>(lp, opt, "cpu");
}

}  // namespace ganit
