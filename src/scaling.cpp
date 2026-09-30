#include "scaling.hpp"

#include <algorithm>
#include <cmath>

namespace ganit {

double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// Norm of the "right-hand side": one representative finite value per row.
double finite_bound_norm(const std::vector<double>& lo, const std::vector<double>& hi) {
    double s = 0.0;
    for (size_t i = 0; i < lo.size(); ++i) {
        double a = std::isfinite(lo[i]) ? std::fabs(lo[i]) : 0.0;
        double b = std::isfinite(hi[i]) ? std::fabs(hi[i]) : 0.0;
        double v = std::max(a, b);
        s += v * v;
    }
    return std::sqrt(s);
}

namespace {
void apply(Csr& a, const std::vector<double>& rs, const std::vector<double>& cs,
           std::vector<double>& R, std::vector<double>& C) {
    for (int i = 0; i < a.rows; ++i)
        for (int p = a.ptr[i]; p < a.ptr[i + 1]; ++p) a.val[p] *= rs[i] * cs[a.idx[p]];
    for (int i = 0; i < a.rows; ++i) R[i] *= rs[i];
    for (int j = 0; j < a.cols; ++j) C[j] *= cs[j];
}
}  // namespace

ScaledLP scale_lp(const LP& lp, int ruiz_iters, bool pock_chambolle, bool bound_obj_rescale) {
    ScaledLP s;
    s.m = lp.m;
    s.n = lp.n;
    s.A = lp.A;
    s.R.assign(lp.m, 1.0);
    s.C.assign(lp.n, 1.0);
    s.maximize = lp.maximize;
    const double sign = lp.maximize ? -1.0 : 1.0;

    std::vector<double> rs(lp.m), cs(lp.n);
    for (int it = 0; it < ruiz_iters; ++it) {
        std::fill(rs.begin(), rs.end(), 0.0);
        std::fill(cs.begin(), cs.end(), 0.0);
        for (int i = 0; i < lp.m; ++i)
            for (int p = s.A.ptr[i]; p < s.A.ptr[i + 1]; ++p) {
                double a = std::fabs(s.A.val[p]);
                rs[i] = std::max(rs[i], a);
                cs[s.A.idx[p]] = std::max(cs[s.A.idx[p]], a);
            }
        for (auto& v : rs) v = v > 0 ? 1.0 / std::sqrt(v) : 1.0;
        for (auto& v : cs) v = v > 0 ? 1.0 / std::sqrt(v) : 1.0;
        apply(s.A, rs, cs, s.R, s.C);
    }
    if (pock_chambolle) {
        std::fill(rs.begin(), rs.end(), 0.0);
        std::fill(cs.begin(), cs.end(), 0.0);
        for (int i = 0; i < lp.m; ++i)
            for (int p = s.A.ptr[i]; p < s.A.ptr[i + 1]; ++p) {
                double a = std::fabs(s.A.val[p]);
                rs[i] += a;
                cs[s.A.idx[p]] += a;
            }
        for (auto& v : rs) v = v > 0 ? 1.0 / std::sqrt(v) : 1.0;
        for (auto& v : cs) v = v > 0 ? 1.0 / std::sqrt(v) : 1.0;
        apply(s.A, rs, cs, s.R, s.C);
    }
    s.AT = transpose(s.A);

    s.c.resize(lp.n);
    s.l.resize(lp.n);
    s.u.resize(lp.n);
    for (int j = 0; j < lp.n; ++j) {
        s.c[j] = sign * lp.c[j] * s.C[j];
        s.l[j] = lp.l[j] / s.C[j];
        s.u[j] = lp.u[j] / s.C[j];
    }
    s.lo.resize(lp.m);
    s.hi.resize(lp.m);
    for (int i = 0; i < lp.m; ++i) {
        s.lo[i] = lp.lo[i] * s.R[i];
        s.hi[i] = lp.hi[i] * s.R[i];
    }
    if (bound_obj_rescale) {
        s.bscale = finite_bound_norm(s.lo, s.hi) + 1.0;
        s.cscale = norm2(s.c) + 1.0;
        for (auto& v : s.c) v /= s.cscale;
        for (auto& v : s.l) v /= s.bscale;
        for (auto& v : s.u) v /= s.bscale;
        for (auto& v : s.lo) v /= s.bscale;
        for (auto& v : s.hi) v /= s.bscale;
    }
    s.Rres.resize(lp.m);
    s.Cres.resize(lp.n);
    for (int i = 0; i < lp.m; ++i) s.Rres[i] = s.R[i] / s.bscale;
    for (int j = 0; j < lp.n; ++j) s.Cres[j] = s.C[j] / s.cscale;
    s.obj_const = sign * lp.obj_const;
    if (lp.has_q()) {
        s.Q = lp.Q;
        for (int i = 0; i < lp.n; ++i)
            for (int p = s.Q.ptr[i]; p < s.Q.ptr[i + 1]; ++p)
                s.Q.val[p] *= sign * s.C[i] * s.C[s.Q.idx[p]] * s.bscale / s.cscale;
        // power iteration for ||Q'||_2 (symmetric)
        std::vector<double> v(lp.n, 1.0), w(lp.n);
        double nv = norm2(v);
        for (auto& t : v) t /= nv;
        double lam = 0.0;
        for (int it = 0; it < 60; ++it) {
            for (int i = 0; i < lp.n; ++i) {
                double t = 0.0;
                for (int p = s.Q.ptr[i]; p < s.Q.ptr[i + 1]; ++p) t += s.Q.val[p] * v[s.Q.idx[p]];
                w[i] = t;
            }
            double nw = norm2(w);
            if (nw == 0.0) break;
            lam = nw;
            for (int i = 0; i < lp.n; ++i) v[i] = w[i] / nw;
        }
        s.normQ = lam * 1.01;
    }
    s.bnorm = finite_bound_norm(lp.lo, lp.hi);
    s.cnorm = norm2(lp.c);
    return s;
}

double estimate_norm(const ScaledLP& s, int iters) {
    if (s.A.nnz() == 0) return 1.0;
    std::vector<double> v(s.n, 1.0), w(s.m), z(s.n);
    double nv = norm2(v);
    for (auto& x : v) x /= nv;
    double lambda = 0.0;
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < s.m; ++i) {
            double t = 0.0;
            for (int p = s.A.ptr[i]; p < s.A.ptr[i + 1]; ++p) t += s.A.val[p] * v[s.A.idx[p]];
            w[i] = t;
        }
        for (int j = 0; j < s.n; ++j) {
            double t = 0.0;
            for (int p = s.AT.ptr[j]; p < s.AT.ptr[j + 1]; ++p) t += s.AT.val[p] * w[s.AT.idx[p]];
            z[j] = t;
        }
        double nz = norm2(z);
        if (nz == 0.0) break;
        lambda = nz;  // since ||v|| = 1, ||A'^T A' v|| -> sigma_max^2
        for (int j = 0; j < s.n; ++j) v[j] = z[j] / nz;
    }
    // Small safety margin: power iteration underestimates.
    return std::sqrt(lambda) * 1.01;
}

}  // namespace ganit
