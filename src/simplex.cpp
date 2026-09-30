#include "simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "scaling.hpp"

namespace ganit {

const char* to_string(LpStatus s) {
    switch (s) {
        case LpStatus::Optimal: return "OPTIMAL";
        case LpStatus::Infeasible: return "INFEASIBLE";
        case LpStatus::Unbounded: return "UNBOUNDED";
        case LpStatus::IterationLimit: return "ITERATION_LIMIT";
        case LpStatus::TimeLimit: return "TIME_LIMIT";
        case LpStatus::Cutoff: return "CUTOFF";
        default: return "NUMERICAL_ERROR";
    }
}



DualSimplex::~DualSimplex() = default;

DualSimplex::DualSimplex(const LP& lp, const SimplexOptions& opt) : opt_(opt) {
    ScaledLP s = scale_lp(lp, opt.ruiz_iters, false, false);
    n_ = s.n;
    m_ = s.m;
    maximize_ = lp.maximize;
    colscale_ = s.C;
    rowscale_ = s.R;
    obj_const_ = s.obj_const;
    double cmax = 0.0;
    for (double v : s.c) cmax = std::max(cmax, std::fabs(v));
    obj_scale_ = cmax > 0 ? cmax : 1.0;

    rows_.assign(m_, {});
    cols_.assign(n_, {});
    for (int i = 0; i < m_; ++i)
        for (int p = s.A.ptr[i]; p < s.A.ptr[i + 1]; ++p) {
            rows_[i].emplace_back(s.A.idx[p], s.A.val[p]);
            cols_[s.A.idx[p]].emplace_back(i, s.A.val[p]);
        }
    const int N0 = n_ + m_;
    cost_.assign(N0, 0.0);
    lb_.resize(N0);
    ub_.resize(N0);
    for (int j = 0; j < n_; ++j) {
        cost_[j] = s.c[j] / obj_scale_;
        lb_[j] = s.l[j];
        ub_[j] = s.u[j];
    }
    for (int i = 0; i < m_; ++i) {
        lb_[n_ + i] = s.lo[i];
        ub_[n_ + i] = s.hi[i];
    }
    init_working_bounds();
    x_.assign(N0, 0.0);
    d_.assign(N0, 0.0);
    status_.assign(N0, AT_LB);
    head_.resize(m_);
    for (int i = 0; i < m_; ++i) {
        head_[i] = n_ + i;
        status_[n_ + i] = BASIC;
    }
    for (int j = 0; j < n_; ++j) {
        bool lf = std::isfinite(lb_[j]), uf = std::isfinite(ub_[j]);
        if (lf && uf) status_[j] = cost_[j] >= 0 ? AT_LB : AT_UB;
        else if (lf) status_[j] = AT_LB;
        else if (uf) status_[j] = AT_UB;
        else status_[j] = AT_FREE;
        set_nonbasic_value(j);
    }
}

void DualSimplex::init_working_bounds() {
    const int NN = N();
    wl_.resize(NN);
    wu_.resize(NN);
    art_lo_.resize(NN);
    art_up_.resize(NN);
    for (int j = 0; j < NN; ++j) {
        art_lo_[j] = !std::isfinite(lb_[j]);
        art_up_[j] = !std::isfinite(ub_[j]);
        wl_[j] = art_lo_[j] ? -bigM_ : lb_[j];
        wu_[j] = art_up_[j] ? bigM_ : ub_[j];
    }
}

void DualSimplex::set_nonbasic_value(int j) {
    switch (status_[j]) {
        case AT_LB: x_[j] = wl_[j]; break;
        case AT_UB: x_[j] = wu_[j]; break;
        case AT_FREE: x_[j] = 0.0; break;
        default: break;
    }
}

bool DualSimplex::is_artificial_bound(int j) const {
    return (status_[j] == AT_LB && art_lo_[j]) || (status_[j] == AT_UB && art_up_[j]);
}

// ---------------------------------------------------------------- inversion
bool DualSimplex::reinvert() {
    for (int attempt = 0; attempt < m_ + 2; ++attempt) {
        std::vector<int> row_in_K(m_, -1), Krow, Spos;
        std::vector<char> covered(m_, 0);
        for (int p = 0; p < m_; ++p) {
            int j = head_[p];
            if (j >= n_) covered[j - n_] = 1;
            else Spos.push_back(p);
        }
        for (int i = 0; i < m_; ++i)
            if (!covered[i]) { row_in_K[i] = static_cast<int>(Krow.size()); Krow.push_back(i); }
        const int s = static_cast<int>(Spos.size());
        if (static_cast<int>(Krow.size()) != s) throw std::runtime_error("simplex: inconsistent basis");

        // kernel KM (s x s) row-major: KM[r][c] = a'(Krow[r], head[Spos[c]])
        std::vector<double> KM(static_cast<size_t>(s) * s, 0.0), INV(static_cast<size_t>(s) * s, 0.0);
        for (int c = 0; c < s; ++c)
            for (auto& e : cols_[head_[Spos[c]]]) {
                int r = row_in_K[e.first];
                if (r >= 0) KM[static_cast<size_t>(r) * s + c] = e.second;
            }
        for (int r = 0; r < s; ++r) INV[static_cast<size_t>(r) * s + r] = 1.0;
        std::vector<int> perm(s);
        for (int r = 0; r < s; ++r) perm[r] = r;
        int singular_col = -1;
        for (int c = 0; c < s && singular_col < 0; ++c) {
            int p = c;
            double best = std::fabs(KM[static_cast<size_t>(c) * s + c]);
            for (int r = c + 1; r < s; ++r) {
                double v = std::fabs(KM[static_cast<size_t>(r) * s + c]);
                if (v > best) { best = v; p = r; }
            }
            if (best < 1e-11) { singular_col = c; break; }
            if (p != c) {
                for (int k = 0; k < s; ++k) {
                    std::swap(KM[static_cast<size_t>(p) * s + k], KM[static_cast<size_t>(c) * s + k]);
                    std::swap(INV[static_cast<size_t>(p) * s + k], INV[static_cast<size_t>(c) * s + k]);
                }
                std::swap(perm[p], perm[c]);
            }
            double* kr = &KM[static_cast<size_t>(c) * s];
            double* ir = &INV[static_cast<size_t>(c) * s];
            double inv = 1.0 / kr[c];
            for (int k = 0; k < s; ++k) { kr[k] *= inv; ir[k] *= inv; }
            for (int r = 0; r < s; ++r) {
                if (r == c) continue;
                double f = KM[static_cast<size_t>(r) * s + c];
                if (f == 0.0) continue;
                double* kr2 = &KM[static_cast<size_t>(r) * s];
                double* ir2 = &INV[static_cast<size_t>(r) * s];
                for (int k = c; k < s; ++k) kr2[k] -= f * kr[k];
                for (int k = 0; k < s; ++k) ir2[k] -= f * ir[k];
            }
        }
        if (singular_col >= 0) {
            // Replace the dependent structural by the logical of an uncovered row.
            int p = Spos[singular_col];
            int j = head_[p];
            int i = Krow[perm[singular_col]];
            status_[j] = (std::isfinite(lb_[j]) || !std::isfinite(ub_[j])) ? AT_LB : AT_UB;
            set_nonbasic_value(j);
            head_[p] = n_ + i;
            status_[n_ + i] = BASIC;
            if (opt_.verbose) std::printf("[simplex] singular basis repaired (col %d -> row %d)\n", j, i);
            continue;
        }
        // Assemble full inverse.
        binv_.assign(static_cast<size_t>(m_) * m_, 0.0);
        std::vector<int> cidx(n_, -1);
        for (int c = 0; c < s; ++c) cidx[head_[Spos[c]]] = c;
        for (int c = 0; c < s; ++c) {
            double* br = &binv_[static_cast<size_t>(Spos[c]) * m_];
            const double* ir = &INV[static_cast<size_t>(c) * s];
            for (int r = 0; r < s; ++r) br[Krow[r]] = ir[r];
        }
        for (int p = 0; p < m_; ++p) {
            int j = head_[p];
            if (j < n_) continue;
            int i = j - n_;
            double* br = &binv_[static_cast<size_t>(p) * m_];
            br[i] = -1.0;
            for (auto& e : rows_[i]) {
                int c = cidx[e.first];
                if (c < 0) continue;
                const double* ir = &INV[static_cast<size_t>(c) * s];
                for (int r = 0; r < s; ++r) br[Krow[r]] += e.second * ir[r];
            }
        }
        dse_.assign(m_, 0.0);
        for (int p = 0; p < m_; ++p) {
            const double* br = &binv_[static_cast<size_t>(p) * m_];
            double t = 0.0;
            for (int i = 0; i < m_; ++i) t += br[i] * br[i];
            dse_[p] = t;
        }
        updates_since_inv_ = 0;
        inverted_ = true;
        return true;
    }
    return false;
}

void DualSimplex::compute_primal() {
    std::vector<double> r(m_, 0.0);
    const int NN = N();
    for (int j = 0; j < NN; ++j) {
        if (status_[j] == BASIC) continue;
        double v = x_[j];
        if (v == 0.0) continue;
        if (j < n_) for (auto& e : cols_[j]) r[e.first] += e.second * v;
        else r[j - n_] -= v;
    }
    for (int p = 0; p < m_; ++p) {
        const double* br = &binv_[static_cast<size_t>(p) * m_];
        double t = 0.0;
        for (int i = 0; i < m_; ++i) t += br[i] * r[i];
        x_[head_[p]] = -t;
    }
}

void DualSimplex::compute_duals() {
    std::vector<double> y(m_, 0.0);
    for (int p = 0; p < m_; ++p) {
        double cb = cost_[head_[p]];
        if (cb == 0.0) continue;
        const double* br = &binv_[static_cast<size_t>(p) * m_];
        for (int i = 0; i < m_; ++i) y[i] += cb * br[i];
    }
    for (int j = 0; j < n_; ++j) {
        if (status_[j] == BASIC) { d_[j] = 0.0; continue; }
        double t = cost_[j];
        for (auto& e : cols_[j]) t -= y[e.first] * e.second;
        d_[j] = t;
    }
    for (int i = 0; i < m_; ++i) d_[n_ + i] = status_[n_ + i] == BASIC ? 0.0 : y[i];
}

int DualSimplex::repair_dual() {
    int flips = 0;
    const int NN = N();
    const double tol = opt_.dual_tol;
    for (int j = 0; j < NN; ++j) {
        signed char st = status_[j];
        if (st == BASIC) continue;
        if (wl_[j] == wu_[j]) { if (st != AT_LB) { status_[j] = AT_LB; set_nonbasic_value(j); } continue; }
        signed char ns = st;
        if (st == AT_LB && d_[j] < -tol) ns = AT_UB;
        else if (st == AT_UB && d_[j] > tol) ns = AT_LB;
        else if (st == AT_FREE && std::fabs(d_[j]) > tol) ns = d_[j] > 0 ? AT_LB : AT_UB;
        if (ns != st) { status_[j] = ns; set_nonbasic_value(j); ++flips; }
    }
    return flips;
}

void DualSimplex::expand_bigM() {
    bigM_ *= 100.0;
    const int NN = N();
    for (int j = 0; j < NN; ++j) {
        if (art_lo_[j]) wl_[j] = -bigM_;
        if (art_up_[j]) wu_[j] = bigM_;
        if (status_[j] != BASIC) set_nonbasic_value(j);
    }
}

void DualSimplex::ftran_col(int j, std::vector<double>& out) const {
    out.assign(m_, 0.0);
    if (j < n_) {
        for (auto& e : cols_[j]) {
            const int i = e.first;
            const double a = e.second;
            for (int p = 0; p < m_; ++p) out[p] += binv_[static_cast<size_t>(p) * m_ + i] * a;
        }
    } else {
        const int i = j - n_;
        for (int p = 0; p < m_; ++p) out[p] = -binv_[static_cast<size_t>(p) * m_ + i];
    }
}

void DualSimplex::compute_row(const double* rho, std::vector<double>& alpha) const {
    alpha.assign(N(), 0.0);
    for (int i = 0; i < m_; ++i) {
        double r = rho[i];
        if (r == 0.0) continue;
        for (auto& e : rows_[i]) alpha[e.first] += r * e.second;
        alpha[n_ + i] = -r;
    }
}

double DualSimplex::scaled_objective() const {
    double t = 0.0;
    for (int j = 0; j < n_; ++j) t += cost_[j] * x_[j];
    return t;
}

// ---------------------------------------------------------------- main loop
LpStatus DualSimplex::solve(long iter_limit, double cutoff, double time_limit) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    if (!inverted_ && !reinvert()) return LpStatus::Numerical;
    compute_duals();
    repair_dual();
    compute_primal();

    const double ptol = opt_.primal_tol, dtol = opt_.dual_tol, ptv = opt_.pivot_tol;
    std::vector<double> alpha, aq, rho(m_), dvec;
    long it = 0;
    int numerical_retries = 0;

    while (true) {
        if (iter_limit >= 0 && it >= iter_limit) return LpStatus::IterationLimit;
        if ((it & 31) == 0 && std::isfinite(time_limit) &&
            std::chrono::duration<double>(Clock::now() - t0).count() > time_limit)
            return LpStatus::TimeLimit;
        if (updates_since_inv_ >= opt_.refactor_every) {
            if (!reinvert()) return LpStatus::Numerical;
            compute_duals();
            repair_dual();
            compute_primal();
        }
        if (std::isfinite(cutoff) && (it % 8) == 0 && dual_bound_min() >= cutoff) return LpStatus::Cutoff;

        // ---- pricing: dual steepest edge
        int r = -1;
        double best = 0.0;
        for (int p = 0; p < m_; ++p) {
            int j = head_[p];
            double v = x_[j], inf = 0.0;
            if (v < wl_[j] - ptol * (1.0 + std::fabs(wl_[j]))) inf = wl_[j] - v;
            else if (v > wu_[j] + ptol * (1.0 + std::fabs(wu_[j]))) inf = v - wu_[j];
            else continue;
            double sc = inf * inf / std::max(dse_[p], 1e-12);
            if (sc > best) { best = sc; r = p; }
        }
        if (r < 0) {
            // Primal feasible for the working bounds: check artificial bounds.
            bool need_expand = false, moved = false;
            const int NN = N();
            for (int j = 0; j < NN; ++j) {
                if (!is_artificial_bound(j)) continue;
                if (std::fabs(d_[j]) > dtol) { need_expand = true; break; }
                // degenerate: move to the finite bound (or free at zero)
                if (status_[j] == AT_LB) status_[j] = art_up_[j] ? AT_FREE : AT_UB;
                else status_[j] = art_lo_[j] ? AT_FREE : AT_LB;
                set_nonbasic_value(j);
                moved = true;
            }
            if (need_expand) {
                if (bigM_ >= 1e12) return LpStatus::Unbounded;
                expand_bigM();
                compute_primal();
                continue;
            }
            if (moved) { compute_primal(); continue; }
            return LpStatus::Optimal;
        }
        const int jl = head_[r];
        const bool to_lower = x_[jl] < wl_[jl];
        const double sgn = to_lower ? 1.0 : -1.0;

        std::copy(&binv_[static_cast<size_t>(r) * m_], &binv_[static_cast<size_t>(r) * m_] + m_, rho.begin());
        compute_row(rho.data(), alpha);

        // ---- Harris ratio test, pass 1
        const int NN = N();
        double tmax = kInf;
        for (int j = 0; j < NN; ++j) {
            signed char st = status_[j];
            if (st == BASIC || wl_[j] == wu_[j]) continue;
            double a = sgn * alpha[j];
            if (std::fabs(a) <= ptv) continue;
            double bnd;
            if (st == AT_LB) { if (a >= 0) continue; bnd = (d_[j] + dtol) / (-a); }
            else if (st == AT_UB) { if (a <= 0) continue; bnd = (-d_[j] + dtol) / a; }
            else bnd = (std::fabs(d_[j]) + dtol) / std::fabs(a);
            if (bnd < tmax) tmax = bnd;
        }
        if (!std::isfinite(tmax)) {
            // No entering candidate: primal infeasible, unless an artificial bound blocks.
            bool helpful = false;
            for (int j = 0; j < NN && !helpful; ++j) {
                signed char st = status_[j];
                if (st == BASIC) continue;
                double a = sgn * alpha[j];
                if (std::fabs(a) <= ptv) continue;
                if ((st == AT_LB && a > 0 && art_lo_[j]) || (st == AT_UB && a < 0 && art_up_[j])) helpful = true;
            }
            if (helpful && bigM_ < 1e12) { expand_bigM(); compute_primal(); continue; }
            return LpStatus::Infeasible;
        }
        // ---- pass 2: largest |alpha| among ratios <= tmax
        int q = -1;
        double qa = 0.0, qratio = 0.0;
        for (int j = 0; j < NN; ++j) {
            signed char st = status_[j];
            if (st == BASIC || wl_[j] == wu_[j]) continue;
            double a = sgn * alpha[j];
            if (std::fabs(a) <= ptv) continue;
            double ratio;
            if (st == AT_LB) { if (a >= 0) continue; ratio = d_[j] / (-a); }
            else if (st == AT_UB) { if (a <= 0) continue; ratio = -d_[j] / a; }
            else ratio = std::fabs(d_[j]) / std::fabs(a);
            if (ratio <= tmax && std::fabs(a) > qa) { qa = std::fabs(a); q = j; qratio = ratio; }
        }
        if (q < 0) return LpStatus::Numerical;
        const double t = std::max(0.0, qratio);

        ftran_col(q, aq);
        const double piv = aq[r];
        if (std::fabs(piv) < ptv || std::fabs(piv - alpha[q]) > 1e-6 * (1.0 + std::fabs(piv))) {
            if (updates_since_inv_ > 0 && numerical_retries < 5) {
                ++numerical_retries;
                if (!reinvert()) return LpStatus::Numerical;
                compute_duals();
                repair_dual();
                compute_primal();
                continue;
            }
            if (std::fabs(piv) < ptv) return LpStatus::Numerical;
        }

        // ---- dual update
        for (int j = 0; j < NN; ++j) {
            if (status_[j] == BASIC || alpha[j] == 0.0) continue;
            d_[j] += t * sgn * alpha[j];
        }
        d_[jl] = to_lower ? t : -t;
        d_[q] = 0.0;

        // ---- primal update
        const double bound = to_lower ? wl_[jl] : wu_[jl];
        const double delta = (x_[jl] - bound) / piv;
        for (int p = 0; p < m_; ++p)
            if (aq[p] != 0.0) x_[head_[p]] -= aq[p] * delta;
        x_[q] += delta;
        x_[jl] = bound;
        status_[jl] = to_lower ? AT_LB : AT_UB;
        status_[q] = BASIC;
        head_[r] = q;

        // ---- inverse update (rank one) + exact DSE weights
        double* br = &binv_[static_cast<size_t>(r) * m_];
        const double ipiv = 1.0 / piv;
        for (int i = 0; i < m_; ++i) br[i] *= ipiv;
        for (int p = 0; p < m_; ++p) {
            if (p == r || aq[p] == 0.0) continue;
            double f = aq[p];
            double* bp = &binv_[static_cast<size_t>(p) * m_];
            double s2 = 0.0;
            for (int i = 0; i < m_; ++i) { bp[i] -= f * br[i]; s2 += bp[i] * bp[i]; }
            dse_[p] = s2;
        }
        {
            double s2 = 0.0;
            for (int i = 0; i < m_; ++i) s2 += br[i] * br[i];
            dse_[r] = s2;
        }
        ++updates_since_inv_;
        ++it;
        ++total_iters_;

        // ---- keep boxed variables dual feasible (bound flips)
        bool flipped = false;
        dvec.assign(m_, 0.0);
        for (int j = 0; j < NN; ++j) {
            signed char st = status_[j];
            if (st == BASIC || wl_[j] == wu_[j]) continue;
            signed char ns = st;
            if (st == AT_LB && d_[j] < -dtol) ns = AT_UB;
            else if (st == AT_UB && d_[j] > dtol) ns = AT_LB;
            else if (st == AT_FREE && std::fabs(d_[j]) > dtol) ns = d_[j] > 0 ? AT_LB : AT_UB;
            if (ns == st) continue;
            double old = x_[j];
            status_[j] = ns;
            set_nonbasic_value(j);
            double dv = x_[j] - old;
            if (j < n_) for (auto& e : cols_[j]) dvec[e.first] += e.second * dv;
            else dvec[j - n_] -= dv;
            flipped = true;
        }
        if (flipped) {
            for (int p = 0; p < m_; ++p) {
                const double* bp = &binv_[static_cast<size_t>(p) * m_];
                double s2 = 0.0;
                for (int i = 0; i < m_; ++i) s2 += bp[i] * dvec[i];
                x_[head_[p]] -= s2;
            }
        }
    }
}

// ---------------------------------------------------------------- queries
double DualSimplex::objective() const {
    double v = obj_const_ + obj_scale_ * scaled_objective();
    return maximize_ ? -v : v;
}

double DualSimplex::dual_bound_min() const {
    // Lagrangian bound: sum_j min_{x in [lb,ub]} d_j x_j over the TRUE bounds.
    double t = 0.0;
    const int NN = N();
    for (int j = 0; j < NN; ++j) {
        if (status_[j] == BASIC) continue;
        double dj = d_[j];
        if (std::fabs(dj) <= 1e-12) continue;
        double b = dj > 0 ? lb_[j] : ub_[j];
        if (!std::isfinite(b)) {
            if (std::fabs(dj) <= opt_.dual_tol) continue;
            return -kInf;
        }
        t += dj * b;
    }
    return obj_const_ + obj_scale_ * t;
}

double DualSimplex::primal(int j) const { return x_[j] * colscale_[j]; }

std::vector<double> DualSimplex::primal() const {
    std::vector<double> v(n_);
    for (int j = 0; j < n_; ++j) v[j] = primal(j);
    return v;
}

void DualSimplex::set_col_bounds(int j, double lo, double hi) {
    lb_[j] = lo / colscale_[j];
    ub_[j] = hi / colscale_[j];
    art_lo_[j] = !std::isfinite(lb_[j]);
    art_up_[j] = !std::isfinite(ub_[j]);
    wl_[j] = art_lo_[j] ? -bigM_ : lb_[j];
    wu_[j] = art_up_[j] ? bigM_ : ub_[j];
    if (status_[j] != BASIC) {
        if (status_[j] == AT_FREE && (!art_lo_[j] || !art_up_[j])) status_[j] = art_lo_[j] ? AT_UB : AT_LB;
        set_nonbasic_value(j);
    }
}
double DualSimplex::col_lower(int j) const { return lb_[j] * colscale_[j]; }
double DualSimplex::col_upper(int j) const { return ub_[j] * colscale_[j]; }

double DualSimplex::var_lower_orig(int j) const {
    return j < n_ ? lb_[j] * colscale_[j] : lb_[j] / rowscale_[j - n_];
}
double DualSimplex::var_upper_orig(int j) const {
    return j < n_ ? ub_[j] * colscale_[j] : ub_[j] / rowscale_[j - n_];
}
double DualSimplex::var_value_orig(int j) const {
    return j < n_ ? x_[j] * colscale_[j] : x_[j] / rowscale_[j - n_];
}

DualSimplex::Basis DualSimplex::get_basis() const { return Basis{head_, status_}; }

void DualSimplex::set_basis(const Basis& b) {
    if (static_cast<int>(b.head.size()) != m_ || static_cast<int>(b.status.size()) != N())
        throw std::runtime_error("set_basis: dimension mismatch");
    head_ = b.head;
    status_ = b.status;
    const int NN = N();
    for (int j = 0; j < NN; ++j) {
        if (status_[j] == BASIC) continue;
        if (status_[j] == AT_FREE && (!art_lo_[j] || !art_up_[j])) status_[j] = art_lo_[j] ? AT_UB : AT_LB;
        set_nonbasic_value(j);
    }
    inverted_ = false;
}

std::unique_ptr<DualSimplex::State> DualSimplex::save_state() const {
    auto s = std::make_unique<State>();
    s->bigM = bigM_;
    s->lb = lb_; s->ub = ub_; s->wl = wl_; s->wu = wu_;
    s->x = x_; s->d = d_; s->binv = binv_; s->dse = dse_;
    s->art_lo = art_lo_; s->art_up = art_up_;
    s->head = head_; s->status = status_;
    s->updates = updates_since_inv_;
    s->inverted = inverted_;
    s->m = m_;
    return s;
}

void DualSimplex::restore_state(const State& s) {
    if (s.m != m_) throw std::runtime_error("restore_state: rows changed");
    bigM_ = s.bigM;
    lb_ = s.lb; ub_ = s.ub; wl_ = s.wl; wu_ = s.wu;
    x_ = s.x; d_ = s.d; binv_ = s.binv; dse_ = s.dse;
    art_lo_ = s.art_lo; art_up_ = s.art_up;
    head_ = s.head; status_ = s.status;
    updates_since_inv_ = s.updates;
    inverted_ = s.inverted;
}

void DualSimplex::add_row(const std::vector<int>& idx, const std::vector<double>& val, double lo, double hi) {
    if (!inverted_ && !reinvert()) throw std::runtime_error("add_row: cannot invert basis");
    std::vector<std::pair<int, double>> row;
    double amax = 0.0;
    for (size_t k = 0; k < idx.size(); ++k) {
        double a = val[k] * colscale_[idx[k]];
        if (a != 0.0) { row.emplace_back(idx[k], a); amax = std::max(amax, std::fabs(a)); }
    }
    const double rs = amax > 0 ? 1.0 / amax : 1.0;
    for (auto& e : row) e.second *= rs;
    const int i = m_;
    const int jl = n_ + i;
    double act = 0.0;
    for (auto& e : row) act += e.second * x_[e.first];
    for (auto& e : row) cols_[e.first].emplace_back(i, e.second);
    rows_.push_back(row);
    rowscale_.push_back(rs);
    cost_.push_back(0.0);
    lb_.push_back(lo * rs);
    ub_.push_back(hi * rs);
    art_lo_.push_back(!std::isfinite(lb_.back()));
    art_up_.push_back(!std::isfinite(ub_.back()));
    wl_.push_back(art_lo_.back() ? -bigM_ : lb_.back());
    wu_.push_back(art_up_.back() ? bigM_ : ub_.back());
    x_.push_back(act);
    d_.push_back(0.0);
    status_.push_back(BASIC);
    head_.push_back(jl);

    // Expand the inverse: new row = [r_B^T Binv, -1].
    const int mo = m_, mn = m_ + 1;
    std::vector<double> nb(static_cast<size_t>(mn) * mn, 0.0);
    for (int p = 0; p < mo; ++p)
        std::copy(&binv_[static_cast<size_t>(p) * mo], &binv_[static_cast<size_t>(p) * mo] + mo,
                  &nb[static_cast<size_t>(p) * mn]);
    std::vector<double> coef_of_var(n_, 0.0);
    for (auto& e : row) coef_of_var[e.first] = e.second;
    double* last = &nb[static_cast<size_t>(mo) * mn];
    for (int p = 0; p < mo; ++p) {
        int j = head_[p];
        if (j >= n_ || coef_of_var[j] == 0.0) continue;
        double f = coef_of_var[j];
        const double* bp = &binv_[static_cast<size_t>(p) * mo];
        for (int k = 0; k < mo; ++k) last[k] += f * bp[k];
    }
    last[mo] = -1.0;
    binv_.swap(nb);
    m_ = mn;
    double s2 = 0.0;
    for (int k = 0; k < mn; ++k) s2 += last[k] * last[k];
    dse_.push_back(s2);
}

int DualSimplex::tableau_row_orig(int r, std::vector<double>& coef) const {
    std::vector<double> alpha;
    compute_row(&binv_[static_cast<size_t>(r) * m_], alpha);
    const int h = head_[r];
    auto f = [&](int j) { return j < n_ ? 1.0 / colscale_[j] : rowscale_[j - n_]; };
    const double fh = f(h);
    const int NN = N();
    coef.assign(NN, 0.0);
    for (int j = 0; j < NN; ++j) {
        if (status_[j] == BASIC || alpha[j] == 0.0) continue;
        coef[j] = alpha[j] * f(j) / fh;
    }
    return h;
}

void DualSimplex::row_orig(int i, std::vector<int>& idx, std::vector<double>& val) const {
    idx.clear();
    val.clear();
    for (auto& e : rows_[i]) {
        idx.push_back(e.first);
        val.push_back(e.second / (rowscale_[i] * colscale_[e.first]));
    }
}

}  // namespace ganit
