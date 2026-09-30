#include "mip.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <queue>

#include "simplex.hpp"

namespace ganit {

const char* to_string(MipStatus s) {
    switch (s) {
        case MipStatus::Optimal: return "OPTIMAL";
        case MipStatus::Infeasible: return "INFEASIBLE";
        case MipStatus::Unbounded: return "UNBOUNDED";
        case MipStatus::TimeLimit: return "TIME_LIMIT";
        case MipStatus::NodeLimit: return "NODE_LIMIT";
        default: return "NUMERICAL_ERROR";
    }
}

namespace {

using Clock = std::chrono::steady_clock;

struct BoundChange { int j; double lo, hi; };

struct Node {
    double bound;
    int depth;
    std::vector<BoundChange> changes;
    std::shared_ptr<DualSimplex::Basis> basis;
    // for pseudocost updates
    double parent_obj = -kInf;
    int branch_var = -1;
    int branch_dir = 0;  // 0 down, 1 up
    double branch_frac = 0;
};

struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.bound != b.bound) return a.bound > b.bound;  // min-heap on bound
        return a.depth < b.depth;
    }
};

class Mip {
   public:
    Mip(const LP& lp, const MipOptions& opt) : lp_(lp), opt_(opt), S_(lp) {
        t0_ = Clock::now();
        const int n = lp.n;
        for (int j = 0; j < n; ++j)
            if (j < static_cast<int>(lp.is_int.size()) && lp.is_int[j]) ints_.push_back(j);
        is_int_.assign(n, 0);
        for (int j : ints_) is_int_[j] = 1;
        // integral bounds for integer columns
        for (int j : ints_) {
            double lo = lp.l[j], hi = lp.u[j];
            if (std::isfinite(lo)) lo = std::ceil(lo - 1e-9);
            if (std::isfinite(hi)) hi = std::floor(hi + 1e-9);
            S_.set_col_bounds(j, lo, hi);
        }
        root_lb_.resize(n);
        root_ub_.resize(n);
        for (int j = 0; j < n; ++j) { root_lb_[j] = S_.col_lower(j); root_ub_[j] = S_.col_upper(j); }
        pc_sum_[0].assign(n, 0.0); pc_sum_[1].assign(n, 0.0);
        pc_n_[0].assign(n, 0); pc_n_[1].assign(n, 0);
        sign_ = lp.maximize ? -1.0 : 1.0;
    }

    MipResult run();

   private:
    const LP& lp_;
    MipOptions opt_;
    DualSimplex S_;
    Clock::time_point t0_;
    std::vector<int> ints_;
    std::vector<char> is_int_;
    std::vector<double> root_lb_, root_ub_;
    double inc_ = kInf;  // min-sense
    std::vector<double> inc_x_;
    std::vector<double> pc_sum_[2];
    std::vector<int> pc_n_[2];
    double sign_ = 1.0;
    long nodes_ = 0;
    int cuts_ = 0;
    double last_log_ = -1e9;

    double elapsed() const { return std::chrono::duration<double>(Clock::now() - t0_).count(); }
    double zmin() const { return sign_ * S_.objective(); }
    double cutoff() const {
        if (!std::isfinite(inc_)) return kInf;
        return inc_ - std::max(opt_.abs_gap, opt_.rel_gap * std::max(1.0, std::fabs(inc_)));
    }
    static double frac(double v) { return v - std::floor(v); }

    int fractional(const std::vector<double>& x, std::vector<int>* out) const {
        int cnt = 0;
        for (int j : ints_) {
            double f = frac(x[j]);
            if (f > opt_.int_tol && f < 1.0 - opt_.int_tol) { ++cnt; if (out) out->push_back(j); }
        }
        return cnt;
    }

    // Verify a candidate against the ORIGINAL model (not the cut-extended LP).
    bool try_incumbent(std::vector<double> x, const char* source) {
        for (int j : ints_) x[j] = std::round(x[j]);
        const double ft = opt_.feas_tol;
        for (int j = 0; j < lp_.n; ++j) {
            double lo = lp_.l[j], hi = lp_.u[j];
            if (x[j] < lo - ft * (1 + std::fabs(lo)) || x[j] > hi + ft * (1 + std::fabs(hi))) return false;
            x[j] = std::min(std::max(x[j], lo), hi);
        }
        for (int i = 0; i < lp_.m; ++i) {
            double a = 0;
            for (int p = lp_.A.ptr[i]; p < lp_.A.ptr[i + 1]; ++p) a += lp_.A.val[p] * x[lp_.A.idx[p]];
            if (a < lp_.lo[i] - ft * (1 + std::fabs(lp_.lo[i])) || a > lp_.hi[i] + ft * (1 + std::fabs(lp_.hi[i])))
                return false;
        }
        double obj = lp_.obj_const;
        for (int j = 0; j < lp_.n; ++j) obj += lp_.c[j] * x[j];
        obj *= sign_;
        if (!std::isfinite(inc_) || obj < inc_ - 1e-9 * (1 + std::fabs(inc_))) {
            inc_ = obj;
            inc_x_ = x;
            if (opt_.verbose)
                std::printf("  * new incumbent %.10g (%s) at node %ld, %.2fs\n", sign_ * inc_, source, nodes_, elapsed());
            return true;
        }
        return false;
    }

    void pc_update(int j, int dir, double gain, double f) {
        if (!(f > 1e-9) || !std::isfinite(gain)) return;
        pc_sum_[dir][j] += std::max(gain, 0.0) / f;
        pc_n_[dir][j] += 1;
    }
    double pc(int j, int dir) const {
        if (pc_n_[dir][j] > 0) return pc_sum_[dir][j] / pc_n_[dir][j];
        double s = 0; long c = 0;
        for (int k : ints_) if (pc_n_[dir][k] > 0) { s += pc_sum_[dir][k] / pc_n_[dir][k]; ++c; }
        return c ? s / c : 1.0;
    }

    void apply_bounds(const std::vector<BoundChange>& ch) {
        for (int j : ints_) S_.set_col_bounds(j, root_lb_[j], root_ub_[j]);
        for (auto& b : ch) S_.set_col_bounds(b.j, b.lo, b.hi);
    }

    int gomory_round();
    void dive();

    // Branching decision. Returns: -2 node infeasible, -1 bound tightened (re-solve), else var.
    int select_branch(const std::vector<double>& x, double z, std::vector<BoundChange>& changes,
                      double& est_down, double& est_up);

    void log(size_t open, double bound, bool force = false) {
        if (!opt_.verbose) return;
        double t = elapsed();
        if (!force && t - last_log_ < opt_.log_interval) return;
        last_log_ = t;
        double gap = std::isfinite(inc_) ? (inc_ - bound) / std::max(1.0, std::fabs(inc_)) : kInf;
        std::printf("%9ld nodes %8zu open | incumbent %16.8g | bound %16.8g | gap %8.3f%% | %7.1fs\n", nodes_, open,
                    std::isfinite(inc_) ? sign_ * inc_ : NAN, sign_ * bound, 100 * gap, t);
    }
};

int Mip::gomory_round() {
    const int n = S_.num_cols();
    const int m = S_.num_rows();
    std::vector<double> x = S_.primal();
    struct Cand { int p; double score; };
    std::vector<Cand> cand;
    for (int p = 0; p < m; ++p) {
        int h = S_.head(p);
        if (h >= n || !is_int_[h]) continue;
        double f = frac(x[h]);
        if (f < 0.005 || f > 0.995) continue;
        cand.push_back({p, std::fabs(f - 0.5)});
    }
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.score < b.score; });
    if (static_cast<int>(cand.size()) > opt_.max_cuts_per_round) cand.resize(opt_.max_cuts_per_round);

    int added = 0;
    std::vector<double> coef, g(n);
    std::vector<int> ridx;
    std::vector<double> rval;
    for (auto& c : cand) {
        int h = S_.tableau_row_orig(c.p, coef);
        double b = S_.var_value_orig(h);
        double f0 = frac(b);
        if (f0 < 0.005 || f0 > 0.995) continue;
        std::fill(g.begin(), g.end(), 0.0);
        double rhs = 1.0;
        bool ok = true;
        const int NN = static_cast<int>(coef.size());
        std::vector<std::pair<int, double>> logical_terms;
        for (int j = 0; j < NN && ok; ++j) {
            double a = coef[j];
            if (std::fabs(a) < 1e-11) continue;
            auto st = S_.status(j);
            if (st == DualSimplex::AT_FREE || S_.is_artificial_bound(j)) { ok = false; break; }
            bool at_lb = st == DualSimplex::AT_LB;
            double bnd = at_lb ? S_.var_lower_orig(j) : S_.var_upper_orig(j);
            if (!std::isfinite(bnd)) { ok = false; break; }
            double ap = at_lb ? a : -a;
            double pi;
            if (j < n && is_int_[j]) {
                double fj = frac(ap);
                pi = fj <= f0 ? fj / f0 : (1.0 - fj) / (1.0 - f0);
            } else {
                pi = ap >= 0 ? ap / f0 : -ap / (1.0 - f0);
            }
            if (pi == 0.0) continue;
            // pi * xt, xt = x - lb (at lb) or ub - x (at ub)
            double gj = at_lb ? pi : -pi;
            rhs += at_lb ? pi * bnd : -pi * bnd;
            if (j < n) g[j] += gj;
            else logical_terms.emplace_back(j - n, gj);
        }
        if (!ok) continue;
        for (auto& lt : logical_terms) {
            S_.row_orig(lt.first, ridx, rval);
            for (size_t k = 0; k < ridx.size(); ++k) g[ridx[k]] += lt.second * rval[k];
        }
        // clean tiny coefficients using bounds; check dynamism
        double gmax = 0;
        for (double v : g) gmax = std::max(gmax, std::fabs(v));
        if (gmax <= 0) continue;
        std::vector<int> idx;
        std::vector<double> val;
        double gmin = kInf;
        for (int j = 0; j < n && ok; ++j) {
            double v = g[j];
            if (v == 0.0) continue;
            if (std::fabs(v) < 1e-9 * gmax) {
                // drop term keeping validity of  sum g x >= rhs
                double lo = S_.col_lower(j), hi = S_.col_upper(j);
                double worst = v > 0 ? v * hi : v * lo;  // max of v*x over the box
                if (!std::isfinite(worst)) { ok = false; break; }
                rhs -= worst;
                continue;
            }
            idx.push_back(j);
            val.push_back(v);
            gmin = std::min(gmin, std::fabs(v));
        }
        if (!ok || idx.empty() || gmax / gmin > 1e7) continue;
        double act = 0, nrm = 0;
        for (size_t k = 0; k < idx.size(); ++k) { act += val[k] * x[idx[k]]; nrm += val[k] * val[k]; }
        double viol = rhs - act;
        if (viol <= 1e-6 * (1 + std::fabs(rhs)) || viol / std::sqrt(nrm) < 1e-5) continue;
        S_.add_row(idx, val, rhs, kInf);
        ++added;
    }
    return added;
}

void Mip::dive() {
    auto saved = S_.save_state();
    std::vector<double> x = S_.primal();
    for (int depth = 0; depth < 100; ++depth) {
        if (elapsed() > opt_.time_limit) break;
        std::vector<int> fr;
        if (fractional(x, &fr) == 0) { try_incumbent(x, "diving"); break; }
        int best = -1;
        double bf = 2;
        for (int j : fr) {
            double f = frac(x[j]);
            double d = std::min(f, 1 - f);
            if (d < bf) { bf = d; best = j; }
        }
        double v = x[best];
        bool up = frac(v) >= 0.5;
        double lo = S_.col_lower(best), hi = S_.col_upper(best);
        if (up) S_.set_col_bounds(best, std::ceil(v), hi); else S_.set_col_bounds(best, lo, std::floor(v));
        LpStatus st = S_.solve(2000, cutoff());
        if (st != LpStatus::Optimal) {
            S_.set_col_bounds(best, lo, hi);
            if (up) S_.set_col_bounds(best, lo, std::floor(v)); else S_.set_col_bounds(best, std::ceil(v), hi);
            st = S_.solve(2000, cutoff());
            if (st != LpStatus::Optimal) break;
        }
        x = S_.primal();
    }
    S_.restore_state(*saved);
}

int Mip::select_branch(const std::vector<double>& x, double z, std::vector<BoundChange>& changes,
                       double& est_down, double& est_up) {
    std::vector<int> fr;
    fractional(x, &fr);
    const double eps = 1e-6;
    struct C { int j; double score; bool reliable; };
    std::vector<C> cs;
    for (int j : fr) {
        double f = frac(x[j]);
        double sd = pc(j, 0) * f, su = pc(j, 1) * (1 - f);
        bool rel = std::min(pc_n_[0][j], pc_n_[1][j]) >= opt_.reliability;
        cs.push_back({j, std::max(sd, eps) * std::max(su, eps), rel});
    }
    std::sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.score > b.score; });

    int best = cs[0].j;
    double best_score = -1;
    est_down = est_up = z;
    std::unique_ptr<DualSimplex::State> saved;
    int sb_done = 0;
    for (auto& c : cs) {
        if (c.reliable || sb_done >= opt_.sb_candidates) {
            if (c.score > best_score) {
                best_score = c.score; best = c.j;
                double f = frac(x[c.j]);
                est_down = z + pc(c.j, 0) * f; est_up = z + pc(c.j, 1) * (1 - f);
            }
            continue;
        }
        if (!saved) saved = S_.save_state();
        const int j = c.j;
        const double v = x[j], f = frac(v);
        const double lo = S_.col_lower(j), hi = S_.col_upper(j);
        double zz[2];
        for (int dir = 0; dir < 2; ++dir) {
            if (dir == 0) S_.set_col_bounds(j, lo, std::floor(v)); else S_.set_col_bounds(j, std::ceil(v), hi);
            LpStatus st = S_.solve(opt_.sb_iter_limit, cutoff());
            if (st == LpStatus::Infeasible || st == LpStatus::Cutoff) zz[dir] = kInf;
            else if (st == LpStatus::Optimal) { zz[dir] = zmin(); pc_update(j, dir, zz[dir] - z, dir ? 1 - f : f); }
            else { double db = S_.dual_bound_min(); zz[dir] = std::isfinite(db) ? std::max(z, db) : z; }
            S_.restore_state(*saved);
        }
        ++sb_done;
        if (!std::isfinite(zz[0]) && !std::isfinite(zz[1])) return -2;
        if (!std::isfinite(zz[0]) || !std::isfinite(zz[1])) {
            // one side infeasible: tighten this node and re-solve
            BoundChange bc = !std::isfinite(zz[0]) ? BoundChange{j, std::ceil(v), hi} : BoundChange{j, lo, std::floor(v)};
            changes.push_back(bc);
            S_.set_col_bounds(bc.j, bc.lo, bc.hi);
            return -1;
        }
        double s = std::max(zz[0] - z, eps) * std::max(zz[1] - z, eps);
        if (s > best_score) { best_score = s; best = j; est_down = zz[0]; est_up = zz[1]; }
    }
    return best;
}

MipResult Mip::run() {
    MipResult res;
    const double inf = kInf;
    if (opt_.verbose)
        std::printf("GANIT MILP | rows %d cols %d (%zu integer) nnz %lld\n", lp_.m, lp_.n, ints_.size(),
                    (long long)lp_.A.nnz());

    LpStatus st = S_.solve(-1, inf, opt_.time_limit);
    if (st == LpStatus::Infeasible) { res.status = MipStatus::Infeasible; res.objective = NAN; res.seconds = elapsed(); return res; }
    if (st == LpStatus::Unbounded) { res.status = MipStatus::Unbounded; res.objective = NAN; res.seconds = elapsed(); return res; }
    if (st != LpStatus::Optimal) { res.status = st == LpStatus::TimeLimit ? MipStatus::TimeLimit : MipStatus::Numerical; res.seconds = elapsed(); return res; }
    res.root_lp = sign_ * zmin();
    if (opt_.verbose) std::printf("Root LP %.10g (%ld simplex iterations, %.2fs)\n", res.root_lp, S_.iterations(), elapsed());

    try_incumbent(S_.primal(), "rounding");
    // ---- root cutting planes
    double zprev = zmin();
    int stall = 0;
    for (int round = 0; round < opt_.cut_rounds && !ints_.empty(); ++round) {
        if (fractional(S_.primal(), nullptr) == 0) break;
        int added = gomory_round();
        if (!added) break;
        cuts_ += added;
        st = S_.solve(-1, cutoff(), opt_.time_limit - elapsed());
        if (st == LpStatus::Infeasible || st == LpStatus::Cutoff) break;
        if (st != LpStatus::Optimal) break;
        double z = zmin();
        if (opt_.verbose) std::printf("  cut round %d: +%d GMI cuts, bound %.10g\n", round + 1, added, sign_ * z);
        if (z - zprev < 1e-4 * std::max(1.0, std::fabs(z))) { if (++stall >= 2) break; } else stall = 0;
        zprev = z;
        try_incumbent(S_.primal(), "rounding");
    }
    if (st != LpStatus::Optimal) {
        // cuts proved the incumbent optimal (cutoff) or the problem infeasible
        if (st == LpStatus::Infeasible || st == LpStatus::Cutoff)
            res.status = std::isfinite(inc_) ? MipStatus::Optimal : MipStatus::Infeasible;
        else
            res.status = st == LpStatus::TimeLimit ? MipStatus::TimeLimit : MipStatus::Numerical;
        res.objective = std::isfinite(inc_) ? sign_ * inc_ : NAN;
        res.bound = res.objective;
        res.gap = 0;
        res.x = inc_x_;
        res.cuts = cuts_;
        res.lp_iterations = S_.iterations();
        res.seconds = elapsed();
        return res;
    }
    res.root_bound = sign_ * zmin();
    if (st == LpStatus::Optimal && !ints_.empty()) dive();

    // ---- branch and bound
    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    Node root;
    root.bound = zmin();
    root.depth = 0;
    root.basis = std::make_shared<DualSimplex::Basis>(S_.get_basis());
    bool have_dive = true;  // root state already loaded in the simplex
    Node cur = root;
    bool warm = true;       // current simplex state matches parent of `cur` (or cur itself for root)
    bool first = true;
    MipStatus stop = MipStatus::Optimal;
    double global_bound = root.bound;

    while (true) {
        if (!have_dive) {
            if (open.empty()) break;
            cur = open.top();
            open.pop();
            if (cur.bound >= cutoff()) { global_bound = inc_; while (!open.empty()) open.pop(); break; }
            apply_bounds(cur.changes);
            S_.set_basis(*cur.basis);
            warm = false;
        }
        have_dive = false;
        if (elapsed() > opt_.time_limit) { stop = MipStatus::TimeLimit; open.push(cur); break; }
        if (nodes_ >= opt_.node_limit) { stop = MipStatus::NodeLimit; open.push(cur); break; }
        ++nodes_;

        global_bound = open.empty() ? cur.bound : std::min(cur.bound, open.top().bound);
        log(open.size(), global_bound);

        if (!first) {
            st = S_.solve(-1, cutoff(), opt_.time_limit - elapsed());
        } else {
            st = LpStatus::Optimal;  // root already solved
        }
        (void)warm;
        first = false;
        if (st == LpStatus::TimeLimit) { stop = MipStatus::TimeLimit; open.push(cur); break; }
        if (st != LpStatus::Optimal) continue;  // infeasible / cutoff / numerical -> prune

        // resolve loop for bound tightening from strong branching
        int j = -1;
        double z = 0, ed = 0, eu = 0;
        std::vector<double> x;
        bool pruned = false;
        for (size_t rep = 0; rep < ints_.size() + 2; ++rep) {
            z = zmin();
            if (z >= cutoff()) { pruned = true; break; }
            if (cur.branch_var >= 0 && rep == 0) pc_update(cur.branch_var, cur.branch_dir, z - cur.parent_obj, cur.branch_frac);
            x = S_.primal();
            if (fractional(x, nullptr) == 0) { try_incumbent(x, "LP solution"); pruned = true; break; }
            if (nodes_ % 20 == 1) try_incumbent(x, "rounding");
            j = select_branch(x, z, cur.changes, ed, eu);
            if (j == -2) { pruned = true; break; }
            if (j == -1) {
                st = S_.solve(-1, cutoff(), opt_.time_limit - elapsed());
                if (st != LpStatus::Optimal) { pruned = true; break; }
                continue;
            }
            break;
        }
        if (pruned || j < 0) continue;

        const double v = x[j], f = frac(v);
        const double lo = S_.col_lower(j), hi = S_.col_upper(j);
        auto basis = std::make_shared<DualSimplex::Basis>(S_.get_basis());
        Node down, up;
        down.depth = up.depth = cur.depth + 1;
        down.changes = cur.changes; down.changes.push_back({j, lo, std::floor(v)});
        up.changes = cur.changes;   up.changes.push_back({j, std::ceil(v), hi});
        down.basis = up.basis = basis;
        down.parent_obj = up.parent_obj = z;
        down.branch_var = up.branch_var = j;
        down.branch_dir = 0; up.branch_dir = 1;
        down.branch_frac = f; up.branch_frac = 1 - f;
        down.bound = std::max(z, ed);
        up.bound = std::max(z, eu);
        // plunge into the more promising child, keep the other
        bool go_up = eu < ed || (eu == ed && f >= 0.5);
        Node& next = go_up ? up : down;
        Node& other = go_up ? down : up;
        if (other.bound < cutoff()) open.push(other);
        if (next.bound < cutoff()) {
            S_.set_col_bounds(j, next.changes.back().lo, next.changes.back().hi);
            cur = next;
            have_dive = true;
            warm = true;
        }
    }

    if (stop == MipStatus::Optimal) {
        res.status = std::isfinite(inc_) ? MipStatus::Optimal : MipStatus::Infeasible;
        global_bound = std::isfinite(inc_) ? inc_ : global_bound;
    } else {
        res.status = stop;
        global_bound = open.empty() ? global_bound : std::min(global_bound, open.top().bound);
        if (std::isfinite(inc_)) global_bound = std::min(global_bound, inc_);
    }
    res.nodes = nodes_;
    res.cuts = cuts_;
    res.lp_iterations = S_.iterations();
    res.objective = std::isfinite(inc_) ? sign_ * inc_ : NAN;
    res.bound = sign_ * global_bound;
    res.gap = std::isfinite(inc_) ? std::fabs(inc_ - global_bound) / std::max(1.0, std::fabs(inc_)) : kInf;
    res.x = inc_x_;
    res.seconds = elapsed();
    log(open.size(), global_bound, true);
    if (opt_.verbose)
        std::printf("Status %s | objective %.10g | bound %.10g | gap %.4f%% | nodes %ld | cuts %d | %.2fs\n",
                    to_string(res.status), res.objective, res.bound, 100 * res.gap, res.nodes, res.cuts, res.seconds);
    return res;
}

}  // namespace

MipResult solve_mip(const LP& lp, const MipOptions& opt) {
    Mip mip(lp, opt);
    return mip.run();
}

}  // namespace ganit
