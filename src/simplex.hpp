// GANIT bounded dual simplex (written from scratch).
//
// Computational form (scaled):   [A  -I] [x; s] = 0,   l <= x <= u,  lo <= s <= hi
// Variables 0..n-1 are structural, n+i is the logical (row activity) of row i.
//
// Design:
//  * explicit dense basis inverse, updated by rank-1 pivots, periodically
//    re-inverted. Re-inversion only factors the "kernel" (rows not covered by
//    basic logicals x basic structural columns), which is usually far smaller
//    than m.
//  * exact dual steepest-edge pricing: the DSE weight of row r is
//    ||e_r^T B^-1||^2, which the explicit inverse gives for free.
//  * Harris two-pass ratio test, bound flipping to keep dual feasibility.
//  * infinite bounds handled by artificial boxing (expanded when active).
//  * warm start after bound changes (branch-and-bound), state save/restore
//    (strong branching), row addition (cutting planes), tableau rows (Gomory).
#pragma once
#include <memory>
#include <vector>

#include "ganit/lp.hpp"

namespace ganit {

enum class LpStatus { Optimal, Infeasible, Unbounded, IterationLimit, TimeLimit, Cutoff, Numerical };
const char* to_string(LpStatus s);

struct SimplexOptions {
    double primal_tol = 1e-7;
    double dual_tol = 1e-7;
    double pivot_tol = 1e-9;
    int refactor_every = 100;
    int ruiz_iters = 10;
    int verbose = 0;
};

class DualSimplex {
   public:
    enum VarStatus : signed char { BASIC = 0, AT_LB = 1, AT_UB = 2, AT_FREE = 3 };
    struct Basis {
        std::vector<int> head;
        std::vector<signed char> status;
    };
    struct State;  // full internal state for strong branching

    DualSimplex(const LP& lp, const SimplexOptions& opt = {});
    ~DualSimplex();

    // Solve from the current basis. iter_limit < 0: unlimited.
    // cutoff: stop early (status Cutoff) once the dual bound proves obj >= cutoff.
    LpStatus solve(long iter_limit = -1, double cutoff = kInf, double time_limit = kInf);

    // --- results in ORIGINAL space (objective in original sense incl. constant)
    double objective() const;
    // Valid lower bound (in minimisation sense) from the current dual solution, or -inf.
    double dual_bound_min() const;
    std::vector<double> primal() const;
    double primal(int j) const;
    long iterations() const { return total_iters_; }

    // --- column bounds in ORIGINAL space
    void set_col_bounds(int j, double lo, double hi);
    double col_lower(int j) const;
    double col_upper(int j) const;

    // --- warm start
    Basis get_basis() const;
    void set_basis(const Basis& b);
    std::unique_ptr<State> save_state() const;
    void restore_state(const State& s);

    // --- cutting planes: lo <= sum val*x[idx] <= hi (original space, structural vars)
    void add_row(const std::vector<int>& idx, const std::vector<double>& val, double lo, double hi);
    int num_rows() const { return m_; }
    int num_cols() const { return n_; }

    // Tableau row of the basic variable at basis position r, expressed in
    // ORIGINAL units: x_head + sum_j coef[j] * x_j = const over nonbasic j.
    // Returns the head variable; coef has size n+m (0 for basic entries).
    int tableau_row_orig(int r, std::vector<double>& coef) const;
    int head(int r) const { return head_[r]; }
    VarStatus status(int j) const { return static_cast<VarStatus>(status_[j]); }
    bool is_artificial_bound(int j) const;
    // original-space bounds of any variable (logical j = n+i -> row bounds)
    double var_lower_orig(int j) const;
    double var_upper_orig(int j) const;
    double var_value_orig(int j) const;
    // row coefficients (original space) of row i
    void row_orig(int i, std::vector<int>& idx, std::vector<double>& val) const;
    bool is_maximize() const { return maximize_; }

   private:
    // problem (scaled)
    int n_ = 0, m_ = 0;
    std::vector<std::vector<std::pair<int, double>>> cols_;  // structural columns
    std::vector<std::vector<std::pair<int, double>>> rows_;  // rows (structural part)
    std::vector<double> cost_;                               // size n+m
    std::vector<double> lb_, ub_;                            // true bounds, size n+m
    std::vector<double> colscale_, rowscale_;                // x = C x', s' = R s
    double obj_scale_ = 1.0, obj_const_ = 0.0;
    bool maximize_ = false;
    SimplexOptions opt_;

    // working state
    double bigM_ = 1e6;
    std::vector<double> wl_, wu_;
    std::vector<char> art_lo_, art_up_;
    std::vector<double> x_, d_;
    std::vector<int> head_;
    std::vector<signed char> status_;
    std::vector<double> binv_;   // m x m row-major, row = basis position, col = constraint row
    std::vector<double> dse_;    // ||row r of binv||^2
    int updates_since_inv_ = 0;
    long total_iters_ = 0;
    bool inverted_ = false;

    int N() const { return n_ + m_; }
    void init_working_bounds();
    void set_nonbasic_value(int j);
    bool reinvert();
    void compute_primal();
    void compute_duals();
    int repair_dual();
    void expand_bigM();
    void ftran_col(int j, std::vector<double>& out) const;
    void compute_row(const double* rho, std::vector<double>& alpha) const;
    double scaled_objective() const;
};

struct DualSimplex::State {  // full solver state (strong branching)
    double bigM;
    std::vector<double> lb, ub, wl, wu, x, d, binv, dse;
    std::vector<char> art_lo, art_up;
    std::vector<int> head;
    std::vector<signed char> status;
    int updates;
    bool inverted;
    int m;
};

}  // namespace ganit
