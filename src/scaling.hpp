#pragma once
#include "ganit/lp.hpp"

namespace ganit {

// Scaled problem:  A' = diag(R) A diag(C),  x = C x',  y = R y'.
// Internally always a minimisation (objective negated for MAX problems).
struct ScaledLP {
    int m = 0, n = 0;
    Csr A, AT;
    std::vector<double> c, l, u, lo, hi;  // scaled data
    std::vector<double> R, C;             // row / column scaling factors
    // Uniform bound/objective rescaling:  x' = x_matrix_scaled / bscale,
    // c' = c_matrix_scaled / cscale.  Rres/Cres map residuals to original space.
    double bscale = 1.0, cscale = 1.0;
    std::vector<double> Rres, Cres;
    Csr Q;               // scaled Hessian: Q' = sign * C Q C * bscale / cscale
    double normQ = 0.0;  // ||Q'||_2 estimate
    bool has_q() const { return Q.nnz() > 0; }
    double obj_const = 0.0;
    double bnorm = 0.0, cnorm = 0.0;      // norms of ORIGINAL rhs and objective
    bool maximize = false;
};

// Ruiz equilibration (inf-norm) followed by optional Pock-Chambolle (alpha=1).
ScaledLP scale_lp(const LP& lp, int ruiz_iters, bool pock_chambolle, bool bound_obj_rescale = true);

// Estimate ||A'||_2 by power iteration on A'^T A'.
double estimate_norm(const ScaledLP& s, int iters = 50);

double norm2(const std::vector<double>& v);
double finite_bound_norm(const std::vector<double>& lo, const std::vector<double>& hi);

}  // namespace ganit
