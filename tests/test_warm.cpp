// Warm-start check: root solve, then tighten one integer bound, re-solve warm,
// and compare with a cold solve of the same modified LP.
#include <cmath>
#include <cstdio>
#include "ganit/lp.hpp"
#include "simplex.hpp"
using namespace ganit;
int main(int argc, char** argv) {
    LP lp = read_mps(argv[1]);
    DualSimplex ds(lp);
    ds.solve();
    double z0 = ds.objective();
    auto x = ds.primal();
    std::printf("root %s %.10g\n", "", z0);
    int bad = 0, tested = 0;
    auto root = ds.save_state();
    for (int j = 0; j < lp.n && tested < 40; ++j) {
        if (!lp.is_int[j]) continue;
        double f = x[j] - std::floor(x[j]);
        if (f < 1e-6 || f > 1 - 1e-6) continue;
        for (int dir = 0; dir < 2; ++dir) {
            double lo = ds.col_lower(j), hi = ds.col_upper(j);
            if (dir == 0) ds.set_col_bounds(j, lo, std::floor(x[j])); else ds.set_col_bounds(j, std::ceil(x[j]), hi);
            LpStatus st = ds.solve();
            double zw = ds.objective();
            LP lp2 = lp;
            if (dir == 0) lp2.u[j] = std::floor(x[j]); else lp2.l[j] = std::ceil(x[j]);
            DualSimplex cold(lp2);
            LpStatus st2 = cold.solve();
            double zc = cold.objective();
            bool ok = st == st2 && (st != LpStatus::Optimal || std::fabs(zw - zc) <= 1e-6 * (1 + std::fabs(zc)));
            if (!ok) { ++bad; std::printf("MISMATCH var %d dir %d: warm %s %.10g cold %s %.10g\n", j, dir, to_string(st), zw, to_string(st2), zc); }
            ds.restore_state(*root);
            ++tested;
        }
    }
    std::printf("tested %d, mismatches %d\n", tested, bad);
}
