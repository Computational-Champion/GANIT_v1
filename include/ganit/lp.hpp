// GANIT - Indigenous GPU-accelerated optimization solver
// Core LP data structures.
//
// LP form handled by the solver:
//     minimize    c'x + obj_const
//     subject to  lo <= A x <= hi        (row bounds, may be +-inf)
//                 l  <=  x  <= u         (column bounds, may be +-inf)
#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ganit {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Compressed sparse row matrix.
struct Csr {
    int rows = 0, cols = 0;
    std::vector<int> ptr;     // size rows+1
    std::vector<int> idx;     // column indices, size nnz
    std::vector<double> val;  // values, size nnz
    int64_t nnz() const { return static_cast<int64_t>(val.size()); }
};

// Build CSR from coordinate triplets. Duplicate entries are summed,
// explicit zeros are dropped.
Csr csr_from_triplets(int rows, int cols, const std::vector<int>& r,
                      const std::vector<int>& c, const std::vector<double>& v);

Csr transpose(const Csr& a);

struct LP {
    std::string name;
    int m = 0, n = 0;
    Csr A;
    std::vector<double> c, l, u, lo, hi;
    double obj_const = 0.0;
    bool maximize = false;
    std::vector<std::string> row_names, col_names;
    std::vector<char> is_int;  // integrality markers
    Csr Q;                     // objective Hessian (n x n, symmetric, full storage):
                               //   minimize c'x + 0.5 x'Qx + obj_const
    bool has_q() const { return Q.nnz() > 0; }
    int num_int() const;
};

// Reads free-format MPS / QPS (QUADOBJ, QMATRIX, QSECTION) incl. .gz on POSIX.
LP read_mps(const std::string& path);

}  // namespace ganit
