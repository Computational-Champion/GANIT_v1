#include "ganit/lp.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ganit {

Csr csr_from_triplets(int rows, int cols, const std::vector<int>& r,
                      const std::vector<int>& c, const std::vector<double>& v) {
    Csr a;
    a.rows = rows;
    a.cols = cols;
    const size_t nz = v.size();
    std::vector<int> count(rows + 1, 0);
    for (size_t k = 0; k < nz; ++k) {
        if (r[k] < 0 || r[k] >= rows || c[k] < 0 || c[k] >= cols)
            throw std::runtime_error("csr_from_triplets: index out of range");
        count[r[k] + 1]++;
    }
    std::partial_sum(count.begin(), count.end(), count.begin());
    std::vector<int> idx(nz);
    std::vector<double> val(nz);
    std::vector<int> pos(count.begin(), count.end() - 1);
    for (size_t k = 0; k < nz; ++k) {
        int p = pos[r[k]]++;
        idx[p] = c[k];
        val[p] = v[k];
    }
    // Sort each row by column, merge duplicates, drop zeros.
    a.ptr.assign(rows + 1, 0);
    a.idx.reserve(nz);
    a.val.reserve(nz);
    std::vector<std::pair<int, double>> buf;
    for (int i = 0; i < rows; ++i) {
        buf.clear();
        for (int p = count[i]; p < count[i + 1]; ++p) buf.emplace_back(idx[p], val[p]);
        std::sort(buf.begin(), buf.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        size_t q = 0;
        while (q < buf.size()) {
            int col = buf[q].first;
            double s = 0.0;
            while (q < buf.size() && buf[q].first == col) s += buf[q++].second;
            if (s != 0.0) {
                a.idx.push_back(col);
                a.val.push_back(s);
            }
        }
        a.ptr[i + 1] = static_cast<int>(a.idx.size());
    }
    return a;
}

Csr transpose(const Csr& a) {
    Csr t;
    t.rows = a.cols;
    t.cols = a.rows;
    t.ptr.assign(a.cols + 1, 0);
    for (int k : a.idx) t.ptr[k + 1]++;
    std::partial_sum(t.ptr.begin(), t.ptr.end(), t.ptr.begin());
    t.idx.resize(a.idx.size());
    t.val.resize(a.val.size());
    std::vector<int> pos(t.ptr.begin(), t.ptr.end() - 1);
    for (int i = 0; i < a.rows; ++i)
        for (int p = a.ptr[i]; p < a.ptr[i + 1]; ++p) {
            int q = pos[a.idx[p]]++;
            t.idx[q] = i;
            t.val[q] = a.val[p];
        }
    return t;
}

int LP::num_int() const {
    return static_cast<int>(std::count(is_int.begin(), is_int.end(), 1));
}

}  // namespace ganit
