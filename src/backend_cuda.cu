// CUDA backend for GANIT PDHG.
// All kernels written from scratch (no cuSPARSE / cuBLAS / solver libraries):
//   * CSR SpMV: scalar (thread-per-row) or vector (warp-per-row) chosen by
//     average row length
//   * fused primal step: projection + extrapolation + running average in one pass
//   * fused dual step:   prox of row bounds + running average in one pass
//   * block reductions with warp shuffles for norms / objectives
// Requires compute capability >= 6.0 (double atomicAdd).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pdhg_impl.hpp"

#define GANIT_CUDA_CHECK(call)                                                          \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess)                                                       \
            throw std::runtime_error(std::string("CUDA error: ") +                      \
                                     cudaGetErrorString(err__) + " at " __FILE__ ":" +  \
                                     std::to_string(__LINE__));                         \
    } while (0)

namespace ganit {
namespace {

constexpr int kBlock = 256;

inline int grid_for(long long n, int block = kBlock) {
    long long g = (n + block - 1) / block;
    if (g < 1) g = 1;
    if (g > 2147483647LL) g = 2147483647LL;
    return static_cast<int>(g);
}

// ---------------------------------------------------------------- device memory
template <class T>
class DevBuf {
   public:
    DevBuf() = default;
    explicit DevBuf(size_t n) : n_(n) {
        if (n_) GANIT_CUDA_CHECK(cudaMalloc(&p_, n_ * sizeof(T)));
    }
    ~DevBuf() { if (p_) cudaFree(p_); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        std::swap(p_, o.p_);
        std::swap(n_, o.n_);
        return *this;
    }
    T* get() const { return p_; }
    size_t size() const { return n_; }
    void upload(const std::vector<T>& h) {
        if (n_) GANIT_CUDA_CHECK(cudaMemcpy(p_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(n_);
        if (n_) GANIT_CUDA_CHECK(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }

   private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

// ---------------------------------------------------------------- kernels
__global__ void k_spmv_scalar(int rows, const int* __restrict__ ptr, const int* __restrict__ idx,
                              const double* __restrict__ val, const double* __restrict__ x,
                              double* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    double t = 0.0;
    for (int p = ptr[i]; p < ptr[i + 1]; ++p) t += val[p] * __ldg(&x[idx[p]]);
    y[i] = t;
}

// One warp per row; blockDim.x must be a multiple of 32 so whole warps exit together.
__global__ void k_spmv_warp(int rows, const int* __restrict__ ptr, const int* __restrict__ idx,
                            const double* __restrict__ val, const double* __restrict__ x,
                            double* __restrict__ y) {
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int row = static_cast<int>(gid >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    double t = 0.0;
    for (int p = ptr[row] + lane; p < ptr[row + 1]; p += 32) t += val[p] * __ldg(&x[idx[p]]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (lane == 0) y[row] = t;
}

__global__ void k_primal_step(int n, const double* __restrict__ x, const double* __restrict__ aty,
                              const double* __restrict__ qx, const double* __restrict__ c,
                              const double* __restrict__ l, const double* __restrict__ u, double tau,
                              double* __restrict__ xn, double* __restrict__ xbar,
                              double* __restrict__ sumx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double xj = x[j];
    double g = c[j] - aty[j] + (qx ? qx[j] : 0.0);
    double v = xj - tau * g;
    v = fmin(fmax(v, l[j]), u[j]);
    xn[j] = v;
    xbar[j] = 2.0 * v - xj;
    sumx[j] += v;
}

__global__ void k_dual_step(int m, const double* __restrict__ y, const double* __restrict__ axbar,
                            const double* __restrict__ lo, const double* __restrict__ hi,
                            double sigma, double* __restrict__ yn, double* __restrict__ sumy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double q = y[i] - sigma * axbar[i];
    double t = fmin(fmax(-q, sigma * lo[i]), sigma * hi[i]);
    double v = q + t;
    yn[i] = v;
    sumy[i] += v;
}

__global__ void k_scale(int n, double a, const double* __restrict__ s, double* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a * s[i];
}

__global__ void k_sub(int n, const double* __restrict__ a, const double* __restrict__ b,
                      double* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] - b[i];
}

__global__ void k_fill(int n, double a, double* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a;
}

// ---- reductions: functor gives per-index contribution(s), block reduce, atomicAdd.
__device__ inline double warp_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

template <class F>
__global__ void k_reduce2(int n, F f, double* out) {
    __shared__ double sh0[kBlock / 32], sh1[kBlock / 32];
    double a0 = 0.0, a1 = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double v0, v1;
        f(i, v0, v1);
        a0 += v0;
        a1 += v1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sh0[wid] = a0; sh1[wid] = a1; }
    __syncthreads();
    if (wid == 0) {
        a0 = (lane < kBlock / 32) ? sh0[lane] : 0.0;
        a1 = (lane < kBlock / 32) ? sh1[lane] : 0.0;
        a0 = warp_sum(a0);
        a1 = warp_sum(a1);
        if (lane == 0) {
            atomicAdd(&out[0], a0);
            atomicAdd(&out[1], a1);
        }
    }
}

struct PrimalResF {
    const double *ax, *lo, *hi, *R;
    __device__ void operator()(int i, double& v0, double& v1) const {
        double p = fmin(fmax(ax[i], lo[i]), hi[i]);
        double r = (ax[i] - p) / R[i];
        v0 = r * r;
        v1 = 0.0;
    }
};
struct DualResF {
    const double *aty, *qx, *c, *l, *u, *C;
    double cw;  // 1 = normal dual, 0 = homogeneous dual (infeasibility ray)
    __device__ void operator()(int j, double& v0, double& v1) const {
        double z = cw * c[j] - aty[j] + (qx ? qx[j] : 0.0);
        double r = 0.0, ob = 0.0;
        if (z > 0) { if (isfinite(l[j])) ob = z * l[j]; else r = z; }
        else if (z < 0) { if (isfinite(u[j])) ob = z * u[j]; else r = z; }
        r /= C[j];
        v0 = r * r;
        v1 = ob;
    }
};
struct RowDualObjF {
    const double *y, *lo, *hi;
    __device__ void operator()(int i, double& v0, double& v1) const {
        double yi = y[i];
        v0 = 0.0;
        if (yi > 0 && isfinite(lo[i])) v0 = yi * lo[i];
        else if (yi < 0 && isfinite(hi[i])) v0 = yi * hi[i];
        v1 = 0.0;
    }
};
struct DotF {
    const double *a, *b;
    __device__ void operator()(int i, double& v0, double& v1) const { v0 = a[i] * b[i]; v1 = 0.0; }
};
struct Diff2F {
    const double *a, *b;
    __device__ void operator()(int i, double& v0, double& v1) const {
        double d = a[i] - b[i];
        v0 = d * d;
        v1 = 0.0;
    }
};

// ---------------------------------------------------------------- backend
struct DevCsr {
    int rows = 0;
    bool warp = false;
    DevBuf<int> ptr, idx;
    DevBuf<double> val;
};

DevCsr to_device(const Csr& a) {
    DevCsr d;
    d.rows = a.rows;
    d.ptr = DevBuf<int>(a.ptr.size());
    d.ptr.upload(a.ptr);
    d.idx = DevBuf<int>(a.idx.size());
    d.idx.upload(a.idx);
    d.val = DevBuf<double>(a.val.size());
    d.val.upload(a.val);
    double avg = a.rows ? static_cast<double>(a.val.size()) / a.rows : 0.0;
    d.warp = avg > 12.0;  // long rows -> warp-per-row kernel
    return d;
}

struct CudaBackend {
    using Vec = DevBuf<double>;
    int m, n, reduce_grid;
    DevCsr A, AT, Q;
    Vec c, l, u, lo, hi, R, C;
    DevBuf<double> scratch;

    explicit CudaBackend(const ScaledLP& s)
        : m(s.m), n(s.n), A(to_device(s.A)), AT(to_device(s.AT)), Q(to_device(s.Q)), scratch(2) {
        c = from_host(s.c); l = from_host(s.l); u = from_host(s.u);
        lo = from_host(s.lo); hi = from_host(s.hi);
        R = from_host(s.Rres); C = from_host(s.Cres);  // residual unscaling
        int dev = 0, sms = 0;
        GANIT_CUDA_CHECK(cudaGetDevice(&dev));
        GANIT_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        reduce_grid = sms * 8;
    }

    Vec zeros(int k) { Vec v(k); fill(v, 0.0); return v; }
    Vec from_host(const std::vector<double>& h) { Vec v(h.size()); v.upload(h); return v; }
    std::vector<double> to_host(const Vec& v) { return v.download(); }
    void copy(const Vec& src, Vec& dst) {
        if (src.size())
            GANIT_CUDA_CHECK(cudaMemcpy(dst.get(), src.get(), src.size() * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    }
    void fill(Vec& v, double a) {
        if (v.size()) k_fill<<<grid_for(v.size()), kBlock>>>((int)v.size(), a, v.get());
    }

    static void spmv(const DevCsr& a, const Vec& x, Vec& y) {
        if (a.rows == 0) return;
        if (a.warp)
            k_spmv_warp<<<grid_for(static_cast<long long>(a.rows) * 32), kBlock>>>(
                a.rows, a.ptr.get(), a.idx.get(), a.val.get(), x.get(), y.get());
        else
            k_spmv_scalar<<<grid_for(a.rows), kBlock>>>(a.rows, a.ptr.get(), a.idx.get(),
                                                        a.val.get(), x.get(), y.get());
    }
    void spmv_A(const Vec& x, Vec& ax) { spmv(A, x, ax); }
    void spmv_AT(const Vec& y, Vec& aty) { spmv(AT, y, aty); }
    void spmv_Q(const Vec& x, Vec& qx) { spmv(Q, x, qx); }
    double dot(const Vec& a, const Vec& b) {
        return reduce(static_cast<int>(a.size()), DotF{a.get(), b.get()}).first;
    }

    void primal_step(const Vec& x, const Vec& aty, const Vec* qx, double tau, Vec& xn, Vec& xbar, Vec& sumx) {
        if (n) k_primal_step<<<grid_for(n), kBlock>>>(n, x.get(), aty.get(), qx ? qx->get() : nullptr,
                                                      c.get(), l.get(), u.get(), tau, xn.get(),
                                                      xbar.get(), sumx.get());
    }
    void dual_step(const Vec& y, const Vec& axbar, double sigma, Vec& yn, Vec& sumy) {
        if (m) k_dual_step<<<grid_for(m), kBlock>>>(m, y.get(), axbar.get(), lo.get(), hi.get(),
                                                    sigma, yn.get(), sumy.get());
    }
    void sub(const Vec& a, const Vec& b, Vec& d) {
        if (a.size()) k_sub<<<grid_for(a.size()), kBlock>>>((int)a.size(), a.get(), b.get(), d.get());
    }
    void scale_into(const Vec& src, double a, Vec& dst) {
        if (src.size()) k_scale<<<grid_for(src.size()), kBlock>>>((int)src.size(), a, src.get(), dst.get());
    }

    template <class F>
    std::pair<double, double> reduce(int k, F f) {
        GANIT_CUDA_CHECK(cudaMemset(scratch.get(), 0, 2 * sizeof(double)));
        if (k > 0) {
            int g = std::min(grid_for(k), reduce_grid);
            k_reduce2<F><<<g, kBlock>>>(k, f, scratch.get());
        }
        double h[2];
        GANIT_CUDA_CHECK(cudaMemcpy(h, scratch.get(), 2 * sizeof(double), cudaMemcpyDeviceToHost));
        return {h[0], h[1]};
    }
    double primal_res2(const Vec& ax) {
        return reduce(m, PrimalResF{ax.get(), lo.get(), hi.get(), R.get()}).first;
    }
    void dual_res(const Vec& aty, const Vec* qx, double& rd2, double& vobj, bool with_c = true) {
        auto r = reduce(n, DualResF{aty.get(), qx ? qx->get() : nullptr, c.get(), l.get(), u.get(),
                                    C.get(), with_c ? 1.0 : 0.0});
        rd2 = r.first;
        vobj = r.second;
    }
    double row_dual_obj(const Vec& y) { return reduce(m, RowDualObjF{y.get(), lo.get(), hi.get()}).first; }
    double dot_c(const Vec& x) { return reduce(n, DotF{c.get(), x.get()}).first; }
    double diff2(const Vec& a, const Vec& b) {
        return reduce(static_cast<int>(a.size()), Diff2F{a.get(), b.get()}).first;
    }
};

}  // namespace

Result solve_pdhg_gpu(const LP& lp, const Options& opt) {
    int dev = 0;
    GANIT_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    GANIT_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (prop.major < 6) throw std::runtime_error("GANIT GPU backend needs compute capability >= 6.0");
    std::string name = std::string("gpu:") + prop.name;
    Result r = run_pdhg<CudaBackend>(lp, opt, name.c_str());
    GANIT_CUDA_CHECK(cudaGetLastError());
    return r;
}

}  // namespace ganit
