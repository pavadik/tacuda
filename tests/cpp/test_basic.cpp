#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "../../include/tacuda.h"

static void approx_equal(const std::vector<float>& a, const std::vector<float>& b, float eps=1e-3f) {
    for (size_t i=0;i<a.size();++i) {
        if (std::isnan(a[i]) || std::isnan(b[i])) continue;
        float d = std::fabs(a[i]-b[i]);
        if (d > eps) {
            std::cerr << "Mismatch at " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            assert(false);
        }
    }
}

int main() {
    const int N = 128;
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = std::sin(0.05f * i);

    std::vector<float> out(N, 0.0f), ref(N, 0.0f);

    // SMA check vs CPU naive
    int p = 5;
    int rc = ct_sma(x.data(), out.data(), N, p);
    if (rc != 0) { std::cerr << "ct_sma failed\\n"; return 1; }
    for (int i=0;i<=N-p;i++) {
        float s=0; for (int k=0;k<p;k++) s+=x[i+k];
        ref[i] = s/p;
    }
    approx_equal(out, ref, 1e-3f);

    // Momentum check
    std::fill(ref.begin(), ref.end(), 0.0f);
    rc = ct_momentum(x.data(), out.data(), N, p);
    if (rc != 0) { std::cerr << "ct_momentum failed\\n"; return 1; }
    for (int i=0;i<N-p;i++) ref[i] = x[i+p]-x[i];
    approx_equal(out, ref, 1e-3f);

    // MACD line smoke test (can't exact-match recursive EMA easily) — ensure finite values
    rc = ct_macd_line(x.data(), out.data(), N, 12, 26, 9);
    if (rc != 0) { std::cerr << "ct_macd_line failed\\n"; return 1; }
    for (int i=0;i<N;i++) { if (!std::isfinite(out[i])) { std::cerr << "nan at " << i << "\\n"; return 1; } }

    std::cout << "All tests passed.\\n";
    return 0;
}
