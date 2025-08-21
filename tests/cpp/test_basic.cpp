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
    ctStatus_t rc = ct_sma(x.data(), out.data(), N, p);
    if (rc != CT_STATUS_SUCCESS) { std::cerr << "ct_sma failed\\n"; return 1; }
    for (int i=0;i<=N-p;i++) {
        float s=0; for (int k=0;k<p;k++) s+=x[i+k];
        ref[i] = s/p;
    }
    approx_equal(out, ref, 1e-3f);
    for (int i=N-p+1;i<N;i++) {
        if (!std::isnan(out[i])) { std::cerr << "expected NaN at tail " << i << "\n"; return 1; }
    }

    // Momentum check
    std::fill(ref.begin(), ref.end(), 0.0f);
    rc = ct_momentum(x.data(), out.data(), N, p);
    if (rc != CT_STATUS_SUCCESS) { std::cerr << "ct_momentum failed\\n"; return 1; }
    for (int i=0;i<N-p;i++) ref[i] = x[i+p]-x[i];
    approx_equal(out, ref, 1e-3f);
    for (int i=N-p;i<N;i++) {
        if (!std::isnan(out[i])) { std::cerr << "expected NaN at tail " << i << "\n"; return 1; }
    }

    // MACD line smoke test — first `slowPeriod` samples should be NaN, rest finite
    int fastP = 12, slowP = 26;
    rc = ct_macd_line(x.data(), out.data(), N, fastP, slowP);
    if (rc != CT_STATUS_SUCCESS) { std::cerr << "ct_macd_line failed\\n"; return 1; }
    for (int i=0;i<slowP;i++) {
        if (!std::isnan(out[i])) { std::cerr << "expected NaN at head " << i << "\\n"; return 1; }
    }
    for (int i=slowP;i<N;i++) {
        if (!std::isfinite(out[i])) { std::cerr << "nan at " << i << "\\n"; return 1; }
    }

    std::cout << "All tests passed.\\n";
    return 0;
}
