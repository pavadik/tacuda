#include <tacuda.h>
#include "test_utils.hpp"
#include <cmath>

TEST(Tacuda, MIDPOINT) {
  const int N = 40;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.1f * i);
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 7;
  ctStatus_t rc = ct_midpoint(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float mn = x[i], mx = x[i];
    for (int k = 1; k < p; ++k) {
      float v = x[i + k];
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    ref[i] = 0.5f * (mx + mn);
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i]));
}
