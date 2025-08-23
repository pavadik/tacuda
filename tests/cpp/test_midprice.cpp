#include <tacuda.h>
#include "test_utils.hpp"
#include <cmath>

TEST(Tacuda, MIDPRICE) {
  const int N = 40;
  std::vector<float> high(N), low(N);
  for (int i = 0; i < N; ++i) {
    high[i] = 10.0f + std::sin(0.1f * i);
    low[i] = high[i] - 1.0f;
  }
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_midprice(high.data(), low.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float maxH = high[i];
    float minL = low[i];
    for (int k = 1; k < p; ++k) {
      maxH = std::max(maxH, high[i + k]);
      minL = std::min(minL, low[i + k]);
    }
    ref[i] = 0.5f * (maxH + minL);
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i]));
}
