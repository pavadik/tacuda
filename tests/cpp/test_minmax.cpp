#include "test_utils.hpp"
#include <cmath>
#include <tacuda.h>

TEST(Tacuda, MINMAX) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> outMin(N, 0.0f), outMax(N, 0.0f);
  std::vector<float> refMin(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refMax(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_minmax(x.data(), outMin.data(), outMax.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float mn = x[i];
    float mx = x[i];
    for (int k = 1; k < p; ++k) {
      mn = std::min(mn, x[i + k]);
      mx = std::max(mx, x[i + k]);
    }
    refMin[i] = mn;
    refMax[i] = mx;
  }
  expect_approx_equal(outMin, refMin);
  expect_approx_equal(outMax, refMax);
  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(outMin[i]));
    EXPECT_TRUE(std::isnan(outMax[i]));
  }
}
