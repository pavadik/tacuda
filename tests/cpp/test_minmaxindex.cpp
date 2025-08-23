#include "test_utils.hpp"
#include <cmath>
#include <tacuda.h>

TEST(Tacuda, MINMAXINDEX) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> outMin(N, 0.0f), outMax(N, 0.0f);
  std::vector<float> refMin(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refMax(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_minmaxindex(x.data(), outMin.data(), outMax.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float mn = x[i];
    float mx = x[i];
    int minIdx = 0;
    int maxIdx = 0;
    for (int k = 1; k < p; ++k) {
      float v = x[i + k];
      if (v < mn) {
        mn = v;
        minIdx = k;
      }
      if (v > mx) {
        mx = v;
        maxIdx = k;
      }
    }
    refMin[i] = static_cast<float>(minIdx);
    refMax[i] = static_cast<float>(maxIdx);
  }
  expect_approx_equal(outMin, refMin);
  expect_approx_equal(outMax, refMax);
  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(outMin[i]));
    EXPECT_TRUE(std::isnan(outMax[i]));
  }
}
