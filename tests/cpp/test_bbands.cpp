#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, BBANDS) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> upper(N, 0.0f), middle(N, 0.0f), lower(N, 0.0f);
  std::vector<float> refU(N, 0.0f), refM(N, 0.0f), refL(N, 0.0f);

  int p = 20;
  float up = 2.0f, down = 2.0f;
  ctStatus_t rc = ct_bbands(x.data(), upper.data(), middle.data(), lower.data(),
                            N, p, up, down);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_bbands failed";

  for (int i = 0; i <= N - p; ++i) {
    float sum = 0.0f, sumSq = 0.0f;
    for (int j = 0; j < p; ++j) {
      float v = x[i + j];
      sum += v;
      sumSq += v * v;
    }
    float mean = sum / p;
    float var = sumSq / p - mean * mean;
    var = std::max(var, 0.0f);
    float stddev = std::sqrt(var);
    refM[i] = mean;
    refU[i] = mean + up * stddev;
    refL[i] = mean - down * stddev;
  }

  expect_approx_equal(upper, refU);
  expect_approx_equal(middle, refM);
  expect_approx_equal(lower, refL);

  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(upper[i])) << "expected NaN at tail " << i;
    EXPECT_TRUE(std::isnan(middle[i])) << "expected NaN at tail " << i;
    EXPECT_TRUE(std::isnan(lower[i])) << "expected NaN at tail " << i;
  }
}
