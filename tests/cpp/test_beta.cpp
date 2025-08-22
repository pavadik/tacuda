#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Beta) {
  const int N = 128;
  std::vector<float> x(N), y(N);
  for (int i = 0; i < N; ++i) {
    x[i] = std::sin(0.01f * i);
    y[i] = std::cos(0.02f * i);
  }
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_beta(x.data(), y.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_beta failed";
  for (int i = 0; i < N - p; ++i) {
    float sumX = 0.0f, sumY = 0.0f;
    for (int j = 0; j < p; ++j) {
      sumX += x[i + j];
      sumY += y[i + j];
    }
    float meanX = sumX / p;
    float meanY = sumY / p;
    float cov = 0.0f, varY = 0.0f;
    for (int j = 0; j < p; ++j) {
      float dx = x[i + j] - meanX;
      float dy = y[i + j] - meanY;
      cov += dx * dy;
      varY += dy * dy;
    }
    ref[i] = (varY == 0.0f) ? 0.0f : cov / varY;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}
