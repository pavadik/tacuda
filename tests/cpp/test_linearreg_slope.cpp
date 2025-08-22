#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, LINEARREG_SLOPE) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.02f * i);
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_linearreg_slope(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_linearreg_slope failed";
  auto ref = linearreg_slope_ref(x, p);
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}
