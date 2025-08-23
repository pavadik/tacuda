#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, ROCR) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = 1.0f + std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_rocr(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i < N - p; ++i)
    ref[i] = x[i + p] / x[i];
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i]));
}
