#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, DEMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_dema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_dema failed";

  auto ref = dema_ref(x, p);
  expect_approx_equal(out, ref);
  for (int i = N - 2 * p + 2; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
