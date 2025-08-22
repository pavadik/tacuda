#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, CMO) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_cmo(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cmo failed";
  for (int i = 0; i < N - p; ++i) {
    float up = 0.0f, down = 0.0f;
    for (int j = 0; j < p; ++j) {
      float diff = x[i + j + 1] - x[i + j];
      if (diff > 0.0f)
        up += diff;
      else
        down -= diff;
    }
    float denom = up + down;
    ref[i] = (denom == 0.0f) ? 0.0f : 100.0f * (up - down) / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}
