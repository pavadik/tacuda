#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, PPO) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f);
  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_ppo(x.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ppo failed";
  auto ref = ppo_ref(x, fastP, slowP);
  expect_approx_equal(out, ref);
  for (int i = 0; i < slowP; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
