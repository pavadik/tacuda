#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Stochastic) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                            49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                              49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> k(N, 0.0f), d(N, 0.0f);

  int kP = 5, dP = 3;
  ctStatus_t rc = ct_stochastic(high.data(), low.data(), close.data(), k.data(),
                                d.data(), N, kP, dP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stochastic failed";

  std::vector<float> refK = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             86.45833f,
                             97.29730f,
                             99.40476f,
                             81.93548f,
                             40.60150f,
                             48.12030f,
                             65.89147f,
                             75.19380f,
                             84.24658f};
  std::vector<float> refD = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             89.94871f,
                             93.85261f,
                             94.38680f,
                             92.87918f,
                             73.98058f,
                             56.88576f,
                             51.53776f,
                             63.06852f,
                             75.11062f};

  expect_approx_equal(k, refK);
  expect_approx_equal(d, refD);
  for (int i = 0; i < kP + dP - 2; ++i) {
    EXPECT_TRUE(std::isnan(k[i])) << "expected NaN at head " << i;
    EXPECT_TRUE(std::isnan(d[i])) << "expected NaN at head " << i;
  }
}
