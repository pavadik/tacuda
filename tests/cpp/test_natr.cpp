#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, NATR) {
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
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 14;
  ctStatus_t rc =
      ct_natr(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_natr failed";

  std::vector<float> tr(N);
  tr[0] = high[0] - low[0];
  for (int i = 1; i < N; ++i) {
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float sum = 0.0f;
  for (int i = 0; i < p; ++i)
    sum += tr[i];
  float atr = sum / p;
  ref[p - 1] = 100.0f * atr / close[p - 1];
  for (int i = p; i < N; ++i) {
    atr = (atr * (p - 1) + tr[i]) / p;
    ref[i] = 100.0f * atr / close[i];
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}
