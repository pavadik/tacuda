#include "test_utils.hpp"
#include <tacuda.h>

static void ema_ref(const std::vector<float> &in, std::vector<float> &out,
                    int size, int period) {
  const float k = 2.0f / (period + 1.0f);
  for (int idx = 0; idx <= size - period; ++idx) {
    float weight = 1.0f;
    float weightedSum = in[idx + period - 1];
    float weightSum = 1.0f;
    for (int i = 1; i < period; ++i) {
      weight *= (1.0f - k);
      weightedSum += in[idx + period - 1 - i] * weight;
      weightSum += weight;
    }
    out[idx] = weightedSum / weightSum;
  }
}

TEST(Tacuda, T3) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  float v = 0.7f;
  ctStatus_t rc = ct_t3(x.data(), out.data(), N, p, v);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_t3 failed";

  std::vector<float> e1(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> e2(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> e3(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> e4(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> e5(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> e6(N, std::numeric_limits<float>::quiet_NaN());

  ema_ref(x, e1, N, p);
  int size2 = N - p + 1;
  ema_ref(e1, e2, size2, p);
  int size3 = size2 - p + 1;
  ema_ref(e2, e3, size3, p);
  int size4 = size3 - p + 1;
  ema_ref(e3, e4, size4, p);
  int size5 = size4 - p + 1;
  ema_ref(e4, e5, size5, p);
  int size6 = size5 - p + 1;
  ema_ref(e5, e6, size6, p);

  float b = v;
  float c1 = -b * b * b;
  float c2 = 3 * b * b + 3 * b * b * b;
  float c3 = -3 * b - 6 * b * b - 3 * b * b * b;
  float c4 = 1 + 3 * b + 3 * b * b + b * b * b;

  int valid = N - 3 * p + 3;
  for (int i = 0; i < valid; ++i) {
    int p1 = p - 1;
    ref[i] = c1 * e6[i + 3 * p1] + c2 * e5[i + 2 * p1] + c3 * e4[i + p1] +
             c4 * e3[i];
  }
  expect_approx_equal(out, ref);
  for (int i = valid; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
