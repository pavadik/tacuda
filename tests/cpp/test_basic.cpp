#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <tacuda.h>
#include <vector>

namespace {

// Helper to compare floating point vectors, ignoring NaNs.
void expect_approx_equal(const std::vector<float> &a,
                         const std::vector<float> &b, float eps = 1e-3f) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i]))
      continue;
    EXPECT_NEAR(a[i], b[i], eps) << "Mismatch at index " << i;
  }
}

} // namespace

TEST(Tacuda, SMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_sma(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sma failed";
  for (int i = 0; i <= N - p; i++) {
    float s = 0.0f;
    for (int k = 0; k < p; k++)
      s += x[i + k];
    ref[i] = s / p;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, Momentum) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_momentum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_momentum failed";
  for (int i = 0; i < N - p; i++)
    ref[i] = x[i + p] - x[i];
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, Macd) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> line(N, 0.0f), signal(N, 0.0f), hist(N, 0.0f);

  int fastP = 12, slowP = 26, signalP = 9;
  ctStatus_t rc = ct_macd(x.data(), line.data(), signal.data(), hist.data(), N,
                          fastP, slowP, signalP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_macd failed";
  for (int i = 0; i < slowP; i++) {
    EXPECT_TRUE(std::isnan(line[i])) << "expected NaN at head " << i;
  }
  for (int i = slowP; i < N; i++) {
    EXPECT_TRUE(std::isfinite(line[i])) << "expected finite value at " << i;
  }
  int warm = slowP + signalP - 1;
  for (int i = 0; i < warm; ++i) {
    EXPECT_TRUE(std::isnan(signal[i])) << "expected NaN in signal at " << i;
    EXPECT_TRUE(std::isnan(hist[i])) << "expected NaN in hist at " << i;
  }
  for (int i = warm; i < N; ++i) {
    EXPECT_TRUE(std::isfinite(signal[i])) << "expected finite signal at " << i;
    EXPECT_TRUE(std::isfinite(hist[i])) << "expected finite hist at " << i;
    if (std::isfinite(line[i]) && std::isfinite(signal[i]))
      EXPECT_NEAR(hist[i], line[i] - signal[i], 1e-3f);
  }
}
