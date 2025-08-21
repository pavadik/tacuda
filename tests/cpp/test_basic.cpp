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

TEST(Tacuda, EMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_ema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ema failed";

  const float k = 2.0f / (p + 1.0f);
  for (int i = 0; i <= N - p; ++i) {
    float weight = 1.0f;
    float weightedSum = x[i + p - 1];
    float weightSum = 1.0f;
    for (int j = 1; j < p; ++j) {
      weight *= (1.0f - k);
      weightedSum += x[i + p - 1 - j] * weight;
      weightSum += weight;
    }
    ref[i] = weightedSum / weightSum;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i) {
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

TEST(Tacuda, MacdLine) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_macd_line(x.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_macd_line failed";
  for (int i = 0; i < slowP; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
  for (int i = slowP; i < N; i++) {
    EXPECT_TRUE(std::isfinite(out[i])) << "expected finite value at " << i;
  }
}

TEST(Tacuda, RSI) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 14;
  ctStatus_t rc = ct_rsi(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_rsi failed";
  for (int i = 0; i < N - p; ++i) {
    float gain = 0.0f;
    float loss = 0.0f;
    for (int j = 0; j < p; ++j) {
      float diff = x[i + j + 1] - x[i + j];
      if (diff > 0.0f)
        gain += diff;
      else
        loss -= diff;
    }
    float avgGain = gain / p;
    float avgLoss = loss / p;
    float rsi;
    if (avgLoss == 0.0f)
      rsi = (avgGain == 0.0f) ? 50.0f : 100.0f;
    else if (avgGain == 0.0f)
      rsi = 0.0f;
    else {
      float rs = avgGain / avgLoss;
      rsi = 100.0f - 100.0f / (1.0f + rs);
    }
    ref[i] = rsi;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
