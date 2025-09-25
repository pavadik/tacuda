#include <tacuda.h>
#include "test_utils.hpp"

static std::vector<float> ema_ref(const std::vector<float>& in, int period) {
  std::vector<float> out(in.size(), std::numeric_limits<float>::quiet_NaN());
  const float k = 2.0f / (period + 1.0f);
  for (size_t i = 0; i + period <= in.size(); ++i) {
    float weight = 1.0f;
    float weightedSum = in[i + period - 1];
    float weightSum = 1.0f;
    for (int j = 1; j < period; ++j) {
      weight *= (1.0f - k);
      weightedSum += in[i + period - 1 - j] * weight;
      weightSum += weight;
    }
    out[i] = weightedSum / weightSum;
  }
  return out;
}

TEST(Tacuda, MA_SMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_ma(x.data(), out.data(), N, p, CT_MA_SMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float s = 0.0f;
    for (int k = 0; k < p; ++k)
      s += x[i + k];
    ref[i] = s / p;
  }
  expect_approx_equal(out, ref);
}

TEST(Tacuda, MA_EMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_ma(x.data(), out.data(), N, p, CT_MA_EMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  auto ref = ema_ref(x, p);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, MA_WMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.05f * i);
  std::vector<float> maOut(N, 0.0f), ref(N, 0.0f);
  int p = 4;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_WMA), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_wma(x.data(), ref.data(), N, p), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, ref);
}

TEST(Tacuda, MA_DEMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.07f * i);
  std::vector<float> maOut(N, 0.0f), ref(N, 0.0f);
  int p = 6;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_DEMA), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_dema(x.data(), ref.data(), N, p), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, ref);
}

TEST(Tacuda, MA_TEMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.09f * i);
  std::vector<float> maOut(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_TEMA), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_tema(x.data(), ref.data(), N, p), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, ref);
}

TEST(Tacuda, MA_TRIMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.03f * i);
  std::vector<float> maOut(N, 0.0f), ref(N, 0.0f);
  int p = 7;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_TRIMA), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_trima(x.data(), ref.data(), N, p), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, ref);
}

TEST(Tacuda, MA_KAMA) {
  const int N = 48;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.04f * i) + 0.1f * i;
  std::vector<float> maOut(N, 0.0f), ref(N, 0.0f);
  int p = 10;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_KAMA), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_kama(x.data(), ref.data(), N, p, 2, 30), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, ref);
}

TEST(Tacuda, MA_MAMA) {
  const int N = 40;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.02f * i) + 0.01f * i;
  std::vector<float> maOut(N, 0.0f), mama(N, 0.0f), fama(N, 0.0f);
  int p = 6;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_MAMA), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_mama(x.data(), mama.data(), fama.data(), N, 0.5f, 0.05f), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, mama);
}

TEST(Tacuda, MA_T3) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.11f * i);
  std::vector<float> maOut(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ASSERT_EQ(ct_ma(x.data(), maOut.data(), N, p, CT_MA_T3), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_t3(x.data(), ref.data(), N, p, 0.7f), CT_STATUS_SUCCESS);
  expect_approx_equal(maOut, ref);
}
