#include <tacuda.h>
#include "test_utils.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

TEST(Tacuda, AvgDev) {
  const int N = 6;
  const int period = 3;
  std::vector<float> data{1.0f, 2.0f, 4.0f, 7.0f, 11.0f, 16.0f};
  std::vector<float> out(N, 0.0f);
  ASSERT_EQ(ct_avgdev(data.data(), out.data(), N, period), CT_STATUS_SUCCESS);
  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = 0; i <= N - period; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < period; ++j) sum += data[i + j];
    float mean = sum / period;
    float dev = 0.0f;
    for (int j = 0; j < period; ++j) dev += std::fabs(data[i + j] - mean);
    ref[i] = dev / period;
  }
  expect_approx_equal(out, ref);
  for (int i = N - period + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i]));
  }
}

TEST(Tacuda, MAVP) {
  {
    const int N = 8;
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> periods{1.0f, 3.0f, 5.0f, 2.0f, 4.5f, 10.0f, 2.0f, 1.0f};
    const int minPeriod = 2;
    const int maxPeriod = 4;
    std::array<ctMaType_t, 9> types = {
        CT_MA_SMA,  CT_MA_EMA,  CT_MA_WMA, CT_MA_DEMA, CT_MA_TEMA,
        CT_MA_TRIMA, CT_MA_KAMA, CT_MA_MAMA, CT_MA_T3};

    for (ctMaType_t type : types) {
      std::vector<float> out(N, 0.0f);
      ASSERT_EQ(ct_mavp(data.data(), periods.data(), out.data(), N, minPeriod,
                        maxPeriod, type),
                CT_STATUS_SUCCESS);

      std::vector<float> ma2(N, 0.0f), ma3(N, 0.0f), ma4(N, 0.0f);
      ASSERT_EQ(ct_ma(data.data(), ma2.data(), N, 2, type), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ma(data.data(), ma3.data(), N, 3, type), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ma(data.data(), ma4.data(), N, 4, type), CT_STATUS_SUCCESS);

      std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
      for (int i = 0; i < N; ++i) {
        int p = static_cast<int>(periods[i]);
        if (p < minPeriod) p = minPeriod;
        if (p > maxPeriod) p = maxPeriod;
        if (p == 2 && i <= N - p) {
          ref[i] = ma2[i];
        } else if (p == 3 && i <= N - p) {
          ref[i] = ma3[i];
        } else if (p == 4 && i <= N - p) {
          ref[i] = ma4[i];
        }
      }

      expect_approx_equal(out, ref);
      EXPECT_TRUE(std::isnan(out[5]));
      EXPECT_TRUE(std::isnan(out[7]));
    }
  }

  {
    const int N = 5;
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> periods{2.0f, 2.5f, 3.0f, 3.0f, 2.0f};
    std::vector<float> out(N, 0.0f);
    ASSERT_EQ(ct_mavp(data.data(), periods.data(), out.data(), N, 2, 3,
                      CT_MA_SMA),
              CT_STATUS_SUCCESS);
    std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
    ref[0] = (data[0] + data[1]) / 2.0f;
    ref[1] = (data[1] + data[2]) / 2.0f;
    ref[2] = (data[2] + data[3] + data[4]) / 3.0f;
    expect_approx_equal(out, ref);
    EXPECT_TRUE(std::isnan(out[3]));
    EXPECT_TRUE(std::isnan(out[4]));
  }
}

TEST(Tacuda, NVI) {
  std::vector<float> close{1.0f, 2.0f, 1.5f, 1.8f};
  std::vector<float> volume{100.0f, 90.0f, 95.0f, 80.0f};
  std::vector<float> out(close.size(), 0.0f);
  ASSERT_EQ(ct_nvi(close.data(), volume.data(), out.data(), close.size()),
            CT_STATUS_SUCCESS);
  std::vector<float> ref{1000.0f, 2000.0f, 2000.0f,
                         2000.0f * (1.0f + (1.8f - 1.5f) / 1.5f)};
  expect_approx_equal(out, ref);
}

TEST(Tacuda, PVI) {
  std::vector<float> close{1.0f, 1.2f, 1.1f, 1.3f};
  std::vector<float> volume{100.0f, 120.0f, 110.0f, 150.0f};
  std::vector<float> out(close.size(), 0.0f);
  ASSERT_EQ(ct_pvi(close.data(), volume.data(), out.data(), close.size()),
            CT_STATUS_SUCCESS);
  std::vector<float> ref(close.size(), 0.0f);
  ref[0] = 1000.0f;
  ref[1] = 1000.0f * (1.0f + (1.2f - 1.0f) / 1.0f);
  ref[2] = ref[1];
  ref[3] = ref[2] * (1.0f + (1.3f - 1.1f) / 1.1f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, IMI) {
  const int N = 5;
  std::vector<float> open{1.0f, 2.0f, 1.0f, 1.5f, 1.2f};
  std::vector<float> close{1.5f, 1.5f, 0.8f, 1.6f, 1.0f};
  std::vector<float> out(N, 0.0f);
  ASSERT_EQ(ct_imi(open.data(), close.data(), out.data(), N, 3), CT_STATUS_SUCCESS);
  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  ref[0] = 100.0f * (0.5f / 1.2f);
  ref[1] = 100.0f * (0.1f / 0.8f);
  ref[2] = 100.0f * (0.1f / 0.5f);
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[3]));
  EXPECT_TRUE(std::isnan(out[4]));
}

TEST(Tacuda, ACCBANDS) {
  const int N = 6;
  const int period = 3;
  std::vector<float> high{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> low{9.0f, 9.5f, 10.0f, 11.0f, 12.0f, 13.0f};
  std::vector<float> close{9.5f, 10.5f, 11.0f, 12.5f, 13.0f, 14.0f};
  std::vector<float> upper(N), middle(N), lower(N);
  ASSERT_EQ(ct_accbands(high.data(), low.data(), close.data(), upper.data(),
                        middle.data(), lower.data(), N, period), CT_STATUS_SUCCESS);
  std::vector<float> rawUpper(N), rawLower(N);
  for (int i = 0; i < N; ++i) {
    float h = high[i];
    float l = low[i];
    float sum = h + l;
    if (sum != 0.0f) {
      float factor = 4.0f * (h - l) / sum;
      rawUpper[i] = h * (1.0f + factor);
      rawLower[i] = l * (1.0f - factor);
    } else {
      rawUpper[i] = h;
      rawLower[i] = l;
    }
  }
  std::vector<float> refUpper(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refMiddle(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refLower(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = 0; i <= N - period; ++i) {
    float sumU = 0.0f, sumM = 0.0f, sumL = 0.0f;
    for (int j = 0; j < period; ++j) {
      sumU += rawUpper[i + j];
      sumM += close[i + j];
      sumL += rawLower[i + j];
    }
    refUpper[i] = sumU / period;
    refMiddle[i] = sumM / period;
    refLower[i] = sumL / period;
  }
  expect_approx_equal(upper, refUpper);
  expect_approx_equal(middle, refMiddle);
  expect_approx_equal(lower, refLower);
}

TEST(Tacuda, CDLThreeOutside) {
  const int N = 5;
  std::vector<float> open{10.0f, 8.5f, 10.4f, 11.5f, 9.4f};
  std::vector<float> high{10.5f, 11.0f, 11.5f, 11.8f, 9.6f};
  std::vector<float> low{9.5f, 8.3f, 10.2f, 9.0f, 8.5f};
  std::vector<float> close{9.0f, 10.6f, 11.2f, 9.5f, 8.8f};
  std::vector<float> out(N, 0.0f);
  ASSERT_EQ(ct_cdl_three_outside(open.data(), high.data(), low.data(), close.data(),
                                out.data(), N, 0), CT_STATUS_SUCCESS);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_FLOAT_EQ(100.0f, out[2]);
  EXPECT_FLOAT_EQ(0.0f, out[3]);
  EXPECT_FLOAT_EQ(-100.0f, out[4]);
}
