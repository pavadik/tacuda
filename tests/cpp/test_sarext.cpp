#include <tacuda.h>
#include "test_utils.hpp"

static std::vector<float> sarext_ref(const std::vector<float>& high,
                                     const std::vector<float>& low,
                                     float startValue, float offset,
                                     float accInitLong, float accLong, float accMaxLong,
                                     float accInitShort, float accShort, float accMaxShort) {
  size_t n = high.size();
  std::vector<float> out(n);
  float sar = (startValue != 0.0f) ? startValue : low[0];
  bool longPos = true;
  float af = accInitLong;
  float ep = high[0];
  out[0] = sar;
  for (size_t i = 1; i < n; ++i) {
    sar = sar + af * (ep - sar);
    if (longPos) {
      sar = std::min(sar, low[i - 1]);
      if (low[i] < sar) {
        longPos = false;
        sar = ep + offset;
        ep = low[i];
        af = accInitShort;
        sar = std::max(sar, high[i - 1]);
      } else if (high[i] > ep) {
        ep = high[i];
        af = std::min(af + accLong, accMaxLong);
      }
    } else {
      sar = std::max(sar, high[i - 1]);
      if (high[i] > sar) {
        longPos = true;
        sar = ep - offset;
        ep = high[i];
        af = accInitLong;
        sar = std::min(sar, low[i - 1]);
      } else if (low[i] < ep) {
        ep = low[i];
        af = std::min(af + accShort, accMaxShort);
      }
    }
    out[i] = sar;
  }
  return out;
}

TEST(Tacuda, SAREXTTrending) {
  std::vector<float> high = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sarext(high.data(), low.data(), out.data(), N,
                            0.0f, 0.0f, 0.02f, 0.02f, 0.2f,
                            0.02f, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sarext failed";
  auto ref = sarext_ref(high, low, 0.0f, 0.0f, 0.02f, 0.02f, 0.2f,
                        0.02f, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, SAREXTRanging) {
  std::vector<float> high = {5.f, 6.f, 5.5f, 6.2f, 5.8f, 6.4f, 5.9f, 6.5f};
  std::vector<float> low = {4.f, 5.f, 4.5f, 5.2f, 4.8f, 5.4f, 4.9f, 5.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sarext(high.data(), low.data(), out.data(), N,
                            0.0f, 0.0f, 0.02f, 0.02f, 0.2f,
                            0.02f, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sarext failed";
  auto ref = sarext_ref(high, low, 0.0f, 0.0f, 0.02f, 0.02f, 0.2f,
                        0.02f, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}
