#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, DX) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 3;
  ctStatus_t rc = ct_dx(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_dx failed";
  for (int i = 0; i < N - p; ++i) {
    float prevHigh = high[i];
    float prevLow = low[i];
    float prevClose = close[i];
    float dmp = 0.0f, dmm = 0.0f, tr = 0.0f;
    for (int j = 1; j <= p; ++j) {
      float curHigh = high[i + j];
      float curLow = low[i + j];
      float upMove = curHigh - prevHigh;
      float downMove = prevLow - curLow;
      float dmPlus = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
      float dmMinus = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
      float trVal = std::max(curHigh - curLow,
                              std::max(std::fabs(curHigh - prevClose),
                                       std::fabs(curLow - prevClose)));
      dmp += dmPlus;
      dmm += dmMinus;
      tr += trVal;
      prevHigh = curHigh;
      prevLow = curLow;
      prevClose = close[i + j];
    }
    float dip = (tr == 0.0f) ? 0.0f : 100.0f * dmp / tr;
    float dim = (tr == 0.0f) ? 0.0f : 100.0f * dmm / tr;
    float denom = dip + dim;
    ref[i] = (denom == 0.0f) ? 0.0f : 100.0f * std::fabs(dip - dim) / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}
