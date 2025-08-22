#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, ADX) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 3;
  ctStatus_t rc =
      ct_adx(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adx failed";

  std::vector<float> dmp(N, 0.0f), dmm(N, 0.0f), tr(N, 0.0f), dx(N, 0.0f);
  for (int i = 1; i < N; ++i) {
    float upMove = high[i] - high[i - 1];
    float downMove = low[i - 1] - low[i];
    dmp[i] = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
    dmm[i] = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float dmp_s = 0.0f, dmm_s = 0.0f, tr_s = 0.0f;
  for (int i = 1; i <= p; ++i) {
    dmp_s += dmp[i];
    dmm_s += dmm[i];
    tr_s += tr[i];
  }
  float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
  float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
  dx[p] =
      (dip + dim == 0.0f) ? 0.0f : 100.0f * std::fabs(dip - dim) / (dip + dim);
  float dx_sum = dx[p];
  for (int i = p + 1; i < N; ++i) {
    dmp_s = dmp_s - dmp_s / p + dmp[i];
    dmm_s = dmm_s - dmm_s / p + dmm[i];
    tr_s = tr_s - tr_s / p + tr[i];
    dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
    dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
    dx[i] = (dip + dim == 0.0f) ? 0.0f
                                : 100.0f * std::fabs(dip - dim) / (dip + dim);
    if (i < 2 * p) {
      dx_sum += dx[i];
      if (i == 2 * p - 1)
        ref[i] = dx_sum / p;
    } else {
      ref[i] = ((ref[i - 1] * (p - 1)) + dx[i]) / p;
    }
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < 2 * p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}
