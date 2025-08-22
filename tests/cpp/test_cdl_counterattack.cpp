#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLCounterAttack) {
  const int N = 3;
  std::vector<float> open{1.0f, 2.5f, 3.0f};
  std::vector<float> high{1.5f, 2.6f, 3.5f};
  std::vector<float> low{0.5f, 1.0f, 2.5f};
  std::vector<float> close{1.4f, 1.4f, 3.2f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_counterattack(open.data(), high.data(), low.data(),
                                       close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_counterattack failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
}
