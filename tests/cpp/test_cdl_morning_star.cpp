#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLMorningStar) {
  const int N = 3;
  std::vector<float> open{10.0f, 9.3f, 9.4f};
  std::vector<float> high{10.5f, 9.5f, 10.6f};
  std::vector<float> low{9.4f, 9.2f, 9.2f};
  std::vector<float> close{9.5f, 9.25f, 10.1f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_morning_star(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_morning_star failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
  EXPECT_TRUE(std::isnan(out[1])) << "expected NaN at head 1";
}
