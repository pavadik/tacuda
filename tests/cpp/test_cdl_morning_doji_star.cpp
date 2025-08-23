#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLMorningDojiStar) {
  const int N = 3;
  std::vector<float> open{10.0f, 9.3f, 9.4f};
  std::vector<float> high{10.5f, 9.4f, 10.6f};
  std::vector<float> low{9.5f, 9.1f, 9.3f};
  std::vector<float> close{9.6f, 9.3f, 10.2f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_morning_doji_star(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_morning_doji_star failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
  EXPECT_TRUE(std::isnan(out[1])) << "expected NaN at head 1";
}
