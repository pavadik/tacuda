#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLDojiStar) {
  const int N = 4;
  std::vector<float> open{1.0f, 1.0f, 1.6f, 1.5f};
  std::vector<float> high{1.1f, 1.5f, 1.65f, 1.6f};
  std::vector<float> low{0.9f, 0.95f, 1.55f, 1.45f};
  std::vector<float> close{1.05f, 1.4f, 1.6f, 1.55f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_doji_star(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_doji_star failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
}
