#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLHikkakeMod) {
  const int N = 5;
  std::vector<float> open{10.5f, 10.6f, 10.5f, 10.8f, 10.9f};
  std::vector<float> high{11.0f, 10.8f, 11.1f, 11.3f, 11.2f};
  std::vector<float> low{10.0f, 10.2f, 10.4f, 10.7f, 10.8f};
  std::vector<float> close{10.6f, 10.4f, 10.9f, 11.2f, 11.0f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = std::numeric_limits<float>::quiet_NaN();
  ref[3] = 1.0f;
  ctStatus_t rc = ct_cdl_hikkake_mod(open.data(), high.data(), low.data(),
                                     close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_hikkake_mod failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_TRUE(std::isnan(out[2]));
}

