#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLAdvanceBlock) {
  const int N = 5;
  std::vector<float> open{10.0f, 10.5f, 10.9f, 10.0f, 10.0f};
  std::vector<float> high{10.9f, 11.2f, 11.4f, 10.1f, 10.1f};
  std::vector<float> low{9.9f, 10.4f, 10.8f, 9.9f, 9.9f};
  std::vector<float> close{10.8f, 11.1f, 11.3f, 10.05f, 10.05f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_advance_block(open.data(), high.data(), low.data(),
                                       close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_advance_block failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
  EXPECT_TRUE(std::isnan(out[1])) << "expected NaN at head 1";
}

