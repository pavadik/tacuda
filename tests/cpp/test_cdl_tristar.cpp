#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLTristar) {
  const int N = 5;
  std::vector<float> open{10.0f, 9.0f, 9.4f, 9.5f, 9.6f};
  std::vector<float> high{10.2f, 9.2f, 9.6f, 9.7f, 9.8f};
  std::vector<float> low{9.8f, 8.8f, 9.2f, 9.4f, 9.5f};
  std::vector<float> close{10.01f, 9.01f, 9.41f, 9.55f, 9.65f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_tristar(open.data(), high.data(), low.data(),
                                 close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_tristar failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

