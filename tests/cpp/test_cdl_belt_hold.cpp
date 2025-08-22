#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>

TEST(Tacuda, CDLBeltHold) {
  const int N = 3;
  std::vector<float> open{10.0f, 11.0f, 12.0f};
  std::vector<float> high{10.8f, 11.5f, 12.5f};
  std::vector<float> low{10.0f, 10.8f, 11.7f};
  std::vector<float> close{10.7f, 11.2f, 12.3f};
  std::vector<float> out(N), ref{1.0f, 0.0f, 0.0f};
  ctStatus_t rc = ct_cdl_belt_hold(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_belt_hold failed";
  expect_approx_equal(out, ref);
}

