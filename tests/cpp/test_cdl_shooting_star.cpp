#include "test_utils.hpp"
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLShootingStar) {
  const int N = 3;
  std::vector<float> open{1.0f, 2.0f, 3.0f};
  std::vector<float> high{1.1f, 2.5f, 3.2f};
  std::vector<float> low{0.9f, 1.98f, 2.8f};
  std::vector<float> close{1.05f, 2.05f, 3.1f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_shooting_star(open.data(), high.data(), low.data(),
                                       close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_shooting_star failed";
  expect_approx_equal(out, ref);
}
