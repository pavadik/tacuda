#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>

TEST(Tacuda, CDLHighWave) {
  const int N = 4;
  std::vector<float> open{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> high{1.2f, 2.3f, 4.5f, 4.1f};
  std::vector<float> low{0.8f, 1.7f, 1.5f, 3.8f};
  std::vector<float> close{1.1f, 2.1f, 3.1f, 3.9f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_high_wave(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_high_wave failed";
  expect_approx_equal(out, ref);
}

