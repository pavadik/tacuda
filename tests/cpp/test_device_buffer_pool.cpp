#include <tacuda.h>
#include <utils/DeviceBufferPool.h>
#include <gtest/gtest.h>
#include <vector>

TEST(DeviceBufferPool, ReusesBuffersInIndicators) {
  const int N = 128;
  std::vector<float> in(N, 1.0f), out(N, 0.0f);
  auto &pool = DeviceBufferPool::instance();
  pool.cleanup();

  ctStatus_t rc = ct_sma(in.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_first = pool.allocationCount();
  EXPECT_GT(after_first, 0u);

  rc = ct_sma(in.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_second = pool.allocationCount();
  EXPECT_EQ(after_first, after_second);

  pool.cleanup();
}
