#include <limits>

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

TEST(DeviceBufferPool, ReusesBuffersInMultiInputWrappers) {
  const int N = 64;
  std::vector<float> high(N, 2.0f), low(N, 1.0f), out(N, 0.0f);
  auto &pool = DeviceBufferPool::instance();
  pool.cleanup();

  ctStatus_t rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_first = pool.allocationCount();
  EXPECT_GT(after_first, 0u);

  rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_second = pool.allocationCount();
  EXPECT_EQ(after_first, after_second);

  pool.cleanup();
}

TEST(DeviceBufferPool, RespectsCacheLimit) {
  auto &pool = DeviceBufferPool::instance();
  pool.cleanup();

  void *ptr = pool.acquire(1024);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(pool.allocationCount(), 1u);

  pool.setMaxCacheBytes(0);
  pool.release(ptr);

  void *ptr2 = pool.acquire(1024);
  EXPECT_EQ(pool.allocationCount(), 2u);
  pool.release(ptr2);

  pool.cleanup();
  pool.setMaxCacheBytes(std::numeric_limits<size_t>::max());
}
