#ifndef PPO_H
#define PPO_H

#include "Indicator.h"

namespace tacuda {
class PPO : public Indicator {
public:
  PPO(int fastPeriod, int slowPeriod);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int fastPeriod;
  int slowPeriod;
};

} // namespace tacuda

#endif
