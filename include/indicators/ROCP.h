#ifndef ROCP_H
#define ROCP_H

#include "Indicator.h"

class ROCP : public Indicator {
public:
  explicit ROCP(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
