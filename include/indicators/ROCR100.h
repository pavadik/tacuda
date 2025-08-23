#ifndef ROCR100_H
#define ROCR100_H

#include "Indicator.h"

class ROCR100 : public Indicator {
public:
  explicit ROCR100(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
