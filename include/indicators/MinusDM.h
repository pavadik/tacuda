#ifndef MINUSDM_H
#define MINUSDM_H

#include "Indicator.h"

class MinusDM : public Indicator {
public:
  explicit MinusDM(int period);
  void calculate(const float *high, const float *low, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
