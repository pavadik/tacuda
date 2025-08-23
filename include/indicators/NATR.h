#ifndef NATR_H
#define NATR_H

#include "Indicator.h"

class NATR : public Indicator {
public:
  explicit NATR(int period);
  void calculate(const float *high, const float *low, const float *close,
                 float *output, int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
