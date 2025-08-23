#ifndef PLUSDI_H
#define PLUSDI_H

#include "Indicator.h"

class PlusDI : public Indicator {
public:
  explicit PlusDI(int period);
  void calculate(const float *high, const float *low, const float *close,
                 float *output, int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
