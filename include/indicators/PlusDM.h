#ifndef PLUSDM_H
#define PLUSDM_H

#include "Indicator.h"

class PlusDM : public Indicator {
public:
  explicit PlusDM(int period);
  void calculate(const float *high, const float *low, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
