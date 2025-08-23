#ifndef KAMA_H
#define KAMA_H

#include "Indicator.h"

class KAMA : public Indicator {
public:
  KAMA(int period, int fastPeriod, int slowPeriod);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
  float fastSC;
  float slowSC;
};

#endif
