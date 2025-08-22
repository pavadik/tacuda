#ifndef PVO_H
#define PVO_H

#include "Indicator.h"

class PVO : public Indicator {
public:
  PVO(int fastPeriod, int slowPeriod);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int fastPeriod;
  int slowPeriod;
};

#endif
