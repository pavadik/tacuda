#ifndef MINUSDI_H
#define MINUSDI_H

#include "Indicator.h"

class MinusDI : public Indicator {
public:
  explicit MinusDI(int period);
  void calculate(const float *high, const float *low, const float *close,
                 float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int period;
};

#endif
