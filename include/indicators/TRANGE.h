#ifndef TRANGE_H
#define TRANGE_H

#include "Indicator.h"

class TRANGE : public Indicator {
public:
  TRANGE() = default;
  void calculate(const float *high, const float *low, const float *close,
                 float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
