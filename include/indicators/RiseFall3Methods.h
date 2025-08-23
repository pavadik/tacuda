#ifndef RISEFALL3METHODS_H
#define RISEFALL3METHODS_H

#include "Indicator.h"

class RiseFall3Methods : public Indicator {
public:
  RiseFall3Methods() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
