#ifndef PIERCING_H
#define PIERCING_H

#include "Indicator.h"

class Piercing : public Indicator {
public:
  Piercing() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
