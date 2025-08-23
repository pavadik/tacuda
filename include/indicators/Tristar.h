#ifndef TRISTAR_H
#define TRISTAR_H

#include "Indicator.h"

class Tristar : public Indicator {
public:
  Tristar() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif

