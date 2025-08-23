#ifndef ONNECK_H
#define ONNECK_H

#include "Indicator.h"

class OnNeck : public Indicator {
public:
  OnNeck() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
