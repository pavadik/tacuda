#ifndef TRIMA_H
#define TRIMA_H

#include "Indicator.h"

class TRIMA : public Indicator {
public:
  explicit TRIMA(int period);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int period;
};

#endif
