#ifndef ROCR_H
#define ROCR_H

#include "Indicator.h"

class ROCR : public Indicator {
public:
  explicit ROCR(int period);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int period;
};

#endif
