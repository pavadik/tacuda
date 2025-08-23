#ifndef SUM_H
#define SUM_H

#include "Indicator.h"

class SUM : public Indicator {
public:
  explicit SUM(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

#endif
