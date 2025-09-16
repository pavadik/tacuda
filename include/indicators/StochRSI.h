#ifndef STOCHRSI_H
#define STOCHRSI_H

#include "Indicator.h"

namespace tacuda {
class StochRSI : public Indicator {
public:
  StochRSI(int rsiPeriod, int kPeriod, int dPeriod);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int rsiPeriod;
  int kPeriod;
  int dPeriod;
};

} // namespace tacuda

#endif
