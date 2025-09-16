#ifndef WILLR_H
#define WILLR_H

#include "Indicator.h"

namespace tacuda {
class WILLR : public Indicator {
public:
  explicit WILLR(int period);
  void calculate(const float *high, const float *low, const float *close,
                 float *output, int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

} // namespace tacuda

#endif
