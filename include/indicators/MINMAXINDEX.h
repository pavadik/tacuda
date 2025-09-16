#ifndef MINMAXINDEX_H
#define MINMAXINDEX_H

#include "Indicator.h"

namespace tacuda {
class MINMAXINDEX : public Indicator {
public:
  explicit MINMAXINDEX(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

} // namespace tacuda

#endif
