#ifndef INNECK_H
#define INNECK_H

#include "Indicator.h"

namespace tacuda {
class InNeck : public Indicator {
public:
  InNeck() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
