#ifndef TRIMA_H
#define TRIMA_H

#include "Indicator.h"

namespace tacuda {
class TRIMA : public Indicator {
public:
  explicit TRIMA(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

} // namespace tacuda

#endif
