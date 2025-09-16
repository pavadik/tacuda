#ifndef MINMAX_H
#define MINMAX_H

#include "Indicator.h"

namespace tacuda {
class MINMAX : public Indicator {
public:
  explicit MINMAX(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

} // namespace tacuda

#endif
