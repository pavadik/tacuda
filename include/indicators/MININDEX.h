#ifndef MININDEX_H
#define MININDEX_H

#include "Indicator.h"

namespace tacuda {
class MININDEX : public Indicator {
public:
  explicit MININDEX(int period);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
};

} // namespace tacuda

#endif
