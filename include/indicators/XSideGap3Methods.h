#ifndef XSIDEGAP3METHODS_H
#define XSIDEGAP3METHODS_H

#include "Indicator.h"

namespace tacuda {
class XSideGap3Methods : public Indicator {
public:
  XSideGap3Methods() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size, cudaStream_t stream = 0) noexcept(false);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif

