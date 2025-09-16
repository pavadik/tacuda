#ifndef MIN_H
#define MIN_H

#include "Indicator.h"

namespace tacuda {
class MIN : public Indicator {
public:
    explicit MIN(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
