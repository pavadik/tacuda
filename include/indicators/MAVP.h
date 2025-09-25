#ifndef MAVP_H
#define MAVP_H

#include "Indicator.h"
#include "MA.h"

namespace tacuda {
class MAVP : public Indicator {
public:
    MAVP(int minPeriod, int maxPeriod, MAType type);
    void calculate(const float* values, const float* periods, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int minPeriod;
    int maxPeriod;
    MAType type;
};

} // namespace tacuda

#endif
