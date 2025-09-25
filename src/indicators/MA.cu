#include <indicators/MA.h>
#include <indicators/SMA.h>
#include <indicators/EMA.h>
#include <indicators/WMA.h>
#include <indicators/DEMA.h>
#include <indicators/TEMA.h>
#include <indicators/TRIMA.h>
#include <indicators/KAMA.h>
#include <indicators/MAMA.h>
#include <indicators/T3.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

tacuda::MA::MA(int period, tacuda::MAType type) : period(period), type(type) {}

void tacuda::MA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MA: invalid period");
    }
    switch (type) {
    case tacuda::MAType::SMA: {
        tacuda::SMA sma(period);
        sma.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::EMA: {
        tacuda::EMA ema(period);
        ema.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::WMA: {
        tacuda::WMA wma(period);
        wma.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::DEMA: {
        tacuda::DEMA dema(period);
        dema.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::TEMA: {
        tacuda::TEMA tema(period);
        tema.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::TRIMA: {
        tacuda::TRIMA trima(period);
        trima.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::KAMA: {
        tacuda::KAMA kama(period, 2, 30);
        kama.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::MAMA: {
        tacuda::MAMA mama(0.5f, 0.05f);
        auto tmp = acquireDeviceBuffer<float>(2 * size);
        mama.calculate(input, tmp.get(), size, stream);
        CUDA_CHECK(cudaMemcpyAsync(output, tmp.get(), size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        break;
    }
    case tacuda::MAType::T3: {
        tacuda::T3 t3(period, 0.7f);
        t3.calculate(input, output, size, stream);
        break;
    }
    default:
        throw std::invalid_argument("MA: unsupported type");
    }
}
