#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "../include/indicators/SMA.h"
#include "../include/indicators/Momentum.h"
#include "../include/indicators/MACD.h"
#include "../include/utils/CudaUtils.h"

int main() {
    const int N = 1024;
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = std::sin(0.01f * i) + 0.5f * std::cos(0.03f * i);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    {
        SMA sma(14);
        sma.calculate(d_in, d_out, N);
        std::vector<float> h_out(N, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "SMA[0..4]: ";
        for (int i = 0; i < 5; ++i) std::cout << h_out[i] << (i<4? ", ":"\n");
    }

    {
        Momentum mom(10);
        mom.calculate(d_in, d_out, N);
        std::vector<float> h_out(N, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Momentum[0..4]: ";
        for (int i = 0; i < 5; ++i) std::cout << h_out[i] << (i<4? ", ":"\n");
    }

    {
        MACD macd(12, 26);
        macd.calculate(d_in, d_out, N);
        std::vector<float> h_out(N, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "MACD[0..4]: ";
        for (int i = 0; i < 5; ++i) std::cout << h_out[i] << (i<4? ", ":"\n");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
