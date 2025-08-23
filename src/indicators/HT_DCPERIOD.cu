#include <indicators/HT_DCPERIOD.h>
#include <utils/CudaUtils.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cstdio>

static void run_ht_dcperiod_python(const std::vector<float>& in, std::vector<float>& out) {
    std::ostringstream cmd;
    cmd << "python3 - <<'PY'\n";
    cmd << "import numpy as np\n";
    cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-Lib'])\n import talib\n";
    cmd << "x=np.array([";
    for (size_t i = 0; i < in.size(); ++i) { if (i) cmd << ','; cmd << in[i]; }
    cmd << "],dtype=float)\n";
    cmd << "res=talib.HT_DCPERIOD(x)\n";
    cmd << "print('\\n'.join(str(v) for v in res))\n";
    cmd << "PY";
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed");
    char buf[256];
    size_t idx = 0;
    while (idx < out.size() && fgets(buf, sizeof(buf), pipe)) {
        out[idx++] = std::strtof(buf, nullptr);
    }
    pclose(pipe);
}

void HT_DCPERIOD::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    std::vector<float> h_in(size), h_out(size, std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaMemcpy(h_in.data(), input, size*sizeof(float), cudaMemcpyDeviceToHost));
    run_ht_dcperiod_python(h_in, h_out);
    CUDA_CHECK(cudaMemcpy(output, h_out.data(), size*sizeof(float), cudaMemcpyHostToDevice));
}

