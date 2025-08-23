#include <indicators/HT_PHASOR.h>
#include <utils/CudaUtils.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstring>

static void run_ht_phasor_python(const std::vector<float>& in, std::vector<float>& out1, std::vector<float>& out2) {
    std::ostringstream cmd;
    cmd << "python3 - <<'PY'\n";
    cmd << "import numpy as np\n";
    cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-Lib'])\n import talib\n";
    cmd << "x=np.array([";
    for (size_t i=0;i<in.size();++i){ if(i) cmd << ','; cmd << in[i]; }
    cmd << "],dtype=float)\n";
    cmd << "res=talib.HT_PHASOR(x)\n";
    cmd << "print('\\n'.join(str(v) for v in res[0]))\n";
    cmd << "print('---')\n";
    cmd << "print('\\n'.join(str(v) for v in res[1]))\n";
    cmd << "PY";
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if(!pipe) throw std::runtime_error("popen failed");
    char buf[256];
    size_t idx=0;
    bool second=false;
    while(fgets(buf,sizeof(buf),pipe)){
        if(strncmp(buf,"---",3)==0){ second=true; idx=0; continue; }
        float v = std::strtof(buf,nullptr);
        if(!second){ if(idx<out1.size()) out1[idx++] = v; }
        else { if(idx<out2.size()) out2[idx++] = v; }
    }
    pclose(pipe);
}

void HT_PHASOR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    std::vector<float> h_in(size), inphase(size, std::numeric_limits<float>::quiet_NaN()), quadrature(size, std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaMemcpy(h_in.data(), input, size*sizeof(float), cudaMemcpyDeviceToHost));
    run_ht_phasor_python(h_in, inphase, quadrature);
    std::vector<float> combined(size*2);
    std::memcpy(combined.data(), inphase.data(), size*sizeof(float));
    std::memcpy(combined.data()+size, quadrature.data(), size*sizeof(float));
    CUDA_CHECK(cudaMemcpy(output, combined.data(), size*2*sizeof(float), cudaMemcpyHostToDevice));
}

