#include "brp/science.hpp"
#include "demod_binary_resamp_cpu.h"
#include "demod_binary_fft_fftw.h"
#include "hs_common.h"
#include "erp_utilities.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <stdexcept>
extern "C" int boinc_resolve_filename(const char*,char*,int) { return -1; }
namespace brp {
std::vector<float> historical_resample(std::span<const float> input,const Params& p) {
    RESAMP_PARAMS rp{p.n,p.unpadded,p.n/2+1,p.tau,p.omega,p.psi,p.dt,p.step_inv,p.s0};
    DIfloatPtr in{},out{};in.host_ptr=const_cast<float*>(input.data());
    if(set_up_resampling(in,&out,&rp,nullptr,nullptr)) throw std::runtime_error("historical resampler allocation failed");
    int status=run_resampling(in,out,&rp);
    std::vector<float> values(out.host_ptr,out.host_ptr+p.n);tear_down_resampling(out);
    if(status) throw std::runtime_error("historical resampler failed");
    return values;
}
std::vector<float> historical_power(std::span<const float> resampled,const Params& p) {
    std::vector<float> values(resampled.begin(),resampled.end());
    DIfloatPtr in{},out{};in.host_ptr=values.data();
    if(set_up_fft(in,&out,p.n,p.n/2+1)) throw std::runtime_error("historical FFT setup failed");
    int status=run_fft(in,out,p.n,p.n/2+1,1.0f/p.n);
    std::vector<float> power(out.host_ptr,out.host_ptr+p.n/2);tear_down_fft(out);
    if(status) throw std::runtime_error("historical FFT failed");
    return power;
}
CPUStats benchmark_cpu(std::span<const float> input,const Params& p,uint32_t reps) {
    RESAMP_PARAMS rp{p.n,p.unpadded,p.n/2+1,p.tau,p.omega,p.psi,p.dt,p.step_inv,p.s0};
    DIfloatPtr in{},out{},power{};in.host_ptr=const_cast<float*>(input.data());
    if(set_up_resampling(in,&out,&rp,nullptr,nullptr)) throw std::runtime_error("CPU benchmark resampling setup failed");
    if(set_up_fft(out,&power,p.n,p.n/2+1)) {tear_down_resampling(out);throw std::runtime_error("CPU benchmark FFT setup failed");}
    std::vector<float> summed(5*p.fundamental);
    uint32_t pages=(p.fundamental>>10)+1;
    std::array<std::vector<int32_t>,5> flags;
    float* sums[5];int32_t* dirty[5];float thresholds[5];
    std::copy_n(p.threshold,5,thresholds);
    for(uint32_t h=0;h<5;++h) {sums[h]=h?summed.data()+h*p.fundamental:power.host_ptr;flags[h].resize(pages);dirty[h]=flags[h].data();}
    std::vector<double> walls,cpus;
    auto cpu_now=[](){timespec t{};clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;};
    for(uint32_t r=0;r<=reps;++r) {
        auto start=std::chrono::steady_clock::now();double cpu=cpu_now();
        run_resampling(in,out,&rp);run_fft(out,power,p.n,p.n/2+1,1.0f/p.n);
        std::fill(summed.begin(),summed.end(),0);
        for(auto& f:flags) std::fill(f.begin(),f.end(),0);
        harmonic_summing(sums,dirty,power.host_ptr,p.window,p.fundamental,p.harmonic,thresholds);
        std::copy_n(power.host_ptr,p.fundamental,summed.data());
        auto candidates=select(summed,p);
        if(r) {cpus.push_back(cpu_now()-cpu);walls.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
    }
    tear_down_fft(power);tear_down_resampling(out);
    std::sort(walls.begin(),walls.end());std::sort(cpus.begin(),cpus.end());
    return {walls[walls.size()/2],cpus[cpus.size()/2]};
}
}
