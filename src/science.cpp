// GPL-2.0-or-later. Resampling follows demod_binary_resamp_cpu.c and erp_utilities.cpp.
#include "brp/science.hpp"
#include "brp/lut.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include "hs_common.h"
// FFTW's stable public C ABI, limited to these functions. Runtime-only installations
// are supported without installing development headers or altering the system.
extern "C" {
typedef struct fftwf_plan_s *fftwf_plan;
fftwf_plan fftwf_plan_dft_r2c_1d(int, float*, float (*)[2], unsigned);
void fftwf_execute(const fftwf_plan);
void fftwf_destroy_plan(fftwf_plan);
}
namespace brp {
float lut_sin(float x) {
    float dummy;
    float xt=std::modf((1.0f/6.283185f)*x, &dummy);
    if(xt<0) xt+=1;
    int i=int(xt*64.0f+0.5f);
    float d=6.283185f*(xt-(1.0f/64)*i), d2=d;
    d2*=0.5f*d;
    return sinSamples[i]+d*cosSamples[i]-d2*sinSamples[i];
}
static float fused_sin(float phase) {
    float whole;float xt=std::modf((1.0f/6.283185f)*phase,&whole);if(xt<0) xt+=1;
    int i=int(std::fma(xt,64.0f,0.5f));
    float d=6.283185f*std::fma(-float(i),1.0f/64,xt);
    float d2=(0.5f*d)*d;
    return std::fma(-d2,sinSamples[i],std::fma(d,cosSamples[i],sinSamples[i]));
}
float sample_offset(uint32_t i, const Params& p) {
    float t=float(i)*p.dt;
    if(p.flags&2) return std::fma(p.tau*fused_sin(std::fma(p.omega,t,p.psi)),p.step_inv,-p.s0);
    return p.tau*lut_sin(p.omega*t+p.psi)*p.step_inv-p.s0;
}
void resampling_steps(Params& p) {
    p.steps=p.unpadded-1;
    while(p.steps && float(p.steps)-sample_offset(p.steps,p)>=float(p.unpadded-1)) --p.steps;
    if(!p.steps) throw std::runtime_error("template produces empty series");
}
Params parameters(uint32_t n,uint32_t unpadded,float tau,float period,float psi,float dt) {
    uint32_t radix=std::has_single_bit(n)?1:((n%3==0&&std::has_single_bit(n/3))?3:0);
    if(!radix||n<64||n>3*(1u<<22)||unpadded<32||unpadded>n||dt<=0||period<=0||tau<0 ||
       !std::isfinite(tau+period+psi+dt)) throw std::runtime_error("invalid observation/template");
    Params p{};
    p.n=n; p.unpadded=unpadded; p.fft_n=n; p.fft_radix=radix; p.fft_base=n/radix;
    p.logn=std::countr_zero(p.fft_base); p.tau=tau;
    p.omega=2.0*std::numbers::pi/period; p.psi=psi; p.dt=dt; p.step_inv=1.0f/dt;
    // Historical main computes S0 with double sin, unlike the LUT in the resampler.
    p.s0=tau*std::sin(double(psi))*p.step_inv;
    if(!std::isfinite(p.omega)||!std::isfinite(p.step_inv)||!std::isfinite(p.s0)||
       !std::isfinite(p.tau*p.step_inv)||!std::isfinite(float(unpadded)*p.dt)||
       !std::isfinite(p.omega*(float(unpadded)*p.dt)+p.psi))
        throw std::runtime_error("template exceeds finite FP32 arithmetic domain");
    resampling_steps(p);
    p.harmonic=n/2; p.fundamental=n/2; p.window=8;
    const float thresholds[]={20,30,45,70,110};
    std::copy_n(thresholds,5,p.threshold); p.capacity=65536;
    return p;
}
std::vector<float> synthetic(uint32_t n,float dt,uint32_t seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> noise(0,1);
    std::vector<float> x(n);
    for(uint32_t i=0;i<n;++i) {
        double phase=2*std::numbers::pi*37.0*i*dt;
        x[i]=noise(rng)+float(0.5*std::cos(phase)+0.25*std::cos(2*phase)+0.15*std::cos(3*phase));
    }
    return x;
}
std::vector<float> resample(std::span<const float> input,const Params& p) {
    if(input.size()!=p.unpadded) throw std::runtime_error("input length mismatch");
    std::vector<float> out(p.n); float sum=0;
    for(uint32_t i=0;i<p.steps;++i) {
        // The +0.5 is a double literal in the supplied CPU source.
        int j=int(double(float(i)-sample_offset(i,p))+0.5);
        if(j<0 || uint32_t(j)>=p.unpadded) throw std::runtime_error("resample index outside observation");
        sum+=out[i]=input[j];
    }
    std::fill(out.begin()+p.steps,out.end(),sum/float(p.steps)); return out;
}
std::vector<std::complex<float>> fft_reference(std::span<const float> input) {
    std::vector<float> copy(input.begin(),input.end());
    std::vector<std::complex<float>> out(input.size()/2+1);
    auto plan=fftwf_plan_dft_r2c_1d(int(input.size()),copy.data(),reinterpret_cast<float (*)[2]>(out.data()),64u);
    if(!plan) throw std::runtime_error("FFTW plan failed");
    fftwf_execute(plan); fftwf_destroy_plan(plan); return out;
}
std::vector<float> harmonic_reference(std::span<const float> power,const Params& p) {
    if(power.size()<p.harmonic) throw std::runtime_error("power length mismatch");
    std::vector<float> out(5*p.fundamental,0);
    uint32_t pages=(p.fundamental>>10)+1;
    std::array<std::vector<int32_t>,5> flags;
    float* sums[5]; int32_t* dirty[5];
    // hs_common only reads sumspec[0]; provide a real independent power buffer.
    std::vector<float> powers(power.begin(),power.end()); sums[0]=powers.data();
    for(int h=0;h<5;++h) { flags[h].resize(pages); dirty[h]=flags[h].data(); if(h) sums[h]=out.data()+h*p.fundamental; }
    auto thresholds=std::array<float,5>{}; std::copy_n(p.threshold,5,thresholds.data());
    harmonic_summing(sums,dirty,powers.data(),p.window,p.fundamental,p.harmonic,thresholds.data());
    std::copy_n(power.begin(),p.fundamental,out.begin()); return out;
}
std::vector<Candidate> select(std::span<const float> summed,const Params& p) {
    std::vector<Candidate> out;
    for(uint32_t h=0;h<5;++h) for(uint32_t j=p.window;j<p.fundamental;++j) {
        float x=summed[h*p.fundamental+j]; if(x>p.threshold[h]) out.push_back({j,h,x,0});
    }
    std::sort(out.begin(),out.end(),[](auto a,auto b){return a.level!=b.level?a.level<b.level:a.bin<b.bin;}); return out;
}
Reference reference(std::span<const float> input,const Params& p) {
    Reference r; r.resampled=resample(input,p); r.fft=fft_reference(r.resampled); r.power.resize(p.n/2);
    for(uint32_t i=1;i<p.n/2;++i) {
        double re=r.fft[i].real(),im=r.fft[i].imag();
        r.power[i]=float(double(1.0f/p.n)*(re*re+im*im));
    }
    r.summed=harmonic_reference(r.power,p); r.candidates=select(r.summed,p); return r;
}
double relative_l2(std::span<const float> a,std::span<const float> b) {
    if(a.size()!=b.size()) throw std::runtime_error("comparison length mismatch");
    double err=0,den=0;
    for(size_t i=0;i<a.size();++i) { if(!std::isfinite(a[i])||!std::isfinite(b[i])) return INFINITY;
        double d=double(a[i])-b[i];err+=d*d;den+=double(b[i])*b[i]; }
    return std::sqrt(err/std::max(den,1e-30));
}
}
