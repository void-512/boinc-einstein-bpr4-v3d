#include "brp/vulkan.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
static double median(std::vector<double> x) {std::sort(x.begin(),x.end());return x[x.size()/2];}
static void require(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
int main(int argc,char** argv) try {
    uint32_t n=65536,unpadded=0,wg=192,tile=512,batch=4,reps=9,seed=42;
    float tau=0.002,period=120,psi=0.3,dt=0.000256,threshold_scale=1;
    bool software=false,stockham=false,radix4=false;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {std::cout<<"brp_bench [--n power-of-two] [--unpadded N] [--wg 16..256] [--tile 0|64..4096] [--stockham --tile 0] [--radix4] [--batch 1..64] [--reps N] [--seed N] [--tau seconds] [--period seconds] [--psi radians] [--dt seconds] [--threshold-scale factor] [--allow-software]\n";return 0;}
        if(key=="--allow-software") {software=true;continue;}
        if(key=="--stockham") {stockham=true;continue;}
        if(key=="--radix4") {radix4=true;continue;}
        require(i+1<argc,"option needs value");std::string v=argv[++i];
        if(key=="--n") n=std::stoul(v);else if(key=="--unpadded") unpadded=std::stoul(v);
        else if(key=="--wg") wg=std::stoul(v);else if(key=="--tile") tile=std::stoul(v);
        else if(key=="--batch") batch=std::stoul(v);else if(key=="--reps") reps=std::stoul(v);
        else if(key=="--seed") seed=std::stoul(v);else if(key=="--tau") tau=std::stof(v);
        else if(key=="--period") period=std::stof(v);else if(key=="--psi") psi=std::stof(v);
        else if(key=="--dt") dt=std::stof(v);else if(key=="--threshold-scale") threshold_scale=std::stof(v);
        else throw std::runtime_error("unknown option: "+key);
    }
    require(reps>0&&reps<=10000 && std::isfinite(threshold_scale)&&threshold_scale>=0,"invalid repetitions/thresholds");
    if(!unpadded) unpadded=n-17;
    auto p=brp::parameters(n,unpadded,tau,period,psi,dt);
    for(auto& t:p.threshold) t*=threshold_scale;
    auto input=brp::synthetic(unpadded,dt,seed);
    auto ref_start=std::chrono::steady_clock::now();auto ref=brp::reference(input,p);
    double reference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ref_start).count();
    require(ref.resampled==brp::historical_resample(input,p),"reference resampling differs from historical source");
    require(ref.power==brp::historical_power(ref.resampled,p),"power oracle differs from historical source");
    auto cpu_baseline=brp::benchmark_cpu(input,p,5);
    auto init_start=std::chrono::steady_clock::now();
    brp::GPU gpu(p,input,wg,tile,batch,software,stockham,radix4);
    double init_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-init_start).count();
    gpu.run();gpu.run(); // untimed warmup; persistent command buffers and pipelines
    auto actual=gpu.snapshot();
    uint32_t index_mismatches=0;
    for(uint32_t i=0;i<p.steps;++i) if(actual.resampled[i]!=ref.resampled[i]) ++index_mismatches;
    double resamp_error=brp::relative_l2(actual.resampled,ref.resampled);
    auto real_span=[](const auto& x){return std::span<const float>(reinterpret_cast<const float*>(x.data()),x.size()*2);};
    double fft_error=brp::relative_l2(real_span(actual.fft),real_span(ref.fft));
    double power_error=brp::relative_l2(actual.power,ref.power),hs_error=brp::relative_l2(actual.summed,ref.summed);
    // Isolate harmonic algorithm errors from upstream FFT error by using GPU power
    // as input to the unchanged historical CPU harmonic routine.
    auto independent=brp::harmonic_reference(actual.power,p);
    double hs_isolated_error=brp::relative_l2(actual.summed,independent);
    uint32_t candidate_mismatches=0;
    if(actual.candidates.size()!=ref.candidates.size()) ++candidate_mismatches;
    else for(size_t i=0;i<ref.candidates.size();++i) {
        auto a=actual.candidates[i],b=ref.candidates[i];
        if(a.bin!=b.bin||a.level!=b.level||std::abs(a.power-b.power)>2e-4f*(1+std::abs(b.power))) ++candidate_mismatches;
    }
    uint32_t rank_mismatches=0;
    for(uint32_t h=0;h<5;++h) {
        std::vector<brp::Candidate> a,b;
        for(auto c:actual.candidates) if(c.level==h) a.push_back(c);
        for(auto c:ref.candidates) if(c.level==h) b.push_back(c);
        auto order=[](auto x,auto y){return x.power!=y.power?x.power>y.power:x.bin<y.bin;};
        std::sort(a.begin(),a.end(),order);std::sort(b.begin(),b.end(),order);
        for(size_t i=0;i<std::min<size_t>(100,std::min(a.size(),b.size()));++i) if(a[i].bin!=b[i].bin) ++rank_mismatches;
    }
    bool valid=!index_mismatches&&!candidate_mismatches&&!rank_mismatches && resamp_error<5e-5&&fft_error<3e-5&&power_error<5e-5&&hs_error<5e-5&&hs_isolated_error<1e-7;
    std::vector<double> wall,cpu,gpu_ms,submit_ms;std::array<std::vector<double>,4> stages;
    double total_wall=0,total_cpu=0;
    for(uint32_t r=0;r<reps;++r) {
        auto t=gpu.run();wall.push_back(t.wall_ms/batch);cpu.push_back(t.cpu_ms/batch);total_wall+=t.wall_ms;total_cpu+=t.cpu_ms;
        submit_ms.push_back(t.submit_ms/batch);
        double sum=0;for(uint32_t s=0;s<4;++s) {stages[s].push_back(t.stage_ms[s]/batch);sum+=t.stage_ms[s];}gpu_ms.push_back(sum/batch);
    }
    auto final=gpu.snapshot();
    bool reproducible=actual.resampled==final.resampled&&actual.fft==final.fft&&actual.power==final.power&&actual.summed==final.summed && actual.candidates.size()==final.candidates.size();
    if(reproducible) for(size_t i=0;i<actual.candidates.size();++i) {
        auto a=actual.candidates[i],b=final.candidates[i];
        if(a.bin!=b.bin||a.level!=b.level||a.power!=b.power) reproducible=false;
    }
    valid=valid&&reproducible;
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::cout<<std::setprecision(10)<<"{\n  \"device\": \""<<gpu.device_info()<<"\",\n"
        <<"  \"n\": "<<n<<", \"unpadded\": "<<unpadded<<", \"steps\": "<<p.steps<<", \"seed\": "<<seed<<",\n"
        <<"  \"tau\": "<<tau<<", \"period\": "<<period<<", \"psi\": "<<psi<<", \"dt\": "<<dt<<", \"threshold_scale\": "<<threshold_scale<<",\n"
        <<"  \"wg\": "<<wg<<", \"tile\": "<<tile<<", \"batch\": "<<batch<<", \"reps\": "<<reps<<",\n"
        <<"  \"stockham\": "<<(stockham?"true":"false")<<",\n"
        <<"  \"twiddle_layout\": \""<<(BRP_STRIDED_TWIDDLES?"strided":"dense")<<"\",\n"
        <<"  \"radix4\": "<<(radix4?"true":"false")<<", \"submission_ms_median\": "<<median(submit_ms)<<",\n"
        <<"  \"valid\": "<<(valid?"true":"false")<<", \"index_mismatches\": "<<index_mismatches<<", \"candidate_mismatches\": "<<candidate_mismatches<<", \"rank_mismatches\": "<<rank_mismatches<<",\n"
        <<"  \"replay_reproducible\": "<<(reproducible?"true":"false")<<",\n"
        <<"  \"resample_l2\": "<<resamp_error<<", \"fft_l2\": "<<fft_error<<", \"power_l2\": "<<power_error<<", \"harmonic_l2\": "<<hs_error<<", \"harmonic_isolated_l2\": "<<hs_isolated_error<<",\n"
        <<"  \"candidates\": "<<actual.candidates.size()<<", \"initialization_ms\": "<<init_ms<<", \"cpu_reference_with_planning_ms\": "<<reference_ms<<",\n"
        <<"  \"cpu_core_wall_ms_median\": "<<cpu_baseline.wall_ms<<", \"cpu_core_process_ms_median\": "<<cpu_baseline.cpu_ms<<",\n"
        <<"  \"wall_ms_median\": "<<median(wall)<<", \"wall_ms_min\": "<<*std::min_element(wall.begin(),wall.end())<<", \"wall_ms_max\": "<<*std::max_element(wall.begin(),wall.end())<<",\n"
        <<"  \"gpu_ms_median\": "<<median(gpu_ms)<<", \"process_cpu_ms_median\": "<<median(cpu)<<", \"cpu_percent\": "<<100*total_cpu/total_wall<<",\n"
        <<"  \"replays_per_second\": "<<1000/median(wall)<<", \"allocated_gpu_bytes\": "<<gpu.allocated_bytes()<<", \"peak_rss_kib\": "<<usage.ru_maxrss<<",\n"
        <<"  \"stage_ms_median\": {\"resample\": "<<median(stages[0])<<", \"fft\": "<<median(stages[1])<<", \"power\": "<<median(stages[2])<<", \"harmonic_candidate\": "<<median(stages[3])<<"},\n  \"wall_ms_samples\": [";
    for(size_t i=0;i<wall.size();++i) std::cout<<(i?", ":"")<<wall[i];
    std::cout<<"]\n}\n";
    return valid?0:2;
} catch(const std::exception& e) {std::cerr<<"brp_bench: "<<e.what()<<'\n';return 1;}
