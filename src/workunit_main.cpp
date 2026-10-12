#include "brp/workload.hpp"
#include "brp/vulkan.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
namespace fs=std::filesystem;
static std::vector<float> read_floats(const fs::path& path,size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file||size_t(file.tellg())!=count*4) throw std::runtime_error("reference float dimensions: "+path.string());
    std::vector<float> values(count);file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),count*4);
    if(!file) throw std::runtime_error("reference read failed");
    return values;
}
static void dump(const fs::path& path,std::span<const float> values) {
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(values.data()),values.size_bytes());
    if(!file) throw std::runtime_error("state dump failed");
}
int main(int argc,char** argv) try {
    std::map<std::string,std::string> options;bool cpu=false,packed=false,validate=false,scatter=false,radix8=false,stockham=true,fused_power=false,radix8_store=false;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--cpu") {cpu=true;continue;}if(key=="--packed") {packed=true;continue;}if(key=="--validate") {validate=true;continue;}
        if(key=="--scatter") {scatter=true;continue;}
        if(key=="--radix8") {radix8=true;continue;}
        if(key=="--dit") {stockham=false;continue;}
        if(key=="--fused-power") {fused_power=true;continue;}
        if(key=="--radix8-store") {radix8_store=true;radix8=true;continue;}
        if(i+1==argc) throw std::runtime_error("option needs a value");
        if(!options.emplace(key,argv[++i]).second) throw std::runtime_error("duplicate option");
    }
    for(auto key:{"--input","--bank","--zap","--padding","--f0","--false-alarm","--window","--output","--metrics"})
        if(!options.contains(key)) throw std::runtime_error(std::string("required option ")+key);
    for(auto& [key,value]:options) if(key!="--input"&&key!="--bank"&&key!="--zap"&&key!="--padding"&&key!="--f0"&&key!="--false-alarm"&&key!="--window"&&key!="--output"&&key!="--metrics"&&key!="--wg"&&key!="--reference-dir"&&key!="--dump-prefix"&&key!="--preprocessed"&&key!="--batch"&&key!="--counter-output"&&key!="--tile")
        throw std::runtime_error("unknown option "+key);
    if(fs::exists(options.at("--output"))||fs::exists(options.at("--metrics"))) throw std::runtime_error("refusing to overwrite results");
    if(options.contains("--counter-output")&&(cpu||fs::exists(options.at("--counter-output")))) throw std::runtime_error("counter output requires a GPU and a new file");
    auto start=std::chrono::steady_clock::now();
    auto o=brp::load_observation(options.at("--input"),std::stod(options.at("--padding")));
    uint32_t window=std::stoul(options.at("--window")),wg=options.contains("--wg")?std::stoul(options.at("--wg")):256;
    uint32_t batch=options.contains("--batch")?std::stoul(options.at("--batch")):1;
    uint32_t tile=options.contains("--tile")?std::stoul(options.at("--tile")):0;
    if((radix8&&!stockham)||(tile&&stockham)) throw std::runtime_error("radix 8 requires Stockham; shared FFT tiles require --dit");
    if(!batch||batch>64||(batch>1&&(cpu||validate||options.contains("--reference-dir"))))
        throw std::runtime_error("batch must be 1..64; intermediate validation requires batch 1");
    if(options.contains("--preprocessed")) o.preprocessed=read_floats(options.at("--preprocessed"),o.samples);
    else brp::preprocess(o,options.at("--zap"),window);
    auto bank=brp::load_bank(options.at("--bank"));
    double prep_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    double decoded_error=-1,preprocessed_error=-1;
    if(options.contains("--reference-dir")) {
        auto folder=fs::path(options.at("--reference-dir"));
        decoded_error=brp::relative_l2(o.decoded,read_floats(folder/"decoded.f32",o.samples));
        preprocessed_error=brp::relative_l2(o.preprocessed,read_floats(folder/"preprocessed.f32",o.samples));
        if(decoded_error!=0||preprocessed_error>=1e-4) throw std::runtime_error("authentic decoding/preprocessing validation failed");
    }
    if(options.contains("--dump-prefix")) {
        dump(options.at("--dump-prefix")+"-decoded.f32",o.decoded);
        dump(options.at("--dump-prefix")+"-preprocessed.f32",o.preprocessed);
    }
    auto first=brp::search_parameters(o,bank[0],std::stod(options.at("--f0")),std::stod(options.at("--false-alarm")),window);
    std::unique_ptr<brp::GPU> gpu;
    if(!cpu) {gpu=std::make_unique<brp::GPU>(first,o.preprocessed,wg,tile,batch,false,stockham,true,packed,scatter,radix8,fused_power,radix8_store);gpu->set_template(first);gpu->run();}
    std::ofstream metrics(options.at("--metrics"));metrics<<std::setprecision(12);
    brp::Toplist top;bool all_valid=true;
    auto bank_start=std::chrono::steady_clock::now();
    brp::Timings group_timing{};size_t group_size=1;
    brp::Params last_parameters=first;
    double total_gpu_ms=0;
    for(size_t index=0;index<bank.size();++index) {
        auto p=brp::search_parameters(o,bank[index],std::stod(options.at("--f0")),std::stod(options.at("--false-alarm")),window);top.thresholds(p);
        last_parameters=p;
        brp::Snapshot actual;brp::Timings timing{};
        brp::Reference reference;
        if(cpu||validate) reference=brp::reference(o.preprocessed,p);
        if(cpu) {actual.resampled=reference.resampled;actual.fft=reference.fft;actual.power=reference.power;actual.summed=reference.summed;actual.candidates=reference.candidates;}
        else {
            if(index%batch==0) {
                std::vector<brp::Params> group;
                group_size=std::min<size_t>(batch,bank.size()-index);
                for(size_t offset=0;offset<group_size;++offset) {
                    auto next=brp::search_parameters(o,bank[index+offset],std::stod(options.at("--f0")),std::stod(options.at("--false-alarm")),window);
                    top.thresholds(next);group.push_back(next);
                }
                gpu->set_templates(group);group_timing=gpu->run();
            }
            timing=group_timing;timing.stage_ms=group_timing.template_stage_ms[index%batch];
            timing.wall_ms/=group_size;timing.cpu_ms/=group_size;
            for(auto duration:timing.stage_ms) total_gpu_ms+=duration;
            bool authentic=options.contains("--reference-dir")&&fs::exists(fs::path(options.at("--reference-dir"))/("resampled-"+std::to_string(index)+".f32"));
            if(validate||authentic) actual=gpu->snapshot();
            else actual.candidates=gpu->candidates(index%batch);
        }
        double resample_error=-1,fft_error=-1,power_error=-1,hs_error=-1;
        uint32_t sampled_mismatches=0,candidate_mismatches=0;
        if(validate&&!cpu) {
            for(uint32_t i=0;i<p.steps;++i) if(actual.resampled[i]!=reference.resampled[i]) ++sampled_mismatches;
            resample_error=brp::relative_l2(actual.resampled,reference.resampled);
            auto floats=[](const auto& a){return std::span<const float>(reinterpret_cast<const float*>(a.data()),a.size()*2);};
            fft_error=brp::relative_l2(floats(actual.fft),floats(reference.fft));power_error=brp::relative_l2(actual.power,reference.power);
            hs_error=brp::relative_l2(actual.summed,brp::harmonic_reference(actual.power,p));
            if(actual.candidates.size()!=reference.candidates.size()) ++candidate_mismatches;
            else for(size_t i=0;i<actual.candidates.size();++i) {
                auto a=actual.candidates[i],b=reference.candidates[i];
                if(a.bin!=b.bin||a.level!=b.level||std::abs(a.power-b.power)>2e-4*(1+std::abs(b.power))) ++candidate_mismatches;
            }
        }
        double authentic_resample=-1,authentic_power=-1;
        if(options.contains("--reference-dir")&&fs::exists(fs::path(options.at("--reference-dir"))/("resampled-"+std::to_string(index)+".f32"))) {
            auto folder=fs::path(options.at("--reference-dir"));
            authentic_resample=brp::relative_l2(actual.resampled,read_floats(folder/("resampled-"+std::to_string(index)+".f32"),p.n));
            auto power=read_floats(folder/("power-"+std::to_string(index)+".f32"),p.n/2+1);power.pop_back();
            authentic_power=brp::relative_l2(actual.power,power);
        }
        bool valid=!sampled_mismatches&&!candidate_mismatches&&resample_error<5e-5&&fft_error<3e-5&&power_error<5e-5&&hs_error<1e-7&&authentic_resample<1e-4&&authentic_power<1e-4;
        all_valid=all_valid&&valid;
        top.retain(actual.candidates,bank[index],p);
        metrics<<"{\"template\":"<<index<<",\"n\":"<<p.n<<",\"steps\":"<<p.steps<<",\"wg\":"<<wg<<",\"batch\":"<<batch<<",\"packed\":"<<(packed?"true":"false")<<",\"cpu\":"<<(cpu?"true":"false")
               <<",\"scatter\":"<<(scatter?"true":"false")<<",\"radix8\":"<<(radix8?"true":"false")<<",\"stockham\":"<<(stockham?"true":"false")<<",\"tile\":"<<tile<<",\"fused_power\":"<<(fused_power?"true":"false")<<",\"radix8_store\":"<<(radix8_store?"true":"false")<<",\"period\":"<<bank[index].period<<",\"tau\":"<<p.tau<<",\"psi\":"<<p.psi<<",\"s0\":"<<p.s0
               <<",\"validated\":"<<(validate?"true":"false")<<",\"valid\":"<<(valid?"true":"false")<<",\"wall_ms\":"<<timing.wall_ms<<",\"process_cpu_ms\":"<<timing.cpu_ms<<",\"gpu_stage_ms\":[";
        for(size_t i=0;i<4;++i) metrics<<(i?",":"")<<timing.stage_ms[i];
        metrics<<"],\"resample_l2\":"<<resample_error<<",\"fft_l2\":"<<fft_error<<",\"power_l2\":"<<power_error<<",\"hs_isolated_l2\":"<<hs_error<<",\"sampled_mismatches\":"<<sampled_mismatches
               <<",\"candidate_mismatches\":"<<candidate_mismatches<<",\"authentic_resample_l2\":"<<authentic_resample<<",\"authentic_power_l2\":"<<authentic_power
               <<",\"candidates\":"<<actual.candidates.size()<<",\"thresholds\":[";
        for(size_t h=0;h<5;++h) metrics<<(h?",":"")<<p.threshold[h];
        metrics<<"]}\n";metrics.flush();
        if(index%100==0) std::cerr<<"Processed "<<index+1<<"/"<<bank.size()<<" orbital templates\n";
        if(!valid) throw std::runtime_error("real template correctness failure at "+std::to_string(index));
    }
    auto result=top.final();brp::write_candidates(options.at("--output"),result,o.n*o.sample_us*1e-6);
    double bank_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-bank_start).count();
    if(options.contains("--counter-output")) {
        gpu->set_template(last_parameters);
        std::ofstream counters(options.at("--counter-output"));counters<<gpu->profile_counters()<<'\n';
        if(!counters) throw std::runtime_error("counter output write failed");
    }
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::cout<<std::setprecision(12)<<"{\"templates\":"<<bank.size()<<",\"candidates\":"<<result.size()<<",\"n\":"<<o.n<<",\"preprocessing_ms\":"<<prep_ms
             <<",\"decoded_l2\":"<<decoded_error<<",\"preprocessed_l2\":"<<preprocessed_error<<",\"valid\":"<<(all_valid?"true":"false")
             <<",\"bank_wall_ms\":"<<bank_wall_ms
             <<",\"total_gpu_ms\":"<<total_gpu_ms<<",\"templates_per_gpu_second\":"<<(total_gpu_ms>0?bank.size()*1000/total_gpu_ms:0)
             <<",\"allocated_gpu_bytes\":"<<(gpu?gpu->allocated_bytes():0)<<",\"peak_rss_kib\":"<<usage.ru_maxrss
             <<",\"process_cpu_s\":"<<usage.ru_utime.tv_sec+usage.ru_utime.tv_usec/1e6+usage.ru_stime.tv_sec+usage.ru_stime.tv_usec/1e6<<"}\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<"brp_workunit: "<<e.what()<<'\n';return 1;}
