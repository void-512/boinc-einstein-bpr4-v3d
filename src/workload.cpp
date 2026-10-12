// GPL-2.0-or-later. Scientific ordering follows demod_binary.c, source attribution in docs.
#include "brp/workload.hpp"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <zlib.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include "rngmed.h"
extern "C" {
typedef struct fftwf_plan_s* fftwf_plan;
fftwf_plan fftwf_plan_dft_r2c_1d(int,float*,float (*)[2],unsigned);
fftwf_plan fftwf_plan_dft_c2r_1d(int,float (*)[2],float*,unsigned);
void fftwf_execute(fftwf_plan);
void fftwf_destroy_plan(fftwf_plan);
}
namespace brp {
Observation load_observation(const std::filesystem::path& path,double padding) {
    if(std::endian::native!=std::endian::little) throw std::runtime_error("DD loader currently requires little-endian host");
    auto stream=gzopen(path.c_str(),"rb");if(!stream) throw std::runtime_error("cannot open DD input");
    struct Close {gzFile s;~Close(){gzclose(s);}} close{stream};
    std::array<unsigned char,1168> header{};
    if(gzread(stream,header.data(),header.size())!=int(header.size())) throw std::runtime_error("truncated DD header");
    auto d=[&](size_t i){double v;std::memcpy(&v,header.data()+i*8,8);return v;};
    Observation out;out.sample_us=d(0);out.observation_s=d(1);out.dm=d(13);out.scale=d(14);out.padding=padding;
    std::memcpy(&out.samples,header.data()+128,4);
    if(!out.samples||out.samples>(1u<<24)||!std::isfinite(out.scale)||out.scale<=0||
       !std::isfinite(out.sample_us)||out.sample_us<=0||!std::isfinite(padding)||padding<1||padding>10)
        throw std::runtime_error("invalid DD metadata");
    double size=padding*out.samples+0.5;
    if(size>3*(1u<<22)+0.5) throw std::runtime_error("observation exceeds supported length");
    out.n=uint32_t(size);
    bool packed=path.extension()==".bin4";
    if(!packed&&path.extension()!=".binary") throw std::runtime_error("unsupported DD sample format");
    if(packed&&(out.samples&1)) throw std::runtime_error("odd packed DD length");
    std::vector<unsigned char> bytes(packed?out.samples/2:out.samples);
    if(gzread(stream,bytes.data(),bytes.size())!=int(bytes.size())) throw std::runtime_error("truncated DD samples");
    unsigned char extra;
    if(gzread(stream,&extra,1)!=0) throw std::runtime_error("trailing DD bytes or gzip error");
    out.decoded.resize(out.samples);
    for(uint32_t i=0;i<bytes.size();++i) {
        if(packed) {out.decoded[2*i]=float(double(bytes[i]>>4)/out.scale);out.decoded[2*i+1]=float(double(bytes[i]&15)/out.scale);}
        else out.decoded[i]=float(double(int8_t(bytes[i]))/out.scale);
    }
    return out;
}
void preprocess(Observation& o,const std::filesystem::path& zap,uint32_t window) {
    if(!window||(window&1)||window>o.n/2) throw std::runtime_error("unsupported whitening window");
    std::vector<float> time(o.n);std::copy(o.decoded.begin(),o.decoded.end(),time.begin());
    std::vector<std::complex<float>> fft(o.n/2+1);
    auto plan=fftwf_plan_dft_r2c_1d(o.n,time.data(),reinterpret_cast<float (*)[2]>(fft.data()),64);
    if(!plan) throw std::runtime_error("whitening R2C plan failed");
    fftwf_execute(plan);fftwf_destroy_plan(plan);
    std::vector<float> power(fft.size()),median(fft.size()-window+1);
    for(size_t i=1;i<fft.size();++i) {double re=fft[i].real(),im=fft[i].imag();power[i]=float(re*re+im*im);}
    rngmed(power.data(),power.size(),window,median.data());
    for(size_t i=0;i<median.size();++i) {
        if(!(median[i]>0)||!std::isfinite(median[i])) throw std::runtime_error("invalid whitening median");
        float factor=std::sqrt(std::numbers::ln2/median[i]);fft[i+window/2]*=factor;
    }
    uint32_t seed=std::bit_cast<uint32_t>(o.decoded[0]);
    std::unique_ptr<gsl_rng,decltype(&gsl_rng_free)> rng(gsl_rng_alloc(gsl_rng_taus2),gsl_rng_free);
    if(!rng) throw std::runtime_error("GSL allocation failed");
    gsl_rng_set(rng.get(),seed);
    std::ifstream file(zap);if(!file) throw std::runtime_error("cannot open zap list");
    double lower,upper;double seconds=o.n*o.sample_us*1e-6;
    while(file>>lower>>upper) {
        if(!std::isfinite(lower+upper)||lower<0||upper<lower) throw std::runtime_error("invalid zap range");
        double first=lower*seconds+0.5,last=upper*seconds+0.5;
        if(last>=fft.size()) throw std::runtime_error("zap exceeds Nyquist");
        for(uint32_t i=uint32_t(first);i<=uint32_t(last);++i) {
            double sigma=std::numbers::sqrt2/2*std::sqrt(o.padding);
            float real=gsl_ran_gaussian_ziggurat(rng.get(),sigma),imag=gsl_ran_gaussian_ziggurat(rng.get(),sigma);
            fft[i]={real,imag};
        }
    }
    if(!file.eof()) throw std::runtime_error("malformed zap list");
    for(uint32_t i=0;i<window/2;++i) fft[i]=fft[fft.size()-i-1]={0,0};
    plan=fftwf_plan_dft_c2r_1d(o.n,reinterpret_cast<float (*)[2]>(fft.data()),time.data(),64);
    if(!plan) throw std::runtime_error("whitening C2R plan failed");
    fftwf_execute(plan);fftwf_destroy_plan(plan);
    float factor=1.0/std::sqrt(float(o.n));o.preprocessed.resize(o.samples);
    for(uint32_t i=0;i<o.samples;++i) o.preprocessed[i]=time[i]*factor;
}
std::vector<Orbit> load_bank(const std::filesystem::path& path) {
    std::ifstream file(path);if(!file) throw std::runtime_error("cannot open orbital bank");
    std::vector<Orbit> out;std::string line;
    while(std::getline(file,line)) {
        Orbit p{};std::string extra;std::istringstream row(line);
        if(!(row>>p.period>>p.tau>>p.psi)||(row>>extra)||!std::isfinite(p.period+p.tau+p.psi)||p.period<=0||p.tau<0)
            throw std::runtime_error("invalid orbital bank row");
        out.push_back(p);
    }
    if(out.empty()) throw std::runtime_error("empty orbital bank");
    return out;
}
Params search_parameters(const Observation& o,const Orbit& orbit,double f0,double false_alarm,uint32_t window) {
    if(!std::isfinite(f0+false_alarm)||f0<=0||false_alarm<=0||false_alarm>=1||!window) throw std::runtime_error("invalid search parameters");
    auto p=parameters(o.n,o.samples,float(orbit.tau),float(orbit.period),float(orbit.psi),float(o.sample_us*1e-6));
    p.flags=2|4; // Production FMA, compensated on V3D.
    p.s0=(p.tau*std::sin(p.psi))*p.step_inv;
    resampling_steps(p);
    double seconds=o.n*o.sample_us*1e-6;
    p.window=uint32_t(window*0.5+0.5);
    if(p.window>=p.n/2+1) throw std::runtime_error("search window exceeds spectrum");
    double limit=p.n/2+1-p.window;
    p.fundamental=uint32_t(std::min(limit,f0*seconds+0.5));
    p.harmonic=uint32_t(std::min(limit,16*f0*seconds+0.5));
    float probability=1.0-std::pow(1.0-false_alarm,1.0/(p.n/2+1));
    if(!(probability>0&&probability<1)) throw std::runtime_error("false-alarm probability outside representable domain");
    for(uint32_t h=0;h<5;++h) p.threshold[h]=0.5*gsl_cdf_chisq_Qinv(probability,2.0*(1u<<h));
    if(p.fundamental<=p.window) throw std::runtime_error("empty frequency search");
    return p;
}
void Toplist::thresholds(Params& p) const {for(uint32_t h=0;h<5;++h) p.threshold[h]=std::max(p.threshold[h],float(rows[(h+1)*100-1].power));}
static int by_power(const void* x,const void* y) {auto a=static_cast<const BankCandidate*>(x),b=static_cast<const BankCandidate*>(y);return a->power<b->power?1:(a->power>b->power?-1:0);}
static int by_significance(const void* x,const void* y) {auto a=static_cast<const BankCandidate*>(x),b=static_cast<const BankCandidate*>(y);return a->significance<b->significance?1:(a->significance>b->significance?-1:0);}
void Toplist::retain(std::span<const Candidate> candidates,const Orbit& orbit,const Params& p) {
    for(auto c:candidates) {
        if(c.level>=5||c.bin<p.window||c.bin>=p.fundamental||!std::isfinite(c.power)) throw std::runtime_error("invalid compact candidate");
        uint32_t first=c.level*100,last=first+99;
        if(c.power<=p.threshold[c.level]||c.power<=rows[last].power) continue;
        int place=last;
        for(uint32_t i=first;i<=last;++i) if(rows[i].bin==c.bin) {place=rows[i].power<c.power?int(i):-1;break;}
        if(place>=int(first)) {
            rows[place]={c.power,float(orbit.period),float(orbit.tau),float(orbit.psi),0,1u<<c.level,c.bin};
            std::qsort(rows.data()+first,100,sizeof(BankCandidate),by_power);
        }
    }
}
std::vector<BankCandidate> Toplist::final() const {
    auto copy=rows;
    for(auto& c:copy) {
        if(!c.harmonics) {c.significance=-10;continue;}
        double probability=gsl_cdf_chisq_Q(2*c.power,2*c.harmonics);
        c.significance=probability>0?-std::log10(probability):320;
        // Historical main uses FP32 square roots for sigma2 and sigma8.
        c.power/=float(std::sqrt(float(c.harmonics)));
    }
    std::qsort(copy.data(),copy.size(),sizeof(BankCandidate),by_significance);
    std::vector<BankCandidate> out;
    while(out.size()<100&&copy[0].significance>0) {
        out.push_back(copy[0]);auto bin=copy[0].bin;
        for(auto& c:copy) if(c.bin==bin) c.significance=-10;
        std::qsort(copy.data(),copy.size(),sizeof(BankCandidate),by_significance);
    }
    return out;
}
void write_candidates(const std::filesystem::path& path,std::span<const BankCandidate> rows,double seconds) {
    auto file=std::fopen(path.c_str(),"w");if(!file) throw std::runtime_error("cannot write offline candidates");
    std::fprintf(file,"%% Experimental offline BRP4 results; never submit to production\n");
    for(auto c:rows) std::fprintf(file,"%6.12f %6.12f %6.12f %6.12f %g %g %d\n",c.bin/seconds,c.period,c.tau,c.psi,c.power,c.significance,c.harmonics);
    std::fprintf(file,"%%DONE%%\n");if(std::fclose(file)) throw std::runtime_error("candidate write failed");
}
}
