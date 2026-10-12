#pragma once
#include "brp/science.hpp"
#include <memory>
#include <string>
namespace brp {
struct Timings { double wall_ms, cpu_ms, submit_ms; std::array<double,4> stage_ms; std::vector<std::array<double,4>> template_stage_ms; };
struct Snapshot {
    std::vector<float> resampled, power, summed;
    std::vector<std::complex<float>> fft;
    std::vector<Candidate> candidates;
};
class GPU {
public:
    GPU(const Params&,std::span<const float>,uint32_t workgroup,uint32_t tile,uint32_t batch,bool software=false,bool stockham=false,bool radix4=false,bool packed=false,bool scatter=false,bool radix8=false,bool fused_power=false,bool radix8_store=false);
    ~GPU();
    GPU(const GPU&)=delete;
    GPU& operator=(const GPU&)=delete;
    Timings run();
    void set_template(const Params&);
    void set_templates(std::span<const Params>);
    Snapshot snapshot() const;
    std::vector<Candidate> candidates(uint32_t slot=UINT32_MAX) const;
    std::string device_info() const;
    uint64_t allocated_bytes() const;
    std::string profile_counters();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
