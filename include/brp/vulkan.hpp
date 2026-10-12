#pragma once
#include "brp/science.hpp"
#include <memory>
#include <string>
namespace brp {
struct Timings { double wall_ms, cpu_ms, submit_ms; std::array<double,4> stage_ms; };
struct Snapshot {
    std::vector<float> resampled, power, summed;
    std::vector<std::complex<float>> fft;
    std::vector<Candidate> candidates;
};
class GPU {
public:
    GPU(const Params&,std::span<const float>,uint32_t workgroup,uint32_t tile,uint32_t batch,bool software=false,bool stockham=false,bool radix4=false);
    ~GPU();
    GPU(const GPU&)=delete;
    GPU& operator=(const GPU&)=delete;
    Timings run();
    Snapshot snapshot() const;
    std::string device_info() const;
    uint64_t allocated_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
