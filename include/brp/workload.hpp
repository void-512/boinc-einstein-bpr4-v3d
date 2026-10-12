#pragma once
#include "brp/science.hpp"
#include <filesystem>
#include <string>
namespace brp {
struct Observation {
    double sample_us{}, observation_s{}, dm{}, scale{}, padding{};
    uint32_t samples{}, n{};
    std::vector<float> decoded, preprocessed;
};
struct Orbit {double period,tau,psi;};
Observation load_observation(const std::filesystem::path&,double padding);
void preprocess(Observation&,const std::filesystem::path& zap,uint32_t window=1000);
std::vector<Orbit> load_bank(const std::filesystem::path&);
Params search_parameters(const Observation&,const Orbit&,double f0=400,double false_alarm=0.08,uint32_t window=1000);
struct BankCandidate {double power{},period{},tau{},psi{},significance{};uint32_t harmonics{},bin{};};
class Toplist {
    std::array<BankCandidate,500> rows{};
public:
    void thresholds(Params&) const;
    void retain(std::span<const Candidate>,const Orbit&,const Params&);
    std::vector<BankCandidate> final() const;
};
void write_candidates(const std::filesystem::path&,std::span<const BankCandidate>,double observation_s);
}
