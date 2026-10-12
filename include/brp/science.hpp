#pragma once
#include <array>
#include <complex>
#include <cstdint>
#include <span>
#include <vector>
namespace brp {
struct Params {
    uint32_t n, unpadded, steps, logn;
    float tau, omega, psi, dt, step_inv, s0;
    uint32_t fundamental, harmonic, window, stage;
    float threshold[5];
    uint32_t capacity;
    uint32_t fft_n, fft_base, fft_radix, flags, twiddle3, twiddle_real;
    uint32_t result_slot;
};
static_assert(sizeof(Params)==108);
struct Candidate { uint32_t bin, level; float power; uint32_t reserved; };
struct Reference {
    std::vector<float> resampled, power, summed;
    std::vector<std::complex<float>> fft;
    std::vector<Candidate> candidates;
};
float lut_sin(float phase);
float sample_offset(uint32_t i,const Params&);
void resampling_steps(Params&);
Params parameters(uint32_t n, uint32_t unpadded, float tau, float period, float psi, float dt);
std::vector<float> synthetic(uint32_t n, float dt, uint32_t seed=42);
std::vector<float> resample(std::span<const float> input, const Params& p);
std::vector<float> historical_resample(std::span<const float> input, const Params& p);
std::vector<float> historical_power(std::span<const float> resampled, const Params& p);
struct CPUStats {double wall_ms,cpu_ms;};
CPUStats benchmark_cpu(std::span<const float> input,const Params& p,uint32_t reps);
std::vector<std::complex<float>> fft_reference(std::span<const float> input);
std::vector<float> harmonic_reference(std::span<const float> power, const Params& p);
std::vector<Candidate> select(std::span<const float> summed, const Params& p);
Reference reference(std::span<const float> input, const Params& p);
double relative_l2(std::span<const float> actual, std::span<const float> expected);
}
