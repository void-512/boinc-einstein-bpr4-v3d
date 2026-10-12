# BRP4 on Raspberry Pi V3D: research prototype

This C++20/Vulkan 1.3 project implements and measures orbital resampling, mean
padding, FFTs, normalized power spectra, 1/2/4/8/16-harmonic search, and candidate
compaction. It validates against the supplied historical Einstein@Home source
and FFTW. It is an experimental computational core, with synthetic inputs;
production BRP4 compatibility has **not** been demonstrated.

The reference repository is read only. Its CPU harmonic summing, resampling and
LUT utilities and FFT/power routine are compiled directly into the validation library. Small header
shims allow the installed FFTW runtime to be used without modifying the system.
BOINC filename resolution is an explicit failing stub, never used by validation.

## Build and CPU tests

Requires a C++20 compiler, CMake 3.20+, Vulkan development headers/loader,
`glslc`, Python 3, and the single-precision FFTW runtime (`libfftw3f.so.3`).
Actual experiments used GCC 14.2, Raspberry Pi 5, Mesa V3DV 26.2.2, and Vulkan
1.3.354. Access to `/dev/dri/renderD128` is required for hardware runs.

```bash
cd /home/wty/Cache/einstine@home
mkdir -p "$HOME/.cache/codex-tmp"
export TMPDIR="$HOME/.cache/codex-tmp"
cmake -S . -B "$HOME/.cache/codex-tmp/brp-v3d-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/.cache/codex-tmp/brp-v3d-build" -j4
ctest --test-dir "$HOME/.cache/codex-tmp/brp-v3d-build" --output-on-failure
```

Build and shader outputs are disposable and will disappear after reboot.
Source, documentation, and measured results remain in this repository.

## Validate and benchmark on the GPU

The harness manages only the task explicitly authorized in the prompt:
`ap27_23650380_1` at `http://www.primegrid.com/`. It inspects the initial state,
suspends an executing task, waits for a stopped/exited process or BOINC suspended
state, runs a bounded subprocess group, and restores the original executing
state in `finally`. An already stopped task is left stopped. Failures and signals
also restore the task. No benchmark runs while suspension is unverified.
BOINC's `QUIT_PENDING` alone is insufficient: the process may still use the GPU.
The harness permits up to 180 seconds for a normal task state transition.
As with any process cleanup, SIGKILL or a power loss can prevent `finally`.

```bash
python3 tools/exclusive_gpu.py --timeout 180 -- \
  "$HOME/.cache/codex-tmp/brp-v3d-build/brp_bench" \
  --n 65536 --wg 256 --tile 0 --stockham --radix4 --batch 8 --reps 31

python3 tools/exclusive_gpu.py --timeout 180 -- \
  python3 tools/sweep.py "$HOME/.cache/codex-tmp/brp-v3d-build/brp_bench" \
  --mode validate --output results/validation.jsonl

python3 tools/exclusive_gpu.py --timeout 180 -- \
  python3 tools/sweep.py "$HOME/.cache/codex-tmp/brp-v3d-build/brp_bench" \
  --mode radix --output results/optimization.jsonl
```

Run these commands outside a sandbox that hides the GPU. They do not require
root when normal GPU and BOINC permissions are available. The backend refuses
silent software fallback. `--allow-software` is an explicit debugging opt-in;
software measurements must never be described as V3D results.

`brp_bench --help` lists options. `--tile 0` selects a global radix-2 FFT;
`--tile 256`, `512`, or `1024` fuse its first stages in workgroup shared memory.
`--stockham --tile 0` selects the alternative ping-pong FFT, avoiding the explicit
bit-reversal permutation. `--radix4` fuses pairs of radix-2 stages, preserving
their arithmetic order. `--wg` is independent of FFT tile size. Dense per-stage
twiddles are the default; `-DBRP_STRIDED_TWIDDLES=ON` reproduces the alternative.

JSON contains correctness checks, FFT errors, candidate/rank mismatches,
initialization time, per-stage GPU timestamps, wall and process CPU times,
memory allocation and peak RSS. The sweep retains individual wall-time samples,
commands, exit codes, timestamps, temperature, GPU clock and throttle flags.
Exit status 2 means numerical/detection validation failed; status 1 means a
runtime/overflow/input failure. Neither is a valid scientific result.

`--batch` replays the **same template** multiple times in one reusable command
buffer; only the final replay's output survives. These are template-equivalent
kernel throughput experiments, not multiple distinct templates or complete work
units. They measure submission amortization, but production batching still needs
independent candidate retention and evolving search thresholds.

## Reproduce the corrected baseline and final suite

```bash
cmake -S . -B "$HOME/.cache/codex-tmp/brp-v3d-baseline" \
  -DCMAKE_BUILD_TYPE=Release -DBRP_DIVISION_FFT=ON -DBRP_SCATTER_PREPARE=ON
cmake --build "$HOME/.cache/codex-tmp/brp-v3d-baseline" -j4
```

Use that executable with `tools/sweep.py --mode sweep`. Baseline shaders remain
under `experiments/baseline/`. Stockham is a separate alternative and must not be
used with the legacy scattered prepare shader. Correctness is mandatory for all
reported configurations.

With both builds available, the complete validation, algorithm sweep, failure
checks, corrected baseline and three-trial confirmation run is:

```bash
python3 -B tools/exclusive_gpu.py --timeout 600 -- \
  python3 -B tools/final_suite.py \
  "$HOME/.cache/codex-tmp/brp-v3d-build/brp_bench" \
  "$HOME/.cache/codex-tmp/brp-v3d-baseline/brp_bench"
python3 -B tools/summarize.py results/baseline-corrected.jsonl \
  results/radix-fusion-strided.jsonl results/radix-fusion-dense.jsonl \
  results/confirmation.jsonl --output results/summary.md
```

These commands overwrite the named experiment records; copy them first to
preserve a previous run. The summary selects the fastest validated record in
each set; the report also gives medians across independent confirmation trials.
The strided sweep is a separate experiment: build with
`BRP_STRIDED_TWIDDLES=ON` and run `tools/sweep.py --mode radix` to regenerate it.

Final synthetic replay medians with WG256/Stockham/radix-4/dense twiddles were
3.278 ms at N=65,536 and 52.474 ms at N=1,048,576, with batch 8. This is 3.17×
and 7.71× faster than the corrected GPU baseline, respectively. CPU reference
times were about 2.16 and 47.54 ms, so GPU throughput has not overtaken the CPU.
See [measured records](results/summary.md) and [the report](docs/research.md).

For optional API validation, `tools/api_validation.py EXECUTABLE LAYER_ROOT`
runs both FFT paths with a Khronos validation layer extracted under `LAYER_ROOT`
(Debian arm64 layout), verifies that it loaded, and retains the diagnostics.
Run it through `tools/exclusive_gpu.py`. Layer timings are diagnostic only.

See [the research report](docs/research.md) for source analysis, measured findings,
precision limitations, and remaining work. Results under `results/` are durable
experiment records. This project follows GPL-2.0-or-later as required by the
historical scientific implementations; see `LICENSE` and source attribution.
