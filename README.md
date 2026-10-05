# GF180 DLA Accelerator: SRAM-Backed GEMM Engine, RTL to GDS

A small deep-learning accelerator (DLA) core written in Verilog and taken through a complete open-source RTL-to-GDS flow (OpenLane) on the GlobalFoundries GF180MCU 180 nm PDK. The core is a 4x4 INT8 matrix-multiply engine whose operand and result buffers are built from SRAM macros, and it is verified against quantized weights of a pretrained MNIST GAN.

> **Status:** the full flow (synthesis, placement, CTS, routing, LVS, streamout) completes and produces a GDS. Timing is clean at the typical corner but **not closed at the slow-slow corner**, and the Magic DRC step reports violations. Details are in [Physical design results](#physical-design-results). I list them openly because they are part of the engineering story.

## What is inside

| Path | Content |
|---|---|
| `rtl/` | Verilog RTL: PE, PE array, controller, SRAM-backed A/B/C buffer banks, SRAM wrapper, top level |
| `tb/` | SystemVerilog testbenches with self-checking, plus input/expected/actual vectors in `tb/data/` |
| `sim/` | Icarus Verilog run script (PowerShell) and simulation logs |
| `scripts/` | Python helpers: checkpoint to int8 conversion, test-vector generation, output verification |
| `weights_vh/` | Quantized int8 weights (`.memh` / `.vh`) and `weights_manifest.json` with shapes and scales |
| `openlane/` | OpenLane configuration, PDN config, macro placement and SRAM macro views (`.lef`, `.lib`, `.gds`) |
| `results/` | Final metrics (`metrics.json`, `metrics.csv`), final LEF and the compressed GDS |
| `Lane_notebook.ipynb` | LibreLane notebook used for early synthesis experiments |

## Architecture

The engine computes `C = A x B` for `N x K` and `K x N` INT8 matrices (default `N = K = 4`, accumulator width 24 bits).

- **PE array:** `N x N` processing elements, each a signed multiply-accumulate with `clear` and `en` controls. All 16 PEs work in parallel.
- **Dataflow:** output-stationary. Partial sums stay in the PE accumulators during the `K` accumulation cycles, so only the A and B operand streams move.
- **Controller:** FSM with `CLEAR`, `COMPUTE` (K cycles) and `DONE` phases and a `k_idx` counter that addresses the buffers.
- **Buffers:** A and B operands are stored in 256x8 1RW GF180 SRAM macros (one per row of A and one per column of B), and a C buffer holds results. SRAM outputs are registered, so the top level delays `en`, `done` and `busy` by `SRAM_LATENCY = 1` cycle.
- **Interface:** a shared write port (`wr_en`, `wr_sel`, `wr_addr`, `wr_data`) loads A and B, and results are read through `rd_addr` / `rd_data`, with `done` and `busy` status flags.

The accumulator is wider than the minimum required for four 8x8 signed products (`2*8 + ceil(log2(4)) = 18` bits) to leave headroom.

The RTL implements the matrix multiply only. Bias, activation and tiling for larger layers are handled outside the hardware, so this is a GEMM building block rather than a complete MNIST classifier or generator.

## Verification

Three self-checking testbenches run with Icarus Verilog and compare the hardware output against reference vectors generated in Python:

| Testbench | What it checks | Result |
|---|---|---|
| `dla_engine_top_d3004_tb` | Test case `d3004`, built from the D300 (discriminator) weights, against expected output | PASS |
| `dla_engine_top_g3005_tb` | Test case `g3005`, built from the G300 (generator) weights, against expected output | PASS |
| `g300_pipeline_tb` | Multi-step generator pipeline against expected output | PASS |

Logs are in `sim/results/`. To reproduce on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\sim\run_iverilog.ps1
```

### Quantized weights

Pretrained MNIST GAN MLP checkpoints (`D--300.ckpt`, `G--300.ckpt`) from [csinva/gan-vae-pretrained-pytorch](https://github.com/csinva/gan-vae-pretrained-pytorch) are converted to int8 with per-tensor scales. The checkpoints themselves are not redistributed here. Download them into `weights/mnist_gan_mlp/` and run:

```bash
python scripts/ckpt_to_vh.py --ckpt weights/mnist_gan_mlp/D--300.ckpt --out weights_vh/mnist_gan_mlp --prefix D300_
python scripts/ckpt_to_vh.py --ckpt weights/mnist_gan_mlp/G--300.ckpt --out weights_vh/mnist_gan_mlp --prefix G300_
```

## Physical design results

Flow: OpenLane (Classic flow, Verilator lint, Yosys, OpenROAD, Magic, KLayout, Netgen) on `gf180mcuD` with the `gf180mcu_fd_sc_mcu7t5v0` standard-cell library. Target clock period: **22 ns**. Source: `results/metrics.json`.

| Metric | Value |
|---|---|
| Die area | about 4.06 mm^2 |
| Instances | 35,243 total (35,232 standard cells + 11 SRAM macros) |
| Instance utilization | 30.2 % overall |
| Total power (reported) | 0.68 W |
| Routing DRC (detailed route) | 0 violations after 5 iterations |
| LVS | 0 errors (no device, net, pin or property differences) |

| Timing corner | Setup worst slack | Setup violations | Hold violations |
|---|---|---|---|
| nom_tt_025C_5v00 | +1.87 ns | 0 | 0 |
| nom_ff_n40C_5v50 | +6.43 ns | 0 | 0 |
| nom_ss_125C_4v50 | -8.57 ns | 310 | 0 |
| max_ss_125C_4v50 (worst) | -8.94 ns | 315 | 2 |

### Known issues

- **Timing is not closed at slow-slow.** At 22 ns the design meets timing at typical and fast corners but fails setup at the 125 C / 4.5 V slow corner. Realistic next steps: relax the clock period, pipeline the read path out of the SRAM macros, or retime the accumulate path.
- **Magic DRC reports 608 errors.** The flow was configured with `ERROR_ON_MAGIC_DRC: false`, so it continued past them. I have not yet triaged them, and they may be related to the SRAM macro views rather than the standard-cell logic.
- **4 antenna-violating nets** remain after antenna repair.

These are tracked as the main follow-up work.

### Layout files

`results/dla_engine_top.gds.gz` is the final layout (compressed to keep the repository small). Decompress and open it in KLayout:

```bash
gunzip -k results/dla_engine_top.gds.gz
klayout results/dla_engine_top.gds
```

## Third-party material

- SRAM macros: [RTimothyEdwards/gf180mcu_ocd_ip_sram](https://github.com/RTimothyEdwards/gf180mcu_ocd_ip_sram) (views included under `openlane/macro/`).
- PDK: [GlobalFoundries GF180MCU](https://github.com/google/gf180mcu-pdk), open source.
- Pretrained GAN checkpoints: see the link above, not included.

## Possible extensions

- Scale the array beyond 4x4 and measure the area/frequency trade-off.
- Add a weight-stationary dataflow option for comparison.
- Add bias, activation and tiling so a complete MLP layer runs in hardware.
- Close slow-slow timing and clear the DRC list.
