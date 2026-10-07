# Roadmap (about 10 weeks, part-time)

Check items off as you finish them. Commit after each one.

## Phase 0: Setup (days 1-3)
- [ ] Create GitHub repo `edge-fault-genai`, push this skeleton
- [ ] Decide hardware: Cortex-M board (Pico 2 / STM32 Nucleo / micro:bit v2), accelerometer (ADXL345 or MPU6050), Raspberry Pi
- [ ] Install toolchain: `arm-none-eabi-gcc`, CMake, and the board SDK (Pico SDK or STM32Cube / Zephyr)
- [ ] Blink an LED and print over UART to confirm the toolchain works
- [ ] Download the CWRU bearing dataset (or another public vibration dataset) into `ml/data/` (git-ignored)

## Phase 1: Sampling firmware (weeks 1-2) - course: Embedded Systems on Arm
- [ ] Read the accelerometer over I2C/SPI (polling first, to prove the wiring)
- [ ] Switch to a hardware timer interrupt for fixed-rate sampling (e.g., 1 kHz)
- [ ] Add DMA or a ring buffer; no sample loss at full rate
- [ ] Build sliding windows of 256 samples with 50% overlap
- [ ] Stream raw windows over UART to the PC and plot them (Python)
- [ ] Measure and document: sample jitter, CPU load, RAM used
- Deliverable: `firmware/` builds cleanly, README section with a block diagram and a jitter measurement

## Phase 2: DSP on the MCU (week 3)
- [ ] FFT with CMSIS-DSP; extract features (RMS, peak frequency, band energies)
- [ ] Cycle counts using the DWT cycle counter
- [ ] Compare against a naive DFT to show the speedup
- Deliverable: feature output matches a NumPy reference within tolerance

## Phase 3: TinyML classifier (weeks 4-5) - course: ML at the Edge on Arm
- [ ] Prepare dataset windows and labels (normal, inner race, outer race, ball fault)
- [ ] Train a small model (1D CNN or MLP on features) in Python
- [ ] Quantize to int8; compare accuracy float vs int8
- [ ] Deploy with TFLite Micro + CMSIS-NN; measure latency, flash, RAM
- Deliverable: confusion matrix, size and latency table

## Phase 4: Arm kernel optimization on Cortex-A (weeks 6-7) - course: Computer Architecture
- [ ] int8 matrix-vector multiply: scalar C
- [ ] Same kernel with NEON intrinsics (use dot-product instructions if the CPU supports them)
- [ ] Cache tiling; measure the effect
- [ ] Read the generated assembly and explain it in the write-up
- Deliverable: speedup chart plus a "why it got faster" explanation

## Phase 5: LLM inference benchmark (week 8) - course: Optimizing GenAI on Arm
- [ ] Build llama.cpp on the Pi
- [ ] Benchmark a 1-3B model at Q4 / Q5 / Q8, varying threads and context length
- [ ] Record tokens/sec, time to first token, RAM
- [ ] Decide: compute-bound or memory-bandwidth-bound, with evidence
- Deliverable: benchmark report in `docs/`

## Phase 6: RAG and integration (weeks 9-10)
- [ ] Write the fault knowledge base (fault descriptions, causes, checks) in `genai/kb/`
- [ ] Embed, index (FAISS or SQLite), retrieve top-k
- [ ] Define the UART message from the MCU: fault class, confidence, key features
- [ ] Prompt builder: MCU report + retrieved chunks -> LLM -> diagnosis
- [ ] End-to-end demo video and final README with results
- Deliverable: demo GIF, complete results table, lessons learned

## Rules for yourself
1. Every phase ends with a measured number in the README.
2. Do not use confidential Cummins material; write the knowledge base yourself or use public sources.
3. Commit small and often; a clean history is part of the portfolio.
