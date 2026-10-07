# Edge Fault Detection + GenAI Diagnostic Assistant on Arm

A Cortex-M node detects machine faults from vibration data in real time (TinyML, int8).
A Cortex-A board (Raspberry Pi) runs a local quantized LLM with RAG over fault manuals and
explains the fault in plain language and suggests next checks. Fully offline.

**Problem:** Factory owners wait days for industrial repair technicians. Diagnosing faults
on-site, before the technician arrives, cuts downtime.

> Status: Phase 0 (setup). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Architecture

```
[Accelerometer] -> [Cortex-M node] --UART--> [Cortex-A (Raspberry Pi)]
                   timer + DMA sampling        llama.cpp (quantized LLM)
                   CMSIS-DSP FFT               + RAG over fault manuals
                   CMSIS-NN int8 classifier    -> plain-language diagnosis
```

## Repo layout

| Folder | Purpose |
|---|---|
| `firmware/` | Cortex-M code: sampling, FFT, inference, UART protocol |
| `ml/` | Dataset prep, training, quantization, model export |
| `kernels/` | Hand-written scalar vs NEON vs tiled kernels for the Cortex-A |
| `genai/` | llama.cpp benchmarks, RAG pipeline, diagnosis app |
| `docs/` | Roadmap, design notes, benchmark reports, diagrams |

## Results (fill in as you go)

| Metric | Result |
|---|---|
| MCU inference latency, float vs int8 | TBD |
| Model flash / RAM | TBD |
| FFT cycles (CMSIS-DSP vs naive) | TBD |
| NEON vs scalar kernel speedup | TBD |
| LLM tokens/sec (Q4 / Q5 / Q8) | TBD |
| Fault-to-diagnosis latency | TBD |

## Lessons learned

(Write this section as you go. It is the most valuable part for reviewers.)
