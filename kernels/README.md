# kernels/

Hand-written compute kernels for the Cortex-A (Raspberry Pi, AArch64).

## Plan
- `matvec_scalar.c`: int8 matrix-vector multiply, plain C
- `matvec_neon.c`: NEON intrinsics (use `vdotq_s32` if the CPU supports dot product)
- `matvec_tiled.c`: cache-tiled version
- `bench.c`: timing harness (many repetitions, report median)

## Method
1. Verify all versions give identical results.
2. Benchmark across several matrix sizes, including ones larger than L1 and L2.
3. Inspect the assembly (`objdump -d`) and explain what the compiler did in each case.
4. Plot speedup vs size and write up what limited performance.
