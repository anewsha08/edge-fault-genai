# genai/

Local LLM inference benchmarks and the RAG diagnostic assistant.

## Plan
- `bench/`: llama.cpp benchmark scripts (Q4 / Q5 / Q8, threads, context length)
- `kb/`: fault knowledge base in plain text or markdown (write your own or use public sources; no confidential material)
- `rag.py`: embed, index, retrieve top-k
- `diagnose.py`: receives the MCU report over UART, builds the prompt, calls the local model, prints the diagnosis

## Metrics to record
| Metric | Q4 | Q5 | Q8 |
|---|---|---|---|
| Tokens/sec (generation) | | | |
| Time to first token | | | |
| RAM | | | |
