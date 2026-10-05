# llm-param-minimal-v1

This directory keeps the executable configuration, runner, checks, and ignored
results for the first stateless direct-five-parameter LLM experiment.

```bash
./experiments/llm_param_minimal_v1/run.sh preflight
./experiments/llm_param_minimal_v1/run.sh smoke
./experiments/llm_param_minimal_v1/run.sh stage1
./experiments/llm_param_minimal_v1/run.sh stage2
```

Run the stages in order. `smoke` is integration-only, `stage1` is the small
signal scan, and `stage2` is the clean-commit core scan. Each stage writes to
`results/<stage>/`; generated results are intentionally ignored by Git while
the scripts and requested configuration remain versioned.

The runner queries Ollama for the resolved `qwen2.5:7b` model metadata before a
result-bearing run and fails clearly if the server or model is unavailable.
