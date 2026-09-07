2026-09-07 CPU verification; no CUDA/model initialization.

```sh
PYTHONPATH=/Users/wenyaogao/dev/sgl/sglang-omni-l5-eos48:/Users/wenyaogao/dev/sgl/sglang/python /Users/wenyaogao/dev/sgl/sglang-omni/.venv/bin/python -B -m pytest tests/unit_test/qwen3_omni/test_code2wav_eos_graph.py tests/unit_test/qwen3_omni/test_code2wav_eos48_graph.py tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py tests/unit_test/qwen3_omni/test_code2wav_batching.py tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py -m 'not accelerator' -q --disable-warnings --tb=short
```

Result: `213 passed, 11 deselected in 6.87s`.

Before implementation, the EOS48 test failed because `_build_eos_cuda_graph_runner`
was absent in L5. After the port, the 88 reused EOS35/EOS48 checks passed. The final
213-test suite includes the added L5 guard check and existing graph, batching and
SnakeBeta CPU regressions. Deselected accelerator tests were not run or claimed.
