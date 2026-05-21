# Ming Streaming TTS Test Evidence

Branch: `feat/migrate-ming-streaming-tts`

## CI Test Layout

The repository CI layout gate in `.github/workflows/test-layout.yaml` allows
Python tests only in these lanes:

- `tests/unit_test/`
- `tests/docs/`
- `tests/test_model/`

This branch adds and updates tests only under the unit-test lane:

- `tests/unit_test/ming_omni/test_streaming_text.py`
- `tests/unit_test/ming_omni/test_streaming_segmenter_scheduler.py`
- `tests/unit_test/ming_omni/test_streaming_decode_scheduler.py`
- `tests/unit_test/ming_omni/test_streaming_talker_scheduler.py`
- `tests/unit_test/ming_omni/test_streaming_pipeline_config.py`
- `tests/unit_test/ming_omni/test_streaming_client_api.py`
- `tests/unit_test/ming_omni/test_thinker_stream_output.py`
- `tests/unit_test/serve/test_openai_api.py`

Local layout check matching CI:

```bash
invalid_files=()
while IFS= read -r path; do
  case "$path" in
    tests/__init__.py|tests/utils.py) ;;
    tests/unit_test/*|tests/docs/*|tests/test_model/*) ;;
    *) invalid_files+=("$path") ;;
  esac
done < <(find tests -type f -name '*.py' | sort)
if (( ${#invalid_files[@]} )); then
  printf '%s\n' "${invalid_files[@]}"
  exit 1
fi
```

Result: passed.

## Local Test Commands

The full CI command from `.github/workflows/test.yaml` is:

```bash
pytest tests/ -v -m "not benchmark and not docs" -x
```

In the local macOS arm64 worktree, project dependency resolution is blocked by
platform-specific packages, so the branch was verified with `uv run --no-project`
and the lightweight dependencies needed by the affected test lanes.

Focused Ming streaming and OpenAI API tests:

```bash
uv run --no-project \
  --with pytest \
  --with pytest-asyncio \
  --with pydantic \
  --with numpy \
  --with torch \
  --with fastapi \
  --with httpx \
  --with msgpack \
  --with pyzmq \
  --with uvicorn \
  pytest \
    tests/unit_test/ming_omni/test_streaming_text.py \
    tests/unit_test/ming_omni/test_streaming_segmenter_scheduler.py \
    tests/unit_test/ming_omni/test_streaming_decode_scheduler.py \
    tests/unit_test/ming_omni/test_streaming_talker_scheduler.py \
    tests/unit_test/ming_omni/test_streaming_pipeline_config.py \
    tests/unit_test/ming_omni/test_streaming_client_api.py \
    tests/unit_test/ming_omni/test_thinker_stream_output.py \
    tests/unit_test/serve/test_openai_api.py \
    -q
```

Result: `80 passed`.

Ming baseline unit tests:

```bash
uv run --no-project \
  --with pytest \
  --with pytest-asyncio \
  --with pydantic \
  --with numpy \
  --with torch \
  --with fastapi \
  --with httpx \
  --with msgpack \
  --with pyzmq \
  --with uvicorn \
  pytest \
    tests/unit_test/ming_omni/test_pipeline.py \
    tests/unit_test/ming_omni/test_talker.py \
    tests/unit_test/ming_omni/test_thinker.py \
    -q
```

Result: `36 passed`.

Serve API focused tests:

```bash
uv run --no-project \
  --with pytest \
  --with pytest-asyncio \
  --with pydantic \
  --with numpy \
  --with torch \
  --with fastapi \
  --with httpx \
  --with msgpack \
  --with pyzmq \
  --with uvicorn \
  pytest tests/unit_test/serve/test_openai_api.py -q
```

Result: `4 passed`.

Task 7 client/API subset:

```bash
uv run --no-project \
  --with pytest \
  --with pytest-asyncio \
  --with pydantic \
  --with numpy \
  --with torch \
  --with fastapi \
  --with httpx \
  --with msgpack \
  --with pyzmq \
  --with uvicorn \
  pytest \
    tests/unit_test/ming_omni/test_streaming_client_api.py \
    tests/unit_test/serve/test_openai_api.py \
    -q
```

Result: `14 passed`.

## Lint

CI lint is defined in `.github/workflows/lint.yaml` as:

```bash
pre-commit run --all-files --show-diff-on-failure
```

Local equivalent:

```bash
uv run --no-project --with pre-commit \
  pre-commit run --all-files --show-diff-on-failure
```

Result: passed.

## Not Run Locally

Adjacent streaming tests:

- `tests/unit_test/qwen3_omni/test_streaming.py`
- `tests/unit_test/fishaudio_s2_pro/test_streaming_vocoder.py`

Local no-project collection was blocked by:

```text
ModuleNotFoundError: No module named 'sglang'
```

GPU quality gates were not run locally because this environment is macOS arm64
with no CUDA devices. These remain required on the GPU validation host before
enabling this path beyond the opt-in branch:

- Baseline and streaming Ming server startup.
- TTFA p50/p95.
- RTF comparison.
- Sample rate validation.
- Empty audio chunk and duplicate-final-audio checks.
- WER or text-audio consistency comparison.
- Cancellation cleanup with a later successful request.
