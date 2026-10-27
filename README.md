# chon

Serving code for **Chon** typed-decision models:

| Model | Base model | GPU memory (bf16) | Notes |
|---|---|---|---|
| [Chon-1.0-12B](https://huggingface.co/ninhnv/chon-1.0-12b) | `google/gemma-4-12B-it` | about 40 GB (+ ~30 GB for vLLM) | most accurate; thinks briefly on unsure questions |
| [Chon-1.0-12B-Fast](https://huggingface.co/ninhnv/chon-1.0-12b-fast) | `google/gemma-4-12B-it` | about 40 GB | the same model, always one forward pass |
| [Chon-1.0-E4B](https://huggingface.co/ninhnv/chon-1.0-e4b) | `google/gemma-4-E4B-it` | 24 GB or more | smaller; Macs with 24 GB or more |
| [Chon-1.0-E2B](https://huggingface.co/ninhnv/chon-1.0-e2b) | `google/gemma-4-E2B-it` | 16 GB or more | smallest and fastest; about 1 s per decision on an 18 GB MacBook Pro |

Given a state (text, JSON or a conversation) and one question with a closed set of options, a Chon model
returns a calibrated probability for every option:

- **choice:** 2-256 options (more than 16 are read in groups of at most 16, then a final round between the
  group winners; every option still gets a probability)
- **yes/no:** P(yes)
- **score:** a distribution over ordered levels

The server speaks the TypeSafe `/v1/systemone` wire format, so tools built for that format, such as the
JevBench `typesafe` adapter, work unchanged.

## Install

```bash
pip install "chon[serve] @ git+https://github.com/xoai/chon"
```

Tested with Python 3.12, torch 2.9.1 (CUDA 12.8), transformers 5.19-5.20 and peft 0.18.1, on one NVIDIA
H100 80 GB. Access to the model's Gemma 4 base model on Hugging Face is required.

## Serve

```bash
python -m chon.serve --model ninhnv/chon-1.0-12b --port 8000
```

`--model` takes a Hugging Face id (optionally `id@revision`) or a local folder holding the model files.
The base model named in the model's `chon_config.json` is loaded with it.

To run fully offline, download both repositories first and point `--model` at the local folder:

```bash
hf download google/gemma-4-12B-it
hf download ninhnv/chon-1.0-12b --local-dir ./chon-1.0-12b
HF_HUB_OFFLINE=1 python -m chon.serve --model ./chon-1.0-12b --port 8000
```

Options:
- `--host`, `--port`: where to listen.
- `CHON_API_KEY`: if set in the environment, requests must send `Authorization: Bearer <key>`.
- `--quantize fp8`: optional FP8 weights (needs `chon[fp8]`). Re-check accuracy if you use it.
- `--no-think`: always answer in a single forward pass, even for a model whose `chon_config.json` turns on
  a short extra pass for unsure questions.
- `--think-set KEY=VALUE`: override one setting of that extra pass, e.g. `--think-set engine=hf-static` to
  skip vLLM (below).
- `--deadline SECONDS` (default 8; `CHON_DEADLINE_S`; `0` turns it off): every reply is sent within this many
  seconds of the request's arrival, time spent waiting for the GPU included. The single-pass answer is always
  computed first and kept; the extra pass for an unsure question runs only with the time left (shortened, or
  skipped when too little remains) and is dropped if the clock runs out, so a client with a time limit never
  waits past it for a better answer. `/health` reports how often the extra pass was shortened, skipped or cut.

### Faster reasoning (vLLM)

Chon-1.0-12B answers most questions in one forward pass (about 0.1 s). On questions it is unsure about, it
first writes up to 512 tokens of reasoning with the plain base model, then reads its answer. The server hands
that writing step to a [vLLM](https://github.com/vllm-project/vllm) server of the base model (FP8 weights),
which it starts itself after loading the model, with the GPU memory left over. vLLM goes in its own
environment, because it brings its own torch:

```bash
python -m venv /opt/vllm-env && /opt/vllm-env/bin/pip install uv
/opt/vllm-env/bin/uv pip install --python /opt/vllm-env/bin/python "vllm==0.32.0" --torch-backend=auto
python -m chon.serve --model ninhnv/chon-1.0-12b --port 8000    # finds /opt/vllm-env/bin/vllm
```

The server looks for `$CHON_VLLM_BIN`, then `vllm` on the PATH, then `/opt/vllm-env/bin/vllm`, or uses
`$CHON_VLLM_URL` if you run a vLLM server of the base model yourself. vLLM is started on the cached local copy
of the base model, so it works offline. Without vLLM, or with too little free GPU memory, the server uses a
static KV cache instead (slower, no second copy of the weights); the answers are always read by the Chon model.
On a machine that has not run it before, vLLM compiles its kernels at start-up: expect 2-5 minutes (up to 20 on
a slow disk) before `/health` answers.

Measured on one H100 80 GB, one request at a time: Chon-1.0-12B answers in 0.11 s at the median, 4.8 s at the
95th percentile (reasoning written by vLLM, FP8) and 0.6 s on average; with the static cache the 95th
percentile is about 10 s, and with plain transformers generation about 24 s.

### Docker

`docker/Dockerfile.serve` builds an image with both environments (the server and vLLM):

```bash
docker build -f docker/Dockerfile.serve -t chon-serve .
docker run --gpus '"device=0"' -e HF_TOKEN -p 127.0.0.1:8000:8000 chon-serve
```

The host needs an NVIDIA driver for CUDA 13.2 (vLLM's torch build); with an older driver vLLM does not start
and the server falls back to the static cache.

### On a Mac or CPU

`--device` defaults to `auto`: an NVIDIA GPU if there is one, else Apple's GPU, else the CPU.

The E4B and E2B models also run without a data-centre GPU. On a Mac with Apple silicon, use its GPU with
`--device mps`; elsewhere `--device cpu`:

```bash
python -m chon.serve --model ninhnv/chon-1.0-e2b --device mps --port 8000
python -m chon.serve --model ninhnv/chon-1.0-e4b --device mps --no-think --port 8000   # 24 GB or more
python -m chon.serve --model ninhnv/chon-1.0-e2b --device cpu --port 8000
```

Chon-1.0-E2B answers in about 1 s on an 18 GB MacBook Pro (M3 Pro) with `--device mps`; on its CPU alone it
is several times slower. Use `--no-think` for Chon-1.0-E4B outside a data-centre GPU: its extra pass for
unsure questions generates text, which is slow there. Each model card reports measured response times.

## Request and response

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Order #4411 shipped on 3 March; the customer reports it arrived damaged on 9 March.",
  "questions": {
    "route":  {"type": "choice", "instructions": "Route this ticket.",
               "criteria": {"refund": "Refund or replacement", "shipping": "Delivery status", "other": "Anything else"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply today?"},
    "tone":   {"type": "score", "instructions": "How upset is the customer?",
               "criteria": ["calm", "annoyed", "angry"]}
  }
}'
```

Each entry under `answers` has a `type` and:
- **choice:** `choice` and `probabilities` over the option labels
- **yes/no:** `noul` = P(yes), and the same as `probabilities` keyed `true` / `false`
- **score:** `score` and `probabilities` over the levels

When option labels are only ids ("0", "1", ...), send `"option_keys": "hide"` in the question and the model
sees only the descriptions.

`usage` reports `input_tokens` and `output_tokens`. Most decisions are a single forward pass with no
output tokens; some take a short extra pass whose generated tokens are counted in `output_tokens`.

Other endpoints: `GET /health` and `GET /v1/models`.

## Python

```python
from chon.infer import DecisionModel

m = DecisionModel.load("ninhnv/chon-1.0-12b")
answers, usage = m.predict(state, {"route": {"type": "choice", "instructions": "...", "criteria": {...}}})
```

## Decision Index

`chon.dindex` runs a Chon model on the [Decision Index](https://github.com/apolinario/decision-index) kit:

```bash
python -m decision_index pipeline --engine chon.dindex:ChonEngine --option model=ninhnv/chon-1.0-12b-fast \
    --out runs/chon-1.0-12b-fast
```

The engine answers in one forward pass per question (that board's latency rule), refuses a state longer
than 32,768 tokens as unsupported instead of truncating it, and reads choices with more than 16 options in
groups. `python -m chon.dindex shard` / `merge` split the suite so several runner processes can share GPUs.

## Tests

```bash
pip install -e ".[serve,dev]" && pytest
```

## Licence

Apache-2.0. The model weights carry their own licence; see each model card.
