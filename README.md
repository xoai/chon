# chon

Serving code for **Chon** typed-decision models:

| Model | Base model | GPU memory (bf16) | Notes |
|---|---|---|---|
| [Chon-1.0-12B](https://huggingface.co/ninhnv/chon-1.0-12b) | `google/gemma-4-12B-it` | about 40 GB | most accurate |
| [Chon-1.0-E4B](https://huggingface.co/ninhnv/chon-1.0-e4b) | `google/gemma-4-E4B-it` | 24 GB or more | smaller; also runs on CPU |
| [Chon-1.0-E2B](https://huggingface.co/ninhnv/chon-1.0-e2b) | `google/gemma-4-E2B-it` | 16 GB or more | smallest and fastest; also runs on CPU |

Given a state (text, JSON or a conversation) and one question with a closed set of options, a Chon model
returns a calibrated probability for every option:

- **choice:** 2-16 options
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

### On CPU

The E4B and E2B models also run without a GPU:

```bash
python -m chon.serve --model ninhnv/chon-1.0-e2b --device cpu --port 8000
python -m chon.serve --model ninhnv/chon-1.0-e4b --device cpu --no-think --port 8000
```

Use `--no-think` for Chon-1.0-E4B on CPU: its extra pass for unsure questions generates text, which is
slow without a GPU. Each model card reports CPU response times.

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
- **yes/no:** `noul` = P(yes)
- **score:** `score` and `probabilities` over the levels

`usage` reports `input_tokens` and `output_tokens`. Most decisions are a single forward pass with no
output tokens; some take a short extra pass whose generated tokens are counted in `output_tokens`.

Other endpoints: `GET /health` and `GET /v1/models`.

## Python

```python
from chon.infer import DecisionModel

m = DecisionModel.load("ninhnv/chon-1.0-12b")
answers, usage = m.predict(state, {"route": {"type": "choice", "instructions": "...", "criteria": {...}}})
```

## Tests

```bash
pip install -e ".[serve,dev]" && pytest
```

## Licence

Apache-2.0. The model weights carry their own licence; see each model card.
