# chon

Serving code for **Chon** typed-decision models:

| Model | Base model | GPU memory (bf16) | Notes |
|---|---|---|---|
| [Chon-1.1-31B](https://huggingface.co/ninhnv/chon-1.1-31b) | `google/gemma-4-31B-it` | 80 GB or more (about 62 GB of weights) | newest; always one forward pass; Decision Index public 65.47 |
| [Chon-1.0-12B](https://huggingface.co/ninhnv/chon-1.0-12b) | `google/gemma-4-12B-it` | about 40 GB (+ ~30 GB for vLLM) | most accurate; thinks briefly on unsure questions |
| [Chon-1.0-12B-Fast](https://huggingface.co/ninhnv/chon-1.0-12b-fast) | `google/gemma-4-12B-it` | about 40 GB | the same model, always one forward pass |
| [Chon-1.0-E4B](https://huggingface.co/ninhnv/chon-1.0-e4b) | `google/gemma-4-E4B-it` | 24 GB or more | smaller; Macs with 24 GB or more |
| [Chon-1.0-E2B](https://huggingface.co/ninhnv/chon-1.0-e2b) | `google/gemma-4-E2B-it` | 16 GB or more | smallest and fastest; about 1 s per decision on an 18 GB MacBook Pro |

Given a state (text, JSON or a conversation) and one or more questions, each with a closed set of options, a
Chon model returns a calibrated probability for every option:

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
H100 80 GB; Chon-1.1-31B also on one RTX PRO 6000 Blackwell (96 GB). Access to the model's Gemma 4 base model on
Hugging Face is required.

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

Chon-1.1-31B needs one GPU with 80 GB or more and serving code 1.5.3 or newer (1.6.0 for the commit band):

```bash
python -m chon.serve --model ninhnv/chon-1.1-31b --port 8000
python -m chon.serve --model ninhnv/chon-1.1-31b --port 8000 --noul-commit 0.199,0.801   # as served for JevBench
```

On the public JevBench files with the commit band (one H100 80 GB, one request at a time): hard 88.3%, original
and easy 100%; median reply 0.11 s, slowest 1.87 s. The [model card](https://huggingface.co/ninhnv/chon-1.1-31b)
has the full results.

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
- `--reread BELOW[,LAYOUT[,MAX_TOKENS[,COMBINE]]]` (`CHON_REREAD`): a second read of unsure answers, e.g.
  `--reread 0.9` or `--reread 0.9,twice,2000,replace`; `--reread off` turns off a model's own setting. See
  [Prompt layout and second read](#prompt-layout-and-second-read).
- `--prefix-cache on|off`: the shared-prefix read (default: the model's `chon_config.json` `"prefix_cache"`, else
  on). See [Shared-prefix read](#shared-prefix-read).
- `--noul-commit LO,HI|off` (since 1.6.0): the yes/no commit band, overriding the model's `chon_config.json`
  `"noul_commit"` (unset keeps it). See [Yes/no commit band](#yesno-commit-band).

### Yes/no commit band

JevBench counts a yes/no answer whose P(yes) is strictly between 0.2 and 0.8 as an abstention, which always scores as
wrong. With the commit band on, such an answer is served just outside the band instead, at `LO` or `HI`, whichever is
nearer (`--noul-commit 0.199,0.801`; needs 0 < LO < 0.5 < HI < 1). A JevBench deployment should serve with the band. The
Decision Index has no such rule, so the release package ships with the band off; `--noul-commit off` also turns off a
band a model's config sets. `/health` (`"noul_commit"`) and the server's start line report it, and the Decision Index
engine takes the same setting as `--option noul_commit=LO,HI` (`true` keeps the model's own band; off by default).

### Prompt layout and second read

Since 1.3.0 the server reads two settings from a model's `chon_config.json`. **Version 1.2.0 and older ignore
both** (they read every model state-first, in one read), so serve a model that sets them with 1.3.0 or newer.

**`prompt_layout`** is the prompt layout the model was trained with:

- `"state_first"` (the default): the state, then `---`, then the question and its lettered options.
- `"header"`: the request's questions listed first, then the state, then the question being read:

  ```
  Questions that will be asked about this record:
  - <instructions of question 1>
  - <instructions of question 2>
  ---
  <state>

  ---
  Question: <instructions of the question being read>
  A) ...
  Answer with one letter.
  ```

  The list holds each distinct `instructions` of the request once, in request order (two questions with the same
  instructions and different options are listed once), with runs of whitespace collapsed to one space; every read
  of the request lists the same questions, including the group and final rounds of a choice with more than 16
  options.
- `"hybrid_header"`: the header for a request with two or more distinct questions, state-first for a request with
  one (counted by distinct instructions).

Any other value is refused when the model loads, rather than read in the wrong layout.

**`reread`** turns on a second read of unsure answers: a choice or yes/no question whose top calibrated probability
is under `below`, in a prompt of at most `max_tokens` tokens, is read once more and its two distributions are
combined. Choices with more than 16 options and score questions are read once.

```json
"reread": {"below": 0.9, "layout": "twice", "max_tokens": 2000, "combine": "avg"}
```

- `layout`: `"twice"` writes the whole request out twice (`... Read the request again: ...`) and reads the letter
  after the second copy; a header read is written out twice with its question list. `"sandwich"` is the question,
  the state, then the question again. `"reverse"` (since 1.5.0, below) is the first read's prompt with the options
  in reverse order, and `"rotate"` (since 1.6.0, below) the same with the options rotated by half.
- `combine`: `"avg"` averages the two distributions; `"replace"` keeps the second read.
- `CHON_REREAD` (or `--reread`) overrides the config: `below[,layout[,max_tokens[,combine]]]`, an empty field keeps
  its value (`0.9,twice,,replace`), and `0` or `off` turns the second read off.
- The second read is skipped when the request's deadline (`--deadline`) leaves too little time for it; the first
  read's answer stands. Its prompt tokens are counted in `usage.input_tokens`, and `usage.reread` says how many
  questions were read twice. `/health` reports the layout, the setting and counts.
- The letters are always read by the loaded model (transformers), so the second read works with every
  reasoning engine, vLLM included.

### Prompt style and uncapped letters

Since 1.4.0 the server reads two more settings from a model's `chon_config.json`. **Version 1.3.0 and older ignore
both** (they read every model in the chon style, under the model's soft cap), so serve a model that sets them with
1.4.0 or newer.

**`prompt_style`** is the prompt style the model was trained with: `"chon"` (the default, above) or `"decider"`, the
chat layout of decider-ai 1.6.0 as FlexDM v1.0 reads Gemma 4:

```
Questions that will be asked about the context:      (header layouts only)
- <instructions of question 1>
- <instructions of question 2>

Context:
<state>

Question: <instructions of the question being read>
Options:
(A) <label>: <description>
(B) ...
```

and the letter is read after `Answer: (` at the start of the reply (a config's `answer_prefix` is ignored). JSON
states are written with spaces, and each element of an array of 8 or more carries its position (`"_index"`); yes/no
questions show `(A) no`, `(B) yes`; a yes/no question without instructions reads "Which answer fits the context?".
The header lists every question of the request in request order, as written (two questions with the same
instructions are listed twice, so `"hybrid_header"` counts them as two). The `"twice"` second read writes the whole
request out twice with "Read the same input again before answering:" between the copies (the header included for a
header read); `"sandwich"` has no decider form and is refused at load. Any other style is refused at load.

**`"uncap_logits": true`** reads the letters with the model's final logit soft cap turned off (Gemma's
`final_logit_softcapping`, `30 * tanh(x / 30)`), as such a model was trained: the top answer is the same, the
probabilities are not. `/health` and the Decision Index provenance report both settings; a config without them
reads exactly as in 1.3.0.

### Reverse-order second read

Since 1.5.0 the second read can also be `"layout": "reverse"`: the unsure question is read again in the first read's
prompt (its layout, question list and state, written once) with its options in reverse order, so the option shown
first becomes the last one and the letters move with them. The second read's probabilities are mapped back to the
options and combined as set (`"avg"` or `"replace"`), which averages out a preference for a position. A yes/no
question is read with its two answers swapped (`(A) yes`, `(B) no` in the decider style).

```json
"reread": {"below": 1.01, "layout": "reverse", "max_tokens": 3000, "combine": "avg"}
```

or `CHON_REREAD="1.01,reverse,3000"` (`--reread 1.01,reverse,3000`). Everything else about the second read applies
(the threshold, the prompt length limit, choices over 16 options and scores read once, skipped near the deadline),
and it reads through the shared-prefix read like the first read, as its prompt starts with the same tokens. The
decider style takes it (unlike `"sandwich"`). **Version 1.4.0 and older refuse `"reverse"`** at load as an unknown
layout.

### Pair read

Since 1.5.0 a model's `chon_config.json` can ask for a **pair read**: every request is read twice in the same prompt,
once with the model's LoRA adapter and once with the adapter switched off (the plain base model), and the two
distributions are averaged: a trained adapter and the plain model it was trained on can each do best on different
kinds of questions, and the pair read combines them.

```json
"pair": {"weight": 0.5, "temperatures": {"choice": 1.0, "noul": 1.0, "score": 1.0}}
```

- `weight`: the adapter read's share; the answer is `weight x adapter + (1 - weight) x base`, label by label
  (default 0.5; `1` is a plain single read).
- `temperatures`: the base read's own per-type temperatures (the adapter read keeps the config's `temperatures`);
  missing types are 1.0.
- Each read is calibrated with its own temperatures and gets its own second read if one is set (each decides by its
  own probabilities which questions are unsure); choices over 16 options are read in groups in both. The final rules,
  post-calibration (`post_temperatures`) and the yes/no commit band (`noul_commit`), run **once, on the average**,
  never on either read.
- The adapter must stay unmerged, so the model is loaded with it as a separate adapter (a release package with
  `adapter_config.json` and `base_model`, or `--adapter`); a pair read of a merged model or a model without an adapter
  is refused at load, and so is a pair read with `"think"`.
- Cost: two reads per request (each through the shared-prefix read when it is on). With a deadline the adapter read
  always comes first, and the base read runs only if the time left allows it; `usage.pair` says how many reads were
  averaged (2, or 1 when only the adapter read fitted), and `usage.input_tokens` counts both. `/health`, the
  server's start line and the Decision Index provenance report the setting (`"pair": null` when there is none).

**Version 1.4.0 and older ignore `"pair"`** (they read the adapter alone, merged), so serve a model that sets it with
1.5.0 or newer. A config without `"pair"` or `"reverse"` reads exactly as in 1.4.0.

### Long prompts on a 96 GB card (1.5.1 to 1.5.3)

A 31B's weights take about 60 GB of the board's 96 GB RTX PRO 6000. In 1.5.0 a long prompt (tens of thousands of
tokens) went through the model in one pass, and that pass's activations needed more than the 35 GB left: the longest
Decision Index requests stopped with "CUDA out of memory", and the kit stops a run at its first error. A 141 GB H200
has the room, which is why it did not show there.

- **Prefill in chunks (1.5.2).** A prompt longer than 4,096 tokens fills the key/value cache 4,096 tokens at a time,
  as the shared-prefix read already does for a request's questions; the question's own tokens are then read against
  the cache. Memory then grows with the cache, not with the prompt's activations. `CHON_PREFILL_CHUNK` sets the
  chunk (0 = one pass, as before).
- **The widest heads in query chunks (1.5.1).** Gemma 4's full-attention heads are 512 wide, too wide for PyTorch's
  fused attention kernels, so their attention holds every head's whole score matrix. When that would pass 1 GB the
  queries go through in chunks against every key; narrower heads stay on the fused kernels (1.5.1 chunked them too,
  which slowed long requests; 1.5.2 does not). Since 1.5.3 the budget is a quarter of the GPU's free memory, between 1
  and 8 GB (1.5.2's fixed 1 GB cut the attention of a 70,000-token prompt, 20-30 questions about one short state,
  into hundreds of small calls: 3-5 minutes per request on the card). `CHON_ATTN_CHUNK_MB` fixes the budget (0 turns
  it off).
- **Guards.** A Decision Index request that still runs out of GPU memory is read once more with a quarter of every
  budget (`usage.oom_retry` = 1); if it still does not fit it is reported as unsupported, so the run goes on.
  `PYTORCH_CUDA_ALLOC_CONF` defaults to `expandable_segments:True`.

Answers are the same up to floating-point rounding (1e-7 in float32 on the tiny test model with tiny chunks, bf16
rounding on a GPU); prompts under 4,096 tokens read exactly as in 1.5.0. Everything else is as in 1.5.0.

### Shared-prefix read

Since 1.3.0 the questions of one request share one read of their state. Every read of a request starts with the same
tokens (the chat template's start, a header model's question list, the state); that part runs through the model
once, and each question then reads only its own tokens (question, options, reply position) against the cached keys
and values, all questions in one packed forward pass, each masked to the prefix and its own tokens. Each question
sees exactly the tokens it sees in a separate pass, so the answers are the same up to floating-point rounding (bf16
rounding on a GPU). A request with k questions about a P-token state costs about P + k x (question tokens) instead
of k x P.

It is on by default and used for a request's first reads, for second reads that start with the same tokens
(`twice`, and `twice_header` for a header model) and for the group and final rounds of a choice with more than 16
options. A request with one question (and no such choice), a shared part under `prefix_min_tokens`, `sandwich`
second reads and the reads after reasoning go through the plain path (one forward pass per read). It needs a
decoder whose layers are sliding-window or full attention (Gemma 4) with the `sdpa` (default) or `eager` attention;
for any other model the server prints one line when it loads and reads every question plainly.

```json
"prefix_cache": true, "prefix_min_tokens": 128, "prefix_pack_tokens": 8192
```

- `prefix_cache`: `false` (or `--prefix-cache off`, or the Decision Index engine's `--option prefix_cache=false`)
  turns it off. `usage.prefix_tokens` says how many tokens a request shared; `/health` reports whether it is on and
  how often it was used.
- `prefix_min_tokens`: a shorter shared part is not worth a separate pass.
- `prefix_pack_tokens`: the most question tokens read in one packed pass (a longer question goes alone). For a prefix
  over about 123,000 tokens the pack is made smaller so that its attention mask stays under 2^30 cells.

**Memory, Gemma 4 31B in bf16 on one 96 GB card, a 50,000-token state with 20 questions.** The weights take about
62 GB. The 31B has 50 sliding-window layers (16 key/value heads of 256, window 1,024) and 10 full-attention layers
(4 key/value heads of 512; keys and values come from one projection but are cached as two tensors):

| Part | Size |
|---|---|
| Kept keys/values, sliding layers: 50 x 1,023 positions x 16 KiB | 0.8 GB |
| Kept keys/values, full layers: 10 x 50,000 positions x 8 KiB | 4.1 GB |
| Prefix pass (once): the activations of one plain read of the prompt, the same as the plain path pays | about 12 GB |
| Each packed pass of up to 8,192 question tokens: full-layer mask 8,192 x 58,192 (0.5 GB, about 1 GB as an attention bias), one full layer's prefix + question keys/values (0.5 GB), activations (about 1 GB) | about 3 GB |

So the peak is about 80 GB (during the prefix pass), and about 70 GB while the questions are read: 20 questions of
a few hundred tokens each fit one packed pass. A sliding layer keeps a copy of its last 1,023 prefix positions: kept
as a view of the whole pass's keys and values (as transformers' cache does), the sliding layers would pin
50 x 50,000 x 16 KiB = 41 GB and the request would not fit. These are computed sizes, not a measurement on that card;
the bound assumes the `sdpa` attention (`eager` materialises the attention scores and needs far more for long
states, in the plain path too).

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

The engine answers each question in one read without reasoning (that board's latency rule: median, mean and 80th
percentile under 1 s per request), refuses a state longer than 32,768 tokens as unsupported instead of truncating
it, and reads choices with more than 16 options in groups. The shared-prefix read is on unless the model's config
or `--option prefix_cache=false` turns it off; with it, a request's state is read once for all its questions. The
model's prompt layout and style, uncapped letters, temperatures per category, second-read, pair-read and
shared-prefix settings (above) apply, and the run's provenance states them. The yes/no commit band is off unless
`--option noul_commit=LO,HI` (or `true`, the model's own band) turns it on.
`python -m chon.dindex shard` / `merge` split the suite so several runner processes can share GPUs.

Chon-1.1-31B scores **65.47** on the public 0.3 suite with this engine (our run with 1.5.3: the full suite in 4
parts on 4 RTX PRO 6000 cards; median 154 ms, mean 725 ms, 80th percentile 585 ms per request). The model card
has the commands to reproduce it.

## Tests

```bash
pip install -e ".[serve,dev]" && pytest
```

`tests/test_parity.py` builds a tiny random Gemma 4 model on the CPU and checks that every read's token ids and
every output probability equal golden values made with the training code (`tests/golden_parity.json`, written by
`tests/make_golden.py`). `tests/test_prefix.py` checks the shared-prefix read: the same reads and probabilities
as the training code's shared-prefix read (`tests/golden_prefix.json`, `make_golden.py --prefix`), and the plain
path's probabilities within 1e-4, for every parity case and for states longer than the sliding window. The
cases cover both prompt styles, uncapped letters on the tiny model given a soft cap, the reverse and rotated second
reads, temperatures per category (every level, bin and state kind), and the pair read on the tiny model with a tiny
LoRA adapter (`tiny_model.tiny_adapter`, its weights seeded like the model's).

## Licence

Apache-2.0. The model weights carry their own licence; see each model card.
