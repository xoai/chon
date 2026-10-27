"""TypeSafe-compatible /v1/systemone server (what the JevBench `typesafe` adapter calls).

    python -m chon.serve --model models/g31b-merged --quantize fp8 --port 8000

Request:  {"state": <text | object | conversation>, "questions": {"<name>": {"type", "instructions", "criteria"}}}
Response: {"model": "...", "answers": {"<name>": {...}}, "usage": {"input_tokens": n, "output_tokens": m}}

Optional bearer auth: set CHON_API_KEY. Requests are served one at a time on the GPU (the
benchmarks measure serial latency; a lock keeps measurements clean).

Deadline (--deadline, default 8 s; $CHON_DEADLINE_S; 0 switches it off): counted from the request's arrival,
waiting for the GPU included. The one-pass answer is always computed and kept; the reasoning of an unsure answer
runs only with the time left, shortened or skipped to fit, and dropped if the clock runs out, so a reply never
waits past a client's time limit for a better answer.
"""


import argparse
import json
import os
import threading
import time


def create_app(model, api_key: str | None = None, deadline_s: float = 0.0):
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse

    from .schema import CapacityError, ItemError

    app = FastAPI(title="chon decision server")
    lock = threading.Lock()

    def check_auth(authorization: str | None):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    def health():
        return {"ok": True, "model": model.name, "deadline_s": deadline_s or None,
                "deadline": dict(getattr(model, "deadline_stats", {}) or {})}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model.name, "object": "model"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: str | None = Header(default=None)):
        check_auth(authorization)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        if not isinstance(body, dict) or "state" not in body or "questions" not in body:
            return JSONResponse({"error": "body needs 'state' and 'questions'"}, status_code=400)
        t0 = time.perf_counter()
        due = time.monotonic() + deadline_s if deadline_s else None   # from arrival: queueing counts too

        def work():
            with lock:
                if due is None:
                    return model.predict(body["state"], body["questions"])
                return model.predict(body["state"], body["questions"], deadline=due)

        try:
            import anyio

            answers, usage = await anyio.to_thread.run_sync(work)
        except CapacityError as e:  # too long to take whole: 413 with the standard "maximum context length" marker
            return JSONResponse({"error": str(e)}, status_code=413)
        except ItemError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        usage["latency_s"] = round(time.perf_counter() - t0, 4)
        return {"model": model.name, "answers": answers, "usage": usage}

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="auto", help="auto (NVIDIA GPU, else Apple GPU, else CPU), cuda, mps or cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize", default=None, choices=[None, "fp8"])
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward (warm-up takes minutes)")
    ap.add_argument("--think", default=None,
                    help="override the think-when-unsure settings (JSON file); default: the model's chon_config.json")
    ap.add_argument("--no-think", action="store_true", help="ignore a think config in the model's chon_config")
    ap.add_argument("--think-set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one think setting, e.g. engine=hf to skip vLLM (repeatable)")
    ap.add_argument("--deadline", type=float, default=float(os.environ.get("CHON_DEADLINE_S", "8")),
                    help="seconds from a request's arrival to its answer: the reasoning is shortened or skipped to "
                         "fit, the one-pass answer always stands (0 = no deadline; default 8, $CHON_DEADLINE_S)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    import uvicorn

    from .infer import DecisionModel

    think = False if args.no_think else (json.load(open(args.think)) if args.think else None)
    model = DecisionModel.load(args.model, adapter=args.adapter, device=args.device, dtype=args.dtype,
                               quantize=args.quantize, compile=args.compile, think=think)
    for kv in args.think_set:
        k, _, v = kv.partition("=")
        if model.think is not None:
            model.think[k] = json.loads(v) if v[:1] in "[{0123456789" or v in ("true", "false", "null") else v
    print(f"[serve] {model.name} on {model.device}"
          + (f", reasoning engine {model.think.get('engine', 'hf')}" if model.think else ", no reasoning"), flush=True)
    # warm-up so the first measured request is not a cold start; the reasoning path too (its first call compiles,
    # and its speed sets how many tokens a deadline allows)
    model.predict("warm-up", {"w": {"type": "noul", "instructions": "Is this a warm-up?"}})
    if model.think:
        warm = model.encode("Warm-up record: the order total is 120 and the limit is 100.",
                            {"type": "noul", "instructions": "Is the order within the limit?"}, ["yes", "no"])
        for _ in range(2):   # the first call compiles: only the second, warm one sets the speed estimate
            model._gen_spt = None
            try:
                model.think_logprobs([warm], [2], budget=48)
            except Exception as e:  # noqa: BLE001
                print(f"[serve] reasoning warm-up failed: {e!r:.200}", flush=True)
                break
        spt = getattr(model, "_gen_spt", None)
        print(f"[serve] reasoning speed {1 / spt:.0f} tokens/s" if spt else "[serve] reasoning speed not measured",
              flush=True)
    if args.deadline:
        print(f"[serve] deadline {args.deadline:g} s per request (one-pass answers always kept)", flush=True)
    uvicorn.run(create_app(model, os.environ.get("CHON_API_KEY"), args.deadline), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
