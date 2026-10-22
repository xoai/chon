"""TypeSafe-compatible /v1/systemone server (what the JevBench `typesafe` adapter calls).

    python -m chon.serve --model models/g31b-merged --quantize fp8 --port 8000

Request:  {"state": <text | object | conversation>, "questions": {"<name>": {"type", "instructions", "criteria"}}}
Response: {"model": "...", "answers": {"<name>": {...}}, "usage": {"input_tokens": n, "output_tokens": m}}

Optional bearer auth: set CHON_API_KEY. Requests are served one at a time on the GPU (the
benchmarks measure serial latency; a lock keeps measurements clean).
"""


import argparse
import json
import os
import threading
import time


def create_app(model, api_key: str | None = None):
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
        return {"ok": True, "model": model.name}

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

        def work():
            with lock:
                return model.predict(body["state"], body["questions"])

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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize", default=None, choices=[None, "fp8"])
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward (warm-up takes minutes)")
    ap.add_argument("--think", default=None,
                    help="override the think-when-unsure settings (JSON file); default: the model's chon_config.json")
    ap.add_argument("--no-think", action="store_true", help="ignore a think config in the model's chon_config")
    ap.add_argument("--think-set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one think setting, e.g. engine=hf to skip vLLM (repeatable)")
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
    # warm-up so the first measured request is not a cold start
    model.predict("warm-up", {"w": {"type": "noul", "instructions": "Is this a warm-up?"}})
    uvicorn.run(create_app(model, os.environ.get("CHON_API_KEY")), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
