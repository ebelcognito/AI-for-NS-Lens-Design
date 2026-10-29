"""Frozen LLM baseline through the Responses function-calling protocol.

No default model is silently selected. Configure OPTICS_MODEL and OPENAI_API_KEY,
and optionally OPTICS_BASE_URL (default https://api.openai.com/v1). Credentials
are read from environment variables and never saved in experiment artifacts.

Official protocol: https://developers.openai.com/api/docs/guides/function-calling
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from nslens.analysis.run_repair_curriculum import digest, write_json
from .episodes import INSTRUCTIONS
from .tools import SCHEMA, TOOL_SCHEMAS, Workbench


class ResponsesTransport:
    def __init__(self, model, *, base_url=None, api_key=None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPTICS_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not model or not self.api_key:
            raise ValueError("configure a model and OPENAI_API_KEY; credentials must not be put in task files")

    def __call__(self, payload):
        body = json.dumps(dict(model=self.model, store=False, **payload), allow_nan=False).encode()
        request = Request(self.base_url+"/responses", data=body, method="POST", headers={
            "Authorization": "Bearer "+self.api_key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as exc:
            # Some gateways echo request metadata in error bodies. Do not log it.
            raise RuntimeError(f"model endpoint returned HTTP {exc.code}") from None


def run_episode(task, transport, *, model, max_requests=16, max_tokens=24000):
    wb = Workbench(task)
    conversation = [dict(role="user", content=json.dumps(task, allow_nan=False))]
    requests, response_log, usage = 0, [], dict(input_tokens=0, output_tokens=0)
    status = "request_limit"
    start = time.monotonic()
    while not wb.done and wb.actions < wb.max_actions and requests < max_requests:
        remaining = max_tokens - sum(usage.values())
        if remaining <= 0:
            status = "token_limit"
            break
        payload = dict(instructions=INSTRUCTIONS, input=conversation, tools=TOOL_SCHEMAS,
                       parallel_tool_calls=False, max_output_tokens=min(1500, remaining))
        # Local transports can count the fully templated input exactly. Reserve
        # that cost before decoding so the cumulative token limit is hard.
        if hasattr(transport, "input_token_count"):
            input_cost = transport.input_token_count(payload)
            if input_cost >= remaining:
                status = "token_limit"
                break
            payload["max_output_tokens"] = min(payload["max_output_tokens"], remaining-input_cost)
        try:
            response = transport(payload)
        except Exception as exc:
            # Never serialize exception text from arbitrary transports: it could
            # contain secrets. HTTP status is already sanitized by our transport.
            status = "transport_error:"+type(exc).__name__
            break
        requests += 1
        response_log.append(response)
        for key in usage:
            usage[key] += int(response.get("usage", {}).get(key, 0))
        outputs = response.get("output", [])
        conversation.extend(outputs)
        calls = [item for item in outputs if item.get("type") == "function_call"]
        if not calls:
            status = "stopped_without_submission"
            break
        for call in calls:
            try:
                arguments = json.loads(call["arguments"])
            except (TypeError, ValueError, KeyError):
                arguments = None
            result = wb.call(call.get("name", ""), arguments)
            conversation.append(dict(type="function_call_output", call_id=call["call_id"],
                                     output=json.dumps(result, allow_nan=False)))
        if wb.done:
            status = "submitted"
    outcome = wb.finalize()
    return dict(schema_version=SCHEMA, provenance="frozen_language_model", model=model,
                task=task, status=status, model_requests=requests, usage=usage,
                elapsed_seconds=time.monotonic()-start, response_log=response_log,
                conversation=conversation, events=wb.events, evaluations=wb.records,
                outcome=outcome)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPTICS_MODEL"))
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.model or not os.environ.get("OPENAI_API_KEY"):
        parser.error("frozen baseline not run: set OPTICS_MODEL/--model and OPENAI_API_KEY")
    if args.limit < 1 or args.out.exists():
        parser.error("positive limit and a new output directory required")
    tasks_path = args.corpus/"tasks.json"
    tasks = [t for t in json.loads(tasks_path.read_text()) if t["split"] == "test"][:args.limit]
    if not tasks:
        parser.error("no held-out test tasks")
    transport = ResponsesTransport(args.model)
    args.out.mkdir(parents=True)
    results = []
    for task in tasks:
        result = run_episode(task, transport, model=args.model)
        write_json(args.out/f"episode{len(results):03d}.json", result)
        results.append(result)
        print(task["task_id"], result["status"], result["outcome"]["success"], flush=True)
        if result["status"].startswith("transport_error"):
            break
    write_json(args.out/"manifest.json", dict(model=args.model, frozen=True,
        source_sha256=digest(__file__), tasks_sha256=digest(tasks_path), episodes=len(results),
        submitted=sum(r["outcome"]["submitted"] for r in results),
        successful=sum(r["outcome"]["success"] for r in results),
        usage={key:sum(r["usage"][key] for r in results) for key in ("input_tokens","output_tokens")}))


if __name__ == "__main__":
    main()
