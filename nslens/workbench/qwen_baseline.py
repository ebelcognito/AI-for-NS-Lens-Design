"""Local, frozen Qwen tool baseline; no service credentials or teacher answers.

Run with a Transformers environment and CUDA_VISIBLE_DEVICES set explicitly.
Weights are loaded only from the pinned snapshot supplied by --weights.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re

from .model_baseline import run_episode
from .tools import Workbench, canonical_hash
from nslens.analysis.run_repair_curriculum import digest, write_json

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def chat_messages(payload):
    messages = [dict(role="system", content=payload["instructions"])]
    for item in payload["input"]:
        kind = item.get("type")
        if kind == "function_call":
            function = dict(name=item["name"], arguments=json.loads(item["arguments"]))
            call = dict(id=item["call_id"], type="function", function=function)
            if messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append(dict(role="assistant", content="", tool_calls=[call]))
        elif kind == "function_call_output":
            messages.append(dict(role="tool", tool_call_id=item["call_id"], content=item["output"]))
        else:
            content = item.get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            messages.append(dict(role=item["role"], content=content))
    return messages


def parse_generation(raw, request_id):
    """Parse only complete native calls; retain malformed text without executing it."""
    outputs, errors = [], []
    blocks = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S))
    prose = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.S).strip()
    if prose:
        outputs.append(dict(type="message", role="assistant", content=[dict(type="output_text", text=prose)]))
    for index, match in enumerate(blocks):
        try:
            obj = json.loads(match.group(1), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
                raise ValueError("invalid call keys")
            if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
                raise ValueError("invalid call types")
            outputs.append(dict(type="function_call", call_id=f"qwen_{request_id}_{index}",
                                name=obj["name"], arguments=json.dumps(obj["arguments"], allow_nan=False)))
        except (ValueError, TypeError):
            errors.append("invalid_tool_call")
    if raw.count("<tool_call>") != len(blocks):
        errors.append("incomplete_tool_call")
    # A malformed or mixed generation is not partially executed.
    if errors:
        outputs = [dict(type="message", role="assistant", content=[dict(type="output_text", text=raw)])]
    return outputs, errors


class LocalQwen:
    def __init__(self, weights, adapter=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            weights, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda:0").eval()
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False).eval()
        self.requests = 0

    def prompt(self, payload):
        tools = [dict(type="function", function={k: tool[k] for k in ("name", "description", "parameters")})
                 for tool in payload["tools"]]
        return self.tokenizer.apply_chat_template(chat_messages(payload), tools=tools,
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def input_token_count(self, payload):
        return len(self.tokenizer(self.prompt(payload))["input_ids"])

    def __call__(self, payload):
        inputs = self.tokenizer(self.prompt(payload), return_tensors="pt").to(self.model.device)
        n = inputs.input_ids.shape[1]
        if n + payload["max_output_tokens"] > 32768:
            raise ValueError("native context limit")
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=payload["max_output_tokens"],
                do_sample=True, temperature=.7, top_p=.8, top_k=20,
                pad_token_id=self.tokenizer.eos_token_id)
        ids = generated[0, n:]
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.requests += 1
        outputs, errors = parse_generation(raw, self.requests)
        print(f"  request={self.requests} input={n} output={len(ids)} calls={[o.get('name') for o in outputs if o['type']=='function_call']}", flush=True)
        return dict(output=outputs, usage=dict(input_tokens=n, output_tokens=len(ids)),
                    raw_generation=raw, parse_errors=errors,
                    reached_output_limit=len(ids) == payload["max_output_tokens"])


def replay(result):
    wb = Workbench(result["task"])
    for event in result["events"]:
        assert canonical_hash(wb.state()) == event["before_hash"]
        assert wb.call(event["tool"], event["arguments"]) == event["output"]
        assert canonical_hash(wb.state()) == event["after_hash"]
    assert wb.finalize() == result["outcome"]
    assert wb.records == result["evaluations"]
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.limit < 1:
        parser.error("new output directory and positive limit required")
    tasks_path = args.corpus / "tasks.json"
    tasks = [t for t in json.loads(tasks_path.read_text()) if t["split"] == args.split][:args.limit]
    if not tasks:
        parser.error("no selected tasks")
    args.out.mkdir(parents=True)
    config = dict(model=MODEL, revision=REVISION, weights=str(args.weights.resolve()),
        frozen=True, thinking=False, dtype="bfloat16", seed=args.seed,
        decoding=dict(do_sample=True, temperature=.7, top_p=.8, top_k=20),
        max_requests=16, max_tokens=args.max_tokens, max_output_tokens_per_request=1500,
        adapter=str(args.adapter) if args.adapter else None,
        adapter_files={p.name:digest(p) for p in args.adapter.iterdir() if p.is_file()} if args.adapter else {},
        tasks_sha256=digest(tasks_path), split=args.split, selected_tasks=[t["task_id"] for t in tasks],
        sources={str(p):digest(p) for p in Path("nslens/workbench").glob("*.py")},
        packages={p:importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers", "numpy", "scipy"]})
    write_json(args.out / "config.json", config)
    transport = LocalQwen(args.weights, args.adapter)
    results = []
    for index, task in enumerate(tasks):
        transport.torch.manual_seed(args.seed + index)
        result = run_episode(task, transport, model=MODEL, max_tokens=args.max_tokens)
        if args.adapter:
            result["provenance"] = "lora_language_model"
            result["adapter"] = str(args.adapter)
        write_json(args.out / f"episode{index:03d}.json", result)
        result["replay_passed"] = replay(result)
        write_json(args.out / f"episode{index:03d}.json", result)
        results.append(result)
        write_json(args.out / "summary.json", dict(episodes=len(results),
            successful=sum(r["outcome"]["success"] for r in results),
            submitted=sum(r["outcome"]["submitted"] for r in results),
            all_replays_passed=all(r["replay_passed"] for r in results),
            complete=len(results)==len(tasks),
            outcomes=[dict(task_id=r["task"]["task_id"],status=r["status"],usage=r["usage"],outcome=r["outcome"]) for r in results]))
        print(task["task_id"],result["status"],result["outcome"]["success"],flush=True)
        if result["status"].startswith("transport_error"):
            raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()
