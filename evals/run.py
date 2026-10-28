"""Run the eval tasks against the REAL model and the REAL Docker sandbox.

    python -m evals.run                       # all tasks, strict pipeline mode
    python -m evals.run --only fizzbuzz,fix_bug
    python -m evals.run --mode auto           # the chat-first mode
    python -m evals.run --list

Each task runs in a throw-away workspace with every tool call auto-approved (the sandbox and the
workspace confinement are the safety net, exactly as in normal use). Results (pass/fail, status,
tokens, time) are printed and saved as JSON under evals/results/, so you can compare runs before and
after changing a prompt, a model or the pipeline. Costs real tokens: start with --only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from evals.tasks import BY_ID, TASKS, Task, _tree

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_task(task: Task, mode: str, workspaces_root: Path) -> dict:
    # Imported late: agent.tools reads the environment (workspace settings) when imported.
    from agent import graph as graph_module
    from agent import history, sandbox, tools

    thread_id = f"eval-{task.id}-{uuid.uuid4().hex[:6]}"
    workspace = tools.workspace_for(thread_id)
    tools.ensure_workspace(workspace)
    for name, content in task.files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    before = _tree(workspace)

    started = time.time()
    error = ""
    graph = graph_module.build_graph(mode)
    config = graph_module.make_config(thread_id, mode)
    steps = 0
    try:
        events = graph_module.drive(graph, config, task.prompt)
        while True:
            for _ in events:
                steps += 1
            if not graph_module.pending_tool_calls(graph, config):
                break
            events = graph_module.drive(graph, config, None)   # approve everything pending
    except Exception as exc:  # API errors, recursion limit, ...
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    status = graph_module.get_run_status(graph, config)
    with tools.use_workspace(workspace):
        if task.expects_changes:
            ok, detail = task.check(workspace)
        else:
            ok = _tree(workspace) == before
            detail = "workspace unchanged" if ok else "the agent modified files it should not touch"
    if error:
        ok, detail = False, f"{error} | {detail}"
    usage = status.get("usage") or {}
    result = {"id": task.id, "passed": bool(ok), "detail": detail, "build_status": status.get("status"),
              "tokens": usage.get("total", 0), "model_calls": usage.get("calls", 0),
              "seconds": round(elapsed, 1), "events": steps, "mode": mode,
              "docker": sandbox.docker_available()}
    history.delete_conversation(thread_id)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval tasks (real model, real Docker)")
    parser.add_argument("--mode", choices=("pipeline", "auto", "react"), default="pipeline")
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    parser.add_argument("--budget", type=int, default=None,
                        help="RUN_TOKEN_BUDGET per task (default: the normal default)")
    parser.add_argument("--keep", action="store_true", help="keep the workspaces for inspection")
    args = parser.parse_args(argv)

    if args.list:
        for task in TASKS:
            print(f"{task.id:18} {task.description}")
        return 0
    selected = TASKS
    if args.only:
        wanted = [part.strip() for part in args.only.split(",") if part.strip()]
        unknown = [w for w in wanted if w not in BY_ID]
        if unknown:
            print(f"Unknown task(s): {', '.join(unknown)}. Use --list.")
            return 2
        selected = [BY_ID[w] for w in wanted]

    tmp = tempfile.mkdtemp(prefix="agent-evals-")
    root = Path(tmp)
    # Everything the evals touch lives in a temporary directory, never in your real workspaces.
    os.environ["WORKSPACE_MODE"] = "session"
    os.environ["WORKSPACES_DIR"] = str(root / "workspaces")
    os.environ["CONVERSATION_DB"] = "memory"
    os.environ.setdefault("RECURSION_LIMIT", "120")
    if args.budget is not None:
        os.environ["RUN_TOKEN_BUDGET"] = str(args.budget)

    from agent import sandbox
    if not sandbox.docker_available() and not sandbox.unsafe_local_allowed():
        print("Docker is not available, so the eval tasks cannot run code. Start Docker (and build "
              "Dockerfile.sandbox), or set ALLOW_UNSAFE_LOCAL_EXECUTION=true for a throw-away machine.")
        return 2

    from agent.llm import describe_llm
    print(f"model: {describe_llm()}   mode: {args.mode}   tasks: {len(selected)}\n")
    results = []
    for task in selected:
        print(f"- {task.id} ...", end=" ", flush=True)
        outcome = run_task(task, args.mode, root)
        results.append(outcome)
        print(("PASS" if outcome["passed"] else "FAIL")
              + f"  ({outcome['tokens']:,} tokens, {outcome['model_calls']} calls, {outcome['seconds']}s)")
        if not outcome["passed"]:
            print("    " + outcome["detail"].replace("\n", "\n    "))

    passed = sum(r["passed"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    print(f"\n{passed}/{len(results)} passed, {total_tokens:,} tokens in total")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / time.strftime("%Y%m%d_%H%M%S.json")
    out.write_text(json.dumps({"model": describe_llm(), "mode": args.mode, "results": results},
                              indent=2), encoding="utf-8")
    print(f"saved {out}")
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
