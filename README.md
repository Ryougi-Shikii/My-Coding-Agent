# LangGraph Coding Agent

An autonomous coding agent built with **LangGraph**, available as a **Streamlit web app** (`app.py`)
and as a **command-line tool** (`cli.py`). It can create, read, edit, list and search files, check
syntax, run Python and tests in a locked-down **Docker sandbox**, install packages (by building a
new sandbox image), use Git, keep a plan / todo list / persistent memory, and review its own changes.

It works with several LLM providers (OpenAI, Anthropic, Google Gemini, xAI/Grok) through
`init_chat_model`, chosen with environment variables.

The agent only ever works inside the `workspace/` directory.

## Two modes

| | `pipeline` (default) | `react` |
|---|---|---|
| Structure | Four LangGraph nodes: **planner → coder → reviewer → committer** | One `create_react_agent` loop |
| Order of work | Enforced by the graph | Encouraged by the system prompt |
| Review before commit | Enforced (cannot commit without an `APPROVE` verdict) | Prompt only |

Choose with `AGENT_MODE` in `.env`, `python cli.py --mode react`, or the mode selector in the
Streamlit sidebar.

### Approval: only dangerous tools need a human

| Needs your approval | Runs automatically |
|---|---|
| `write_file`, `edit_file`, `create_directory`, `run_python_file`, `run_tests`, `install_packages`, `git_commit` | `list_files`, `read_file`, `search_text`, `check_python_syntax`, `git_init`, `git_status`, `git_add`, `git_diff`, `git_log`, `read_agent_state`, `update_plan`, `update_todo`, `save_memory`, `review_changes` |

* Nothing dangerous runs before you click **Approve** (web) or answer `y` (CLI). **Reject** tells
  the agent its call was refused (it receives a `ToolMessage` saying so) and it must change approach.
* Unknown tools always need approval. Set `APPROVAL_MODE=all` to be asked about every tool call.
* If one batch of tool calls contains both safe and dangerous tools, you approve or reject the batch as a whole.
* Mechanics: in `pipeline` mode a conditional edge routes only dangerous batches through the
  single `approval` node, which is the graph's only interrupt. In `react` mode the graph still pauses
  before every `tools` step (`interrupt_before=["tools"]`) and the shared driver resumes safe batches by itself.

### What the pipeline enforces (in code, not in the prompt)

* **Per-stage tools.** Each node only gets its own tools, and the shared tools node checks again, so a call the model
  should not make is answered with an error instead of running:
  * planner: read-only tools plus `update_plan` / `update_todo`
  * coder: the only stage that can write, edit, run code or install packages; cannot review or commit
  * reviewer: read-only (no write/edit/install tools), plus `review_changes`
  * committer: the only stage that can `git_add` / `git_commit`
* **Gates between stages**, decided from tool results, never from what the model says:
  * planner → coder only after `update_plan` **and** `update_todo` succeeded
  * coder → reviewer only after a `write_file` / `edit_file` succeeded (otherwise the run ends: nothing to review)
  * reviewer → committer only if the `review_changes` tool output contains a whole-line `VERDICT: APPROVE`.
    `REQUEST_CHANGES`, a hedged verdict such as `APPROVE_WITH_NITS`, or a missing verdict sends the findings back
    to the coder, up to `MAX_REVIEW_ROUNDS`
  * the run is *committed* only after `git_commit` succeeded
* **Only the reviewed tree can be committed.** `review_changes` fingerprints every non-ignored file it reviews
  (`REVIEWED-TREE: <sha256>`). If anything changes afterwards (for example `run_tests` writes a file), the approval is void
  and the reviewer must review again; `git_add` / `git_commit` are refused in the committer stage unless the tree still matches.
* **The whole diff is reviewed.** Large diffs are split into parts and every part is reviewed; one failing part fails the review,
  and a change too large to review is never approved.
* A stage that stops without completing its required step is reminded up to 3 times, then the run stops with a clear message.

What is **not** enforced by the graph: that the coder actually ran tests, and the quality of the review itself
(it is an LLM judgement, so treat it as a second opinion).

## Setup

Requires Python 3.10+ and Docker.

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env
```

### Configure `.env`

```
LLM_PROVIDER=google_genai        # openai | anthropic | google_genai | xai
MODEL_NAME=gemini-2.0-flash      # optional: each provider has a default
GOOGLE_API_KEY=...               # the key for the provider you chose
```

For OpenAI / Anthropic set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. For xAI install the optional package
(`pip install langchain-xai`, listed as a comment in `requirements.txt`) and set `XAI_API_KEY`.
Model names change often, so set `MODEL_NAME` explicitly if the default is not available to you.
Never commit your `.env` (it is git-ignored). Other settings (timeouts, sandbox image, approval mode, ...) are
documented in `.env.example`.

### Pull the Docker sandbox image

```bash
docker pull python:3.12-slim
```

### Run

```bash
streamlit run app.py        # web app
python cli.py               # terminal version
python cli.py --mode react  # classic single-agent mode
```

The web sidebar shows whether Docker is available, the current sandbox image, the pipeline stage, and the top-level
files in the workspace. **New conversation** starts a fresh LangGraph thread.
In the CLI, type `/help` for commands (`/state`, `/status`, `/new`, `/exit`).

## Safety model

* **Workspace only.** Every path is resolved (symlinks included) and must stay inside `WORKSPACE_DIR`
  (default `./workspace`, relative to the project root). `.git/` and `.agent/sandbox_image.txt` cannot be written by file tools.
* **Docker-only execution by default.** Code runs with `--network none`, 512 MB RAM, 1 CPU, 256 processes, all capabilities dropped,
  `no-new-privileges`, read-only root filesystem, a 128 MB `/tmp`, a non-root user, and a timeout
  (a timed-out container is removed). `.git` and `.agent` are mounted read-only inside the sandbox.
  If you run as root, the container uses user `65534`; set `SANDBOX_USER=uid:gid` to change it.
  If Docker is not available, nothing is executed. `ALLOW_UNSAFE_LOCAL_EXECUTION=true` allows running directly on your machine
  (**not sandboxed**, minimal environment so API keys are not passed on); only use it if you accept that risk.
* **Package installation builds a new sandbox image.** `install_packages` validates each requirement (plain names, optional extras and version
  specifiers only), builds a new image `FROM` the current sandbox image in a temporary build context (`Dockerfile`, `packages.json`, `install.py`; package
  strings never go through a shell), then records the new tag in `workspace/.agent/sandbox_image.txt`. All later runs use that image.
  Packages are never installed on your machine. Note that the build itself has network access to PyPI.
* **No shell.** Git and Docker are called with argument lists. Git runs only inside the workspace (it never uses a parent repository)
  with hooks disabled.
* **Untrusted `.git`.** Code running in the sandbox can create files in the workspace, including a `.git` directory with a
  malicious config (git filter drivers, external diff tools, ... run programs on your machine). Before every git call the agent checks
  that `.git` is a real directory whose config contains only the settings `git init` writes; otherwise it refuses to run git
  and tells you to delete it.
* **Special files.** Tools skip symlinks, FIFOs and devices, and reads are size-bounded, so sandboxed code cannot hang or exhaust the host process.
* **Human approval** is required for dangerous tools (see above).

## Agent state

The agent keeps its notes in `workspace/.agent/`: `PLAN.md`, `TODO.md`, `MEMORY.md` (timestamped notes), `reviews/review_<timestamp>.md`,
and `sandbox_image.txt`. This folder is excluded from the workspace's own Git repository (`git_init` writes a `.gitignore` for it).
Conversation history lives in memory only and is lost when the process restarts; the files in `.agent/` persist.

## Project layout

```
app.py            Streamlit front end
cli.py            command-line front end
agent/llm.py      get_chat_model(): init_chat_model, provider/model from env, cached
agent/prompts.py  system prompt and per-stage prompts
agent/sandbox.py  Docker execution and image building
agent/tools.py    workspace-confined tools, approval policy, per-stage tool lists
agent/graph.py    react graph, pipeline graph, shared run/approval driver
workspace/        the only directory the agent should modify
tests/            pytest suite (no API key or Docker needed)
```

## Tests

```bash
pip install pytest
python -m pytest
```

The suite uses a scripted fake chat model and mocks Docker, so it needs neither an API key nor a Docker daemon.
It checks the tools, path safety, the exact `docker run` / `docker build` commands, the approval policy, the stage enforcement
(including attempts to bypass it), the CLI, and `app.py` driven through a fake `streamlit` module.
It does **not** run a real container or call a real model, so do a first run with your own key and Docker before relying on it.
