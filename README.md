# Diploid

A dual-model inference orchestrator that combines a GPU-resident model and a CPU-resident model into a single OpenAI-compatible API endpoint, with three modes that trade latency for quality.

---

## Purpose

Most local LLM inference setups force a choice: a small model on the GPU (fast but limited) or a large model on the CPU (smart but slow). Hardware with significant RAM but modest VRAM — a very common configuration — leaves capability on the table no matter which you pick.

Diploid runs both simultaneously and exposes their combined capability behind one API. A caller pointing at Diploid sees a single OpenAI-compatible endpoint. Behind it, two `llama.cpp` server instances cooperate in one of three modes:

- **`diploid-fast`** — GPU drafts tokens, GPU target verifies. Speculative decoding accelerates the target model 2-3x without changing its outputs.
- **`diploid-balanced`** — GPU produces a draft response. CPU model critiques and revises. Output quality exceeds either model alone on tasks where review helps.
- **`diploid-deep`** — Both models answer. A merge step reconciles into a final response. Highest quality ceiling, highest latency.

The user picks the mode per-request by setting the `model` field in their OpenAI request. Pass-through routing to either backend is also supported for benchmarking and debugging.

### Design principles

1. **One binary, one daemon, one port.** Users should not manage two `llama-server` instances by hand. Diploid spawns and supervises them.
2. **OpenAI API compatibility is non-negotiable.** Every existing client (OpenAI SDK, LangChain, LlamaIndex, curl scripts) must work without modification.
3. **GGUF-only.** It's the only format that runs on both CPU and GPU through one engine (llama.cpp). EXL2, AWQ, GPTQ, and MLX are non-goals.
4. **Native binaries by default.** Docker is optional, not required. Containerization adds CUDA/driver/volume failure surfaces that aren't worth it for a single-user dev tool.
5. **Observability is a feature.** Draft acceptance rate, revision rate, merge agreement rate are first-class metrics. Without them, users can't tell when a mode is helping vs hurting.
6. **Honest defaults, configurable internals.** Critic and merge prompts ship with sane defaults but live in user-editable config files. They are the highest-leverage tuning surface.

### Non-goals

- Multi-tenant serving. Single-user, single-concurrent-request first. (`--parallel` support is a later phase.)
- Training, fine-tuning, or LoRA management.
- Model formats other than GGUF.
- A web UI. CLI + API only. Frontends can be built on top.
- Cloud model providers (OpenAI, Anthropic, etc.). Local-only.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Client (OpenAI SDK, curl, Animus Prion, etc.)       │
└─────────────────────┬────────────────────────────────┘
                      │ POST /v1/chat/completions
                      │ { "model": "diploid-fast" | "diploid-balanced"
                      │            | "diploid-deep" | "<passthrough-id>" }
                      ▼
┌──────────────────────────────────────────────────────┐
│  diploid orchestrator (default port 8000)            │
│  ┌────────────────────────────────────────────────┐  │
│  │ HTTP layer (axum)                              │  │
│  │   /v1/chat/completions   (streaming + non)     │  │
│  │   /v1/completions                              │  │
│  │   /v1/models                                   │  │
│  │   /health                                      │  │
│  │   /metrics  (Prometheus)                       │  │
│  │   /diploid/status  (rich status JSON)          │  │
│  └────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────┐  │
│  │ Mode router                                    │  │
│  │   fast      → GPU server (with draft model)    │  │
│  │   balanced  → GPU server → CPU server (critic) │  │
│  │   deep      → GPU + CPU parallel → merge       │  │
│  │   passthrough → either server                  │  │
│  └────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────┐  │
│  │ Subprocess supervisor                          │  │
│  │   spawn / health / restart / shutdown          │  │
│  └────────────────────────────────────────────────┘  │
└──────────┬──────────────────────────┬────────────────┘
           │                          │
           ▼                          ▼
┌────────────────────────┐  ┌──────────────────────────┐
│ llama-server (GPU)     │  │ llama-server (CPU)       │
│ port 8080              │  │ port 8081                │
│ --n-gpu-layers 99      │  │ --n-gpu-layers 0         │
│ --model <gpu-target>   │  │ --model <cpu>            │
│ [--model-draft <draft>]│  │                          │
│ small/fast model       │  │ large/slow model         │
└────────────────────────┘  └──────────────────────────┘
```

The orchestrator is the only thing the user talks to. The two `llama-server` processes are implementation details and never accept external traffic — they bind to `127.0.0.1` only.

### Why this shape

- **Speculative decoding lives in one process.** llama.cpp's native speculation requires draft and target in the same `llama-server` instance because per-token logit verification has no IPC. Diploid does not reimplement this; it lets `llama-server` handle it via `--model-draft`.
- **Producer/critic and ensemble live in the orchestrator.** These are HTTP-orchestrated patterns: the orchestrator makes one or more calls to each backend and assembles the final response.
- **Three model slots, not two.** `gpu-draft`, `gpu-target`, `cpu`. The draft slot is optional; without it, `diploid-fast` is unavailable but `balanced` and `deep` still work.

---

## Tech stack

- **Language: Rust.** Single static binary, fast subprocess management, clean async story for streaming. Crates:
  - `tokio` — async runtime
  - `axum` — HTTP server
  - `tower` / `tower-http` — middleware
  - `reqwest` — HF + URL downloads, backend HTTP calls
  - `serde` / `serde_json` — request/response serialization
  - `clap` — CLI parsing
  - `tracing` / `tracing-subscriber` — structured logging
  - `sha2` — content addressing
  - `toml` — config files
  - `prometheus` — metrics
- **Build target:** Linux x86_64 first. Windows is a nice-to-have once Linux is solid (LogOS is the primary deployment target).
- **External dependency:** a working `llama-server` binary on PATH or at a configured location. Diploid does not vendor or build llama.cpp. Document the minimum llama.cpp version (initial target: a recent release with stable speculative decoding support; pin and document at first release).

---

## Filesystem layout

```
~/.config/diploid/
  config.toml                # main configuration
  prompts/
    critic.toml              # critic prompt template (balanced mode)
    merge.toml               # merge prompt template (deep mode)
    debate.toml              # debate round prompt (deep mode, optional)

~/.local/share/diploid/
  models/
    <sha256>/
      model.gguf
      metadata.json          # source URL, family, params, quant, tokenizer hash
  logs/
    diploid.log
    llama-gpu.log
    llama-cpu.log
  state/
    pid.lock                 # running daemon PID
    runtime.json             # current slot assignments, ports, etc.
```

Models are content-addressed by SHA256. Identical files pulled from different sources (HF mirror, direct URL) dedupe. Metadata records the tokenizer hash, which is checked at `run` time when `diploid-fast` mode is requested — speculative decoding requires identical tokenizers between draft and target, and silently mismatched tokenizers are the most common footgun.

---

## CLI surface

### Model management

```
diploid pull <source> --as <gpu-draft|gpu-target|cpu> [--name <alias>]

# Sources:
#   hf:<repo>:<filename>           Hugging Face GGUF file
#   https://...                    Direct URL
#   /absolute/path/to/model.gguf   Local file (copies into store)

# Examples:
diploid pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q4_k_m.gguf --as gpu-draft
diploid pull hf:Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q4_k_m.gguf --as gpu-target
diploid pull hf:Qwen/Qwen2.5-32B-Instruct-GGUF:qwen2.5-32b-instruct-q4_k_m.gguf --as cpu
```

```
diploid models                # list all pulled models with aliases, sizes, slot tags
diploid models rm <id|alias>  # remove from store
diploid models verify         # checksum all stored models
```

### Runtime

```
diploid run \
  [--gpu-draft <id|alias>] \
  --gpu-target <id|alias> \
  --cpu <id|alias> \
  [--port 8000] \
  [--gpu-ctx 8192] \
  [--cpu-ctx 8192] \
  [--detach]

diploid stop
diploid restart
diploid status                # human-readable summary
diploid status --json         # machine-readable
diploid logs [--follow] [--server gpu|cpu|orchestrator]
```

### Config

```
diploid config edit            # opens $EDITOR on config.toml
diploid config show            # prints effective config
diploid config prompts edit <critic|merge|debate>
```

### Diagnostics

```
diploid bench --mode <fast|balanced|deep|gpu-only|cpu-only> --prompts <file>
# Runs a benchmark suite, reports tok/s, TTFT, draft acceptance, etc.

diploid doctor
# Checks: llama-server on PATH, version compatibility, model tokenizer matches,
#         VRAM/RAM available, CUDA visible, port 8000 free, etc.
```

---

## OpenAI API surface

Implement enough to satisfy the major SDKs:

### `GET /v1/models`

Returns the diploid mode IDs (only those whose required slots are filled) plus pass-through IDs for both backends.

```json
{
  "object": "list",
  "data": [
    {"id": "diploid-fast",     "object": "model", "owned_by": "diploid"},
    {"id": "diploid-balanced", "object": "model", "owned_by": "diploid"},
    {"id": "diploid-deep",     "object": "model", "owned_by": "diploid"},
    {"id": "qwen2.5-7b-q4",    "object": "model", "owned_by": "diploid:gpu"},
    {"id": "qwen2.5-32b-q4",   "object": "model", "owned_by": "diploid:cpu"}
  ]
}
```

### `POST /v1/chat/completions`

Standard OpenAI request body. The `model` field selects the mode. Streaming via `"stream": true` returns SSE.

Per-request diploid configuration goes in a top-level `diploid` object (ignored by other servers, accepted by ours):

```json
{
  "model": "diploid-deep",
  "messages": [{"role": "user", "content": "..."}],
  "stream": true,
  "temperature": 0.7,
  "diploid": {
    "deep_strategy": "critic-merge",
    "debate_rounds": 2,
    "vote_n": 4,
    "stream_drafts": false,
    "merge_judge": "cpu"
  }
}
```

Unknown `diploid.*` fields are ignored. Defaults apply when omitted.

### `POST /v1/completions`

Legacy text completion. Thin wrapper. Same mode selection via `model`.

### `GET /health`

Returns 200 if both backends pass their health checks; 503 otherwise. Body:

```json
{
  "status": "healthy",
  "backends": {
    "gpu": {"status": "healthy", "model": "qwen2.5-7b-q4", "draft": "qwen2.5-1.5b-q4"},
    "cpu": {"status": "healthy", "model": "qwen2.5-32b-q4"}
  }
}
```

### `GET /metrics`

Prometheus exposition format. See "Observability" below.

### `GET /diploid/status`

Rich JSON for `diploid status`. Includes per-mode recent stats (rolling window).

---

## Mode specifications

### Mode 1: `diploid-fast` (speculative decoding)

**Slots required:** `gpu-draft`, `gpu-target`. Both must use compatible tokenizers (identical vocab and special tokens). Diploid checks at startup and refuses to enable this mode if they don't match.

**Implementation:** Pass through to the GPU `llama-server`, which is launched with both `--model <gpu-target>` and `--model-draft <gpu-draft>`. llama.cpp handles the actual speculation. The orchestrator just routes the request and forwards the stream.

**Output guarantee:** Byte-identical to running the target model alone. Speculation is a performance optimization, not a quality change.

**Streaming:** Native. Tokens stream as they're verified.

**Metrics emitted per request:**
- Draft acceptance rate (count of accepted draft tokens / total proposed)
- Tokens/sec (decode)
- Time to first token

**Tuning surface:** llama.cpp's `--draft-max` and `--draft-min` flags, exposed via `config.toml`.

### Mode 2: `diploid-balanced` (producer/critic)

**Slots required:** `gpu-target`, `cpu`. `gpu-draft` optional (accelerates the producer step if present).

**Pipeline:**

1. Producer call to GPU backend. Generate a complete draft response.
2. Critic call to CPU backend with the critic prompt template, the original user message, and the draft.
3. CPU returns the revised response.
4. Stream the revised response to the client.

**Streaming:** Wait-and-stream by default. Producer must complete before streaming starts (cannot commit to an unrevised answer). First-token latency = producer total time + critic TTFT.

Optional: `diploid.stream_drafts: true` enables a side channel — producer tokens stream as SSE *comments* (lines beginning with `:`), which OpenAI-compatible clients ignore but observability tools can scrape. The official `data:` events still carry only the revised output.

**Critic prompt:** loaded from `~/.config/diploid/prompts/critic.toml`. Default template:

```toml
[critic]
template = """You are reviewing a draft response to a user's question. Identify any factual errors, logical gaps, missed requirements, or unclear passages. Then produce a revised response that fixes these issues. If the draft is already correct and complete, return it unchanged.

User question:
{{question}}

Draft response:
{{draft}}

Revised response:
"""
```

The `{{question}}` and `{{draft}}` placeholders are required. Other variables can be added (e.g., `{{system}}` for the original system prompt) and will be substituted if referenced.

**Metrics emitted per request:**
- Revision rate (fraction of requests where critic output differs non-trivially from draft, measured by token-level edit distance threshold — configurable, default 5%)
- Producer tok/s, critic tok/s
- Producer time, critic time, total time

### Mode 3: `diploid-deep` (ensemble)

**Slots required:** `gpu-target`, `cpu`. `gpu-draft` optional.

Three sub-strategies, selected via `diploid.deep_strategy`:

#### 3a. `critic-merge` (default)

1. GPU and CPU generate independently in parallel.
2. Merge call: the configured `merge_judge` (default `cpu`) is given both responses and the merge prompt. Returns the synthesized response.
3. Stream the merged response.

Wall-clock latency ≈ max(gpu_gen, cpu_gen) + merge_time.

#### 3b. `debate`

1. Round 1: GPU and CPU generate independently in parallel.
2. Rounds 2 through N (`debate_rounds`): each model is shown the other's previous round answer and asked to revise its own.
3. Final merge: `merge_judge` selects or merges the round-N responses.

Default `debate_rounds: 2`. Higher values increase cost roughly linearly.

#### 3c. `vote`

1. Generate `vote_n` candidates total, distributed across both backends (e.g., `vote_n: 4` → 2 from GPU, 2 from CPU, different sampling seeds).
2. Judge call: `merge_judge` ranks all candidates and returns the winner (or a synthesis).

Default `vote_n: 4`.

**Streaming:** Same wait-and-stream policy as `balanced`. The final merged/selected response streams. Intermediate drafts available via SSE-comment side channel when `stream_drafts: true`.

**Merge prompt:** loaded from `~/.config/diploid/prompts/merge.toml`. Default:

```toml
[merge]
template = """Two assistants answered the same question. Produce a single best response, preferring correctness, completeness, and clarity. When the assistants disagree, choose the better-supported claim. Do not mention the assistants in your output — return only the final response.

User question:
{{question}}

Assistant A response:
{{response_a}}

Assistant B response:
{{response_b}}

Final response:
"""
```

For `debate`, the round prompt lives in `prompts/debate.toml`:

```toml
[debate_round]
template = """You previously answered this question:

Question: {{question}}
Your previous answer: {{your_previous}}

Another assistant answered:
{{other_previous}}

Considering their answer, revise your own. If you still believe your answer is correct, defend it. If they identified something you missed, incorporate it. Return only your revised answer.
"""
```

**Important quality note:** Ensemble methods help most when underlying models make *different* mistakes. Two models from the same family (e.g., Qwen2.5-7B and Qwen2.5-32B) make highly correlated mistakes — they share training data and architecture. Document this prominently. Cross-family pairs (e.g., Qwen + Llama, Qwen + Mistral, Qwen + DeepSeek) are where `diploid-deep` shines.

**Metrics emitted per request:**
- Merge agreement rate (when judge picks A entirely, B entirely, or blends — three-way distribution)
- Debate convergence (token-level similarity between round N and round N-1)
- Per-stage timings

### Pass-through routing

`model: "<gpu-target-id>"` or `model: "<gpu-target-alias>"` routes directly to the GPU server, bypassing all diploid logic. Same for CPU. Used for benchmarking and debugging. Streams natively.

---

## Subprocess management

### Startup sequence

1. Validate config (`config.toml` parseable, prompt templates parseable, llama-server binary exists).
2. Verify model files exist for all configured slots; checksum-verify.
3. If `diploid-fast` will be available, check tokenizer compatibility between `gpu-draft` and `gpu-target`. If incompatible, log a warning and disable `diploid-fast`; do not abort.
4. Spawn GPU `llama-server` with computed flags (see below). Capture stdout/stderr to `logs/llama-gpu.log`.
5. Spawn CPU `llama-server` similarly.
6. Poll `/health` on each backend until both report ready, with a timeout (default 120s — large models take time to mmap).
7. Bind orchestrator HTTP server on configured port.
8. Write `state/pid.lock` and `state/runtime.json`.

### Backend launch flags

GPU server (with optional draft):

```
llama-server \
  --model <gpu-target-path> \
  [--model-draft <gpu-draft-path>] \
  --port 8080 \
  --host 127.0.0.1 \
  --n-gpu-layers 99 \
  --ctx-size <gpu-ctx> \
  --parallel 1 \
  --threads 2 \
  --log-file <logs>/llama-gpu.log \
  --no-webui
```

CPU server:

```
llama-server \
  --model <cpu-path> \
  --port 8081 \
  --host 127.0.0.1 \
  --n-gpu-layers 0 \
  --ctx-size <cpu-ctx> \
  --parallel 1 \
  --threads <auto: nproc - 3> \
  --log-file <logs>/llama-cpu.log \
  --no-webui
```

Thread allocation rationale: leave 2 cores for GPU server's host-side work (sampling, tokenization), 1 core for the orchestrator. Remaining cores go to the CPU model. On an 8-core machine: GPU=2, orchestrator=1, CPU=5. Configurable via `config.toml` for users who want to override.

### Health checks

Poll `GET /health` on each backend every 5s after startup. Three consecutive failures mark the backend `unhealthy`. The orchestrator's `/health` returns 503 if any backend is unhealthy.

### Crash recovery

If a backend exits unexpectedly:

1. Mark unhealthy.
2. Attempt one restart with the same flags.
3. If restart succeeds, resume.
4. If restart fails, leave it down. Modes requiring that backend return 503; modes not requiring it continue working. Log loudly.

### Shutdown

`diploid stop`:

1. Stop accepting new requests.
2. Wait up to 30s for in-flight requests to finish.
3. Send SIGTERM to backends; wait 10s.
4. SIGKILL anything still running.
5. Remove `state/pid.lock`.

---

## Configuration

`~/.config/diploid/config.toml`:

```toml
[server]
port = 8000
host = "127.0.0.1"

[backends]
llama_server_path = "llama-server"  # or absolute path
gpu_port = 8080
cpu_port = 8081
gpu_ctx = 8192
cpu_ctx = 8192
gpu_threads = 2
cpu_threads = "auto"  # auto = nproc - gpu_threads - 1
startup_timeout_secs = 120

[modes]
default_deep_strategy = "critic-merge"
default_debate_rounds = 2
default_vote_n = 4
default_merge_judge = "cpu"
revision_threshold = 0.05  # min token edit distance to count as a revision

[fast]
draft_max = 16
draft_min = 4

[logging]
level = "info"  # trace, debug, info, warn, error
format = "json"  # json or pretty

[metrics]
enabled = true
prometheus_path = "/metrics"
```

---

## Observability

### Prometheus metrics

Per backend:
- `diploid_backend_up{backend="gpu|cpu"}` — gauge, 1 if healthy
- `diploid_backend_request_duration_seconds{backend, mode, stage}` — histogram
- `diploid_backend_tokens_generated_total{backend, mode}` — counter
- `diploid_backend_tokens_per_second{backend, mode}` — gauge (rolling)

Per mode:
- `diploid_mode_requests_total{mode}` — counter
- `diploid_mode_request_duration_seconds{mode}` — histogram
- `diploid_mode_ttft_seconds{mode}` — histogram

Mode-specific:
- `diploid_fast_draft_acceptance_rate` — gauge (rolling)
- `diploid_balanced_revision_rate` — gauge (rolling)
- `diploid_deep_merge_picks_total{pick="a|b|blend"}` — counter
- `diploid_deep_debate_convergence` — gauge (rolling)

### Status JSON (`GET /diploid/status`)

```json
{
  "version": "0.1.0",
  "uptime_secs": 12345,
  "slots": {
    "gpu_draft": {"id": "qwen2.5-1.5b-q4", "loaded": true, "bytes": 1073741824},
    "gpu_target": {"id": "qwen2.5-7b-q4", "loaded": true, "bytes": 4831838208},
    "cpu": {"id": "qwen2.5-32b-q4", "loaded": true, "bytes": 21260683264}
  },
  "modes_available": ["diploid-fast", "diploid-balanced", "diploid-deep"],
  "recent": {
    "fast": {"requests": 42, "avg_accept_rate": 0.74, "avg_tps": 38.2},
    "balanced": {"requests": 17, "avg_revision_rate": 0.31, "avg_total_secs": 8.4},
    "deep": {"requests": 6, "merge_picks": {"a": 2, "b": 1, "blend": 3}}
  }
}
```

---

## Build phases

The phases below are sized to be small enough to verify independently. Each ends with a working binary.

### Phase 0: feasibility spike

A bash script that launches two `llama-server` instances with hardcoded paths, plus a Python script that hits both and prints tokens/sec. Confirms the hardware and llama.cpp version behave as expected.

**Deliverable:** `scripts/spike.sh` and `scripts/spike_bench.py`. Documented in `docs/feasibility.md`.

### Phase 1: orchestrator MVP

Rust binary with:
- `clap`-based CLI: `pull`, `models`, `run`, `stop`, `status`
- `pull` for HF, URL, and local paths
- Subprocess supervisor: spawn, health-check, shutdown
- HTTP server on port 8000 with `/v1/models`, `/v1/chat/completions`, `/health`
- Pass-through routing only (route by `model` field to GPU or CPU backend)
- No diploid modes yet — the diploid model IDs are exposed but return 501.

**Acceptance test:** `curl` and the OpenAI Python SDK can hit the orchestrator and get streaming responses from either backend by name.

### Phase 2: `diploid-fast`

- Wire GPU server's `--model-draft` flag.
- Tokenizer compatibility check at startup.
- Expose `diploid-fast` model ID (only when both slots filled and tokenizers match).
- Forward requests to GPU server transparently.
- Emit draft acceptance metrics (parse from llama.cpp's response or its log output — verify what the upstream actually exposes).

**Acceptance test:** Same prompt to `diploid-fast` and to `gpu-target` directly produces identical output (allowing for sampling variance with `temperature: 0`). `diploid-fast` is faster.

### Phase 3: `diploid-balanced`

- Two-stage call (producer → critic).
- Critic prompt template loading and substitution.
- Wait-and-stream behavior.
- Optional `stream_drafts` SSE-comment side channel.
- Revision rate metric.

**Acceptance test:** Inject a deliberately wrong producer response (via mocked GPU backend in tests) and verify the critic corrects it.

### Phase 4: `diploid-deep` — critic-merge strategy

- Parallel calls to both backends.
- Merge prompt template loading.
- Configurable `merge_judge`.
- Merge agreement metric.

**Acceptance test:** When both backends are pointed at the same model file, `diploid-deep` output is roughly comparable to either backend alone (no obvious quality regression). Verifies the merge step doesn't degrade output.

### Phase 5: `diploid-deep` — debate and vote

- `debate` strategy with N rounds.
- `vote` strategy with `vote_n` samples.
- Debate convergence metric.

**Acceptance test:** On a curated set of reasoning questions, `debate` with cross-family models beats single-model baseline on accuracy. (This is the empirical bet; if it doesn't show wins, the feature is documented as experimental.)

### Phase 6: polish and ship

- Prometheus `/metrics`.
- `diploid bench` subcommand.
- `diploid doctor` subcommand.
- Systemd unit file in `packaging/`.
- Optional Docker Compose file in `packaging/docker/`.
- HF auto-update for tracked models (`diploid pull --update`).
- Pre-built binary releases for Linux x86_64.
- README, install script, examples.

---

## Project layout

```
diploid/
  Cargo.toml                       # workspace root
  Cargo.lock
  README.md
  LICENSE                          # MIT or Apache-2.0
  rust-toolchain.toml              # pin Rust version
  .github/
    workflows/
      ci.yml                       # fmt, clippy, test
      release.yml                  # tag → binary release

  crates/
    diploid-cli/                   # binary crate, the `diploid` command
      src/
        main.rs
        commands/
          pull.rs
          models.rs
          run.rs
          stop.rs
          status.rs
          config.rs
          bench.rs
          doctor.rs
    diploid-core/                  # library crate, all the logic
      src/
        lib.rs
        config.rs                  # config.toml parsing
        store.rs                   # model storage, hashing, metadata
        pull.rs                    # HF, URL, local pull
        supervisor.rs              # subprocess spawn / health / restart
        backend.rs                 # HTTP client to llama-server
        tokenizer.rs               # tokenizer hash / compatibility
        modes/
          mod.rs
          fast.rs
          balanced.rs
          deep.rs
          passthrough.rs
        prompts.rs                 # template loading and substitution
        metrics.rs                 # prometheus registry
    diploid-api/                   # library crate, HTTP surface
      src/
        lib.rs
        openai.rs                  # request/response types
        routes.rs                  # axum routes
        sse.rs                     # streaming helpers

  prompts/                         # default templates, embedded in binary
    critic.toml
    merge.toml
    debate.toml

  packaging/
    systemd/
      diploid.service
    docker/
      Dockerfile
      docker-compose.yml

  scripts/
    spike.sh
    spike_bench.py

  docs/
    feasibility.md
    architecture.md
    modes.md
    config.md
    troubleshooting.md

  tests/
    integration/
      fast_mode.rs
      balanced_mode.rs
      deep_mode.rs
      passthrough.rs
      pull.rs
```

---

## Open questions to resolve during implementation

These are deliberately left open in this spec because the answers depend on empirical data Claude Code will gather while building:

1. **What does llama.cpp's HTTP API actually expose for draft acceptance metrics?** If it's not in the response JSON, parse it from logs. If not in logs either, compute it indirectly (compare draft model's solo speed to spec-decode speed and infer).
2. **Token budget enforcement.** `diploid-deep` can multiply token usage 4-5x. Probably want a `diploid.max_total_tokens` to bound the whole pipeline. Decide once we measure typical multipliers.
3. **What's the minimum llama.cpp version that has stable speculative decoding via `--model-draft`?** Pin in the README and `doctor` command.
4. **Streaming semantics for `balanced` and `deep` when the critic/merge call itself fails.** Fall back to the producer/draft? Error out? Default: fall back, with a header indicating degraded mode. Confirm during implementation.
5. **How to handle the case where GPU has free VRAM and CPU's model could partially offload to it.** Out of scope for v1 (we're explicitly running CPU-only there), but worth a config flag in v2.

---

## Glossary

- **Draft model** — small, fast model used to propose tokens in speculative decoding.
- **Target model** — larger model that verifies the draft's proposals; its outputs are what the user sees.
- **Critic** — model that reviews and revises a draft response (balanced mode).
- **Merge judge** — model that synthesizes two responses into one (deep mode).
- **Slot** — a model role: `gpu-draft`, `gpu-target`, or `cpu`. Each slot can hold one model at a time.
- **Backend** — one of the two `llama-server` subprocesses (GPU or CPU).
- **Mode** — one of `diploid-fast`, `diploid-balanced`, `diploid-deep`, or pass-through.
