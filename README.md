# browseruse-runner (Raspberry Pi)

Minimal FastAPI runner for browser-use, packaged for Docker Compose on a Raspberry Pi.
This repo contains only the runner service; it assumes n8n is already installed and running.

## Overview

- Runs browser-use tasks synchronously via `POST /run`
- Supports async jobs with FIFO queue via `POST /jobs`
- Persists run artifacts, per-step screenshots, and an HTML report
- Exposes step list + per-step screenshot + report endpoints
- Supports human-in-the-loop pause/resume during active runs
- Includes artifact cleanup and profile management endpoints
- Keeps API internal-only (Docker network) and protected by `X-API-Key`

## Prerequisites

- Docker + Docker Compose
- A working n8n stack (already installed)

## Quick start

1) Clone the repo

```
git clone https://github.com/PowerThomas/browseruse-runner-pi.git
cd browseruse-runner-pi
```

2) Create `.env` from the example

```
cp .env.example .env
```

3) Fill in `.env`

- `RUNNER_API_KEY`: shared secret for all API calls
- `BROWSER_USE_API_KEY`: your browser-use API key
- Optional provider keys if you want to use non-browseruse LLMs (see `.env.example`)

4) Start the runner

```
docker compose up -d --build
```

## Using it from n8n (beginner flow)

1) Node A: HTTP Request (POST)
   - URL: `/run`
   - JSON: `include_steps=true`, `include_step_screenshots="paths"`
2) Node B: Split Out Items on `response.steps`
3) Node C: HTTP Request (GET)
   - URL: `/runs/{{$json.run_id}}/steps/{{$json.step_number}}/screenshot`
   - Enable "Download" so it becomes binary

Each item now includes a binary PNG screenshot for the step.

To fetch the HTML report, add a HTTP Request node:
- `GET /runs/{{$json.run_id}}/report`

For async jobs, POST `/jobs` and poll `GET /jobs/{{$json.run_id}}` until `status` is `completed`.

For human-in-the-loop automation, use `/jobs` and poll `/runs/{run_id}/status` or
`/jobs/{run_id}` until `paused=true`, then prompt for input and call
`POST /runs/{run_id}/resume` with the extra guidance.

## Configuration

The runner reads these env vars:

- `RUNNER_API_KEY` (required)
- `BROWSER_USE_API_KEY` (required)
- `ARTIFACTS_MAX_DAYS` (optional, default 7)
- `ARTIFACTS_MAX_RUNS` (optional, default 100)
- Optional LLM provider keys and Azure/Ollama settings are documented in `.env.example`.

## LLM selection

You can choose a supported LLM provider per request using the `llm` object in
`/run` and `/jobs`:

```json
{
  "task": "Open https://example.com and report the title.",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini"
  }
}
```

API keys must come from `.env` (never from the request body).

## Docker Compose notes

This repo ships a standalone `docker-compose.yml` with only the runner service.
If you already have an n8n compose stack, you can copy the `browseruse-runner` service
into your existing file instead of running a second compose project.

## API at a glance

Base URL from the n8n container: `http://browseruse-runner:8000`

All requests must include:

```
X-API-Key: <RUNNER_API_KEY>
```

Common endpoints:
- `POST /run`
- `GET /llms`
- `GET /runs/{run_id}/steps`
- `GET /runs/{run_id}/steps/{step_number}/screenshot`
- `GET /runs/{run_id}/report`
- `POST /jobs`
- `GET /jobs/{run_id}`
- `POST /runs/{run_id}/pause`
- `POST /runs/{run_id}/resume`

Detailed API docs live in `browseruse-runner/README.md`.

## Live view (noVNC/VNC)

To watch a run live:

1) Call `/run` with `interactive=true`
2) Open `http://127.0.0.1:7900/vnc.html`
3) If connecting remotely, use SSH port forwarding:

```
ssh -L 7900:127.0.0.1:7900 pi@<your-pi-host>
```

## Maintenance

- Cleanup old artifacts: `POST /maintenance/cleanup`
- List profiles: `GET /profiles`
- Reset profile: `POST /profiles/{name}/reset`
- Clone profile: `POST /profiles/{name}/clone` with `{"to":"newname"}`

## Tests

Smoke tests run inside the runner container:

```
docker compose exec -T browseruse-runner python -m unittest /app/tests/test_api_smoke.py
```

Optional tests (destructive) can be enabled via env vars:
- `RUN_TEST_CLEANUP=1`
- `RUN_TEST_PROFILE_MUTATION=1`
- `RUN_TEST_REAL_SITES=1`
- `RUN_TEST_HITL=1`

Test details live in `browseruse-runner/tests/README.md`.

## Project layout

- `browseruse-runner/` service code
- `browseruse-runner/README.md` endpoint details
- `browseruse-runner/tests/` smoke tests

## License

MIT License. See `LICENSE`.

## Contributing

See `CONTRIBUTING.md`.
