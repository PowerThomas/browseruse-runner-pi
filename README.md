# browseruse-runner (Raspberry Pi)

Minimal FastAPI runner for browser-use, packaged for Docker Compose on a Raspberry Pi.
This repo contains only the runner service; it assumes n8n is already installed and running.

## What it does

- Runs browser-use tasks synchronously via `POST /run`
- Supports async jobs with FIFO queue via `POST /jobs`
- Persists run artifacts, per-step screenshots, and an HTML report
- Exposes step list + per-step screenshot + report endpoints
- Includes artifact cleanup and profile management endpoints
- Keeps API internal-only (Docker network) and protected by `X-API-Key`

## Requirements

- Docker + Docker Compose
- A working n8n stack (already installed)

## Setup

1) Clone the repo

```
git clone https://github.com/PowerThomas/browseruse-runner-pi.git
cd browseruse-runner-pi
```

2) Create `.env` from the example

```
cp .env.example .env
```

3) Update `.env`

- `RUNNER_API_KEY`: shared secret for all API calls
- `BROWSER_USE_API_KEY`: your browser-use API key

4) Start the runner

```
docker compose up -d --build
```

## Docker Compose

This repo ships a standalone `docker-compose.yml` with only the runner service.
If you already have an n8n compose stack, you can copy the `browseruse-runner` service
into your existing file instead of running a second compose project.

## API Usage

Base URL from the n8n container: `http://browseruse-runner:8000`

All requests must include:

```
X-API-Key: <RUNNER_API_KEY>
```

### POST /run

Runs a task synchronously and returns results plus optional step metadata.

Request body:

```json
{
  "task": "Open https://example.com and report the title.",
  "url": "https://example.com",
  "profile_id": "default",
  "record_trace": true,
  "include_steps": true,
  "include_step_screenshots": "paths"
}
```

### GET /runs/{run_id}/steps

Returns an array of steps with `screenshot_file` when available.

### GET /runs/{run_id}/steps/{step_number}/screenshot

Downloads the screenshot for a step as `image/png`.

### GET /runs/{run_id}/report

Returns the generated HTML report (`text/html`).

### POST /maintenance/cleanup

Cleans old artifacts under `/app/artifacts`.

Defaults:
- `ARTIFACTS_MAX_DAYS=7`
- `ARTIFACTS_MAX_RUNS=100`

### Profile management

- `GET /profiles` list profiles (name, last_modified, size_bytes)
- `POST /profiles/{name}/reset` delete a profile
- `POST /profiles/{name}/clone` with body `{"to":"newname"}`

### Async jobs

- `POST /jobs` with the same body as `/run`
- `GET /jobs/{run_id}` returns status + response when done
- `POST /jobs/{run_id}/cancel` best-effort cancel

If a job is running, new jobs are queued and start automatically (FIFO).

### Errors

- `401 Unauthorized`: missing or invalid `X-API-Key`
- `409 Profile is in use`: Chrome is already running in the container
- `429 Runner is busy`: another run is in progress
- `404`: run/step/screenshot not found

## n8n flow (recommended)

1) Node A: HTTP Request (POST)
   - URL: `/run`
   - JSON: `include_steps=true`, `include_step_screenshots="paths"`
2) Node B: Split Out Items on `response.steps`
3) Node C: HTTP Request (GET)
   - URL: `/runs/{{$json.run_id}}/steps/{{$json.step_number}}/screenshot`
   - Enable “Download” so it becomes binary

Each item now includes a binary PNG screenshot for the step.

To fetch the HTML report, add a HTTP Request node:
- `GET /runs/{{$json.run_id}}/report`

For async jobs, POST `/jobs` and poll `GET /jobs/{{$json.run_id}}` until `status` is `completed`.

## Project layout

- `browseruse-runner/` service code
- `browseruse-runner/README.md` endpoint details

## Tests

Smoke tests run inside the runner container:

```
docker compose exec -T browseruse-runner python -m unittest /app/tests/test_api_smoke.py
```

Optional tests (destructive) can be enabled via env vars:
- `RUN_TEST_CLEANUP=1`
- `RUN_TEST_PROFILE_MUTATION=1`

## License

MIT License. See `LICENSE`.

## Contributing

See `CONTRIBUTING.md`.
