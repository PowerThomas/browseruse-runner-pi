# browseruse-runner API

A minimal FastAPI service that runs browser-use tasks behind an internal Docker
network. All endpoints require the `X-API-Key` header matching `RUNNER_API_KEY`.

Base URL (from the n8n container): `http://browseruse-runner:8000`

## Start here

1) Add the auth header to every request:

```
X-API-Key: <RUNNER_API_KEY>
```

2) Run a task:

```bash
curl -sS -X POST http://browseruse-runner:8000/run \
  -H "X-API-Key: $RUNNER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"task":"Open https://example.com and report the title.","url":"https://example.com","include_steps":true,"include_step_screenshots":"paths"}'
```

3) Use the `steps` array to fetch screenshots:

```
GET /runs/{run_id}/steps/{step_number}/screenshot
```

## Concurrency and profile safety

- If a run is already in progress, the API returns `429 Runner is busy`.
- If Chromium is running inside the container, the API returns `409 Profile is in use`
  and will not delete profile locks.

## Runs (sync)

### POST /run

Runs a task synchronously and returns the full result.

Request body:

```json
{
  "task": "Open https://example.com and report the title.",
  "url": "https://example.com",
  "profile_id": "default",
  "record_trace": true,
  "interactive": false,
  "keep_open_seconds": 0,
  "include_steps": true,
  "include_step_screenshots": "paths",
  "llm": {
    "provider": "browseruse",
    "model": "bu-latest"
  }
}
```

Fields:
- `task` (string, required)
- `url` (string, optional)
- `profile_id` (string, default: `default`)
- `record_trace` (bool, default: `true`)
- `interactive` (bool, default: `false`) launches a headed browser for VNC/noVNC
- `keep_open_seconds` (int, default: `0`) keeps the browser open after completion
- `include_steps` (bool, default: `true`)
- `include_step_screenshots` (`none` or `paths`, default: `none`)
- `llm` (object, optional) select an LLM provider + model (see below)

Response (truncated):

```json
{
  "run_id": "uuid",
  "artifacts_path": "/app/artifacts/<run_id>",
  "profile_path": "/app/profiles/default",
  "result": { "...": "..." },
  "live_view": {
    "vnc_host": "127.0.0.1",
    "novnc_url": "http://127.0.0.1:7900/vnc.html",
    "display": ":99"
  },
  "steps": [
    {
      "step_number": 1,
      "url": "https://example.com",
      "title": "Initial Actions",
      "action": [{ "navigate": { "url": "https://example.com", "new_tab": false } }],
      "screenshot_file": "screenshots/step_1.png"
    }
  ]
}
```

## Steps and artifacts

## LLM selection

By default the runner uses Browser Use hosted models (`provider=browseruse`).
To use other providers supported by your installed `browser_use` version, pass
an `llm` object in `/run` or `/jobs`:

```json
{
  "task": "Open https://example.com and report the title.",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1"
  }
}
```

`base_url` is optional and only used when supported by the selected provider.
API keys must be provided via environment variables (see `.env.example`), never
in the request payload.

### GET /runs/{run_id}/steps

Returns the saved step list for a run. Always includes `screenshot_file` when present.

### GET /runs/{run_id}/steps/{step_number}/screenshot

Returns `image/png` bytes for a step screenshot.

- `200` with `content-type: image/png` if the file exists.
- `404` if the screenshot is missing.

### GET /runs/{run_id}/report

Returns the generated HTML report (`text/html`).

## Jobs (async)

### POST /jobs

Submits an async job (same body as `/run`).

### GET /jobs/{run_id}

Returns job status and, when complete, the same payload as `/run`.

### POST /jobs/{run_id}/cancel

Best-effort cancel. If a job is queued, it will be removed from the queue.

## Human-in-the-loop control

Use these endpoints to pause a running agent, take over manually (via noVNC),
then resume with extra guidance.

### Auto-pause (agent requests help)

`/run` and `/jobs` accept an optional `hitl` object to pause automatically on
specific actions, URL patterns, or error keywords.

Example (pause on login/captcha errors):

```json
{
  "task": "Open https://example.com and log in.",
  "interactive": true,
  "hitl": {
    "mode": "auto"
  }
}
```

Example (pause when a submit/click happens):

```json
{
  "task": "Fill out the form and wait for my approval.",
  "interactive": true,
  "hitl": {
    "mode": "manual",
    "pause_on_action_types": ["click", "input"]
  }
}
```

When auto-pause triggers, `GET /runs/{run_id}/status` and `GET /jobs/{run_id}`
include `paused`, `pause_reason`, and `pause_message`.

### GET /runs/{run_id}/status

Returns status for a run or job. Example:

```json
{
  "run_id": "uuid",
  "status": "running",
  "paused": false,
  "step_number": 3
}
```

### POST /runs/{run_id}/pause

Pauses the active run (only works while the run is executing).

### POST /runs/{run_id}/resume

Resumes a paused run. Optionally include extra instructions:

```json
{
  "text": "Use the company login on the top-right before continuing."
}
```

If `text` is provided, it is appended as a follow-up user request before resuming.

## Live view (noVNC/VNC)

To view the live browser while a run is executing:

1) Call `/run` with:
   - `interactive=true`
   - `keep_open_seconds` set to a short window (e.g. 20)
2) Open the noVNC URL from the response:
   - `http://127.0.0.1:7900/vnc.html`
3) If connecting from another machine, use SSH port forwarding:

```
ssh -L 7900:127.0.0.1:7900 pi@<your-pi-host>
```

## Maintenance and profiles

### POST /maintenance/cleanup

Deletes old run artifacts under `/app/artifacts`.

Defaults:
- `ARTIFACTS_MAX_DAYS=7`
- `ARTIFACTS_MAX_RUNS=100`

### GET /profiles

Lists available profiles.

### POST /profiles/{name}/reset

Deletes a profile directory.

### POST /profiles/{name}/clone

Clones a profile. Body: `{"to":"newname"}`.

## Health

`GET /health` returns `{ "status": "ok" }`.

## Errors

- `401 Unauthorized`: missing or invalid `X-API-Key`
- `409 Profile is in use`: Chrome is already running in the container
- `429 Runner is busy`: another run is in progress
- `404`: run/step/screenshot not found

## n8n usage

1) Node A: HTTP Request (POST)
   - URL: `/run`
   - JSON body: `include_steps=true`, `include_step_screenshots="paths"`
2) Node B: Split Out Items on `response.steps`
3) Node C: HTTP Request (GET)
   - URL: `/runs/{{$json.run_id}}/steps/{{$json.step_number}}/screenshot`
   - Enable "Download" to store the PNG in the binary output

Each item will carry the step metadata and a binary screenshot for preview in n8n.
