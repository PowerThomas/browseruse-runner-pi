# browseruse-runner API

This service runs Browser Use tasks behind an internal Docker network.
All endpoints require the `X-API-Key` header matching `RUNNER_API_KEY`.

Base URL (from n8n container): `http://browseruse-runner:8000`

## Auth

Add this header to every request:

```
X-API-Key: <RUNNER_API_KEY>
```

## Concurrency and profile safety

- If a run is already in progress, the API returns `429 Runner is busy`.
- If Chromium is running inside the container, the API returns `409 Profile is in use`
  and will not delete profile locks.

## POST /run

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
  "include_step_screenshots": "paths"
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

## GET /runs/{run_id}/steps

Returns the saved step list for a run. Always includes `screenshot_file` when present.

Example:

```json
[
  {
    "step_number": 2,
    "url": "https://example.com/",
    "title": "example.com",
    "action": [{ "done": { "text": "..." } }],
    "screenshot_file": "screenshots/step_2.png"
  }
]
```

## GET /runs/{run_id}/steps/{step_number}/screenshot

Returns `image/png` bytes for a step screenshot.

- `200` with `content-type: image/png` if the file exists.
- `404` if the screenshot is missing.

## GET /runs/{run_id}/report

Returns the generated HTML report for a run (`text/html`). The report uses
per-step screenshot URLs instead of embedding base64 images.

## POST /maintenance/cleanup

Cleans old artifacts under `/app/artifacts`.

Defaults:
- `ARTIFACTS_MAX_DAYS=7`
- `ARTIFACTS_MAX_RUNS=100`

Response:

```json
{"deleted": 5, "kept": 100}
```

## Profile management

- `GET /profiles` list profiles (name, last_modified, size_bytes)
- `POST /profiles/{name}/reset` delete a profile
- `POST /profiles/{name}/clone` with body `{"to":"newname"}`

If Chrome is running in the container, profile actions return `409`.

## Async jobs

`/jobs` lets you start a run asynchronously while keeping the single-run lock:

- `POST /jobs` with the same body as `/run`
- `GET /jobs/{run_id}` returns status + response when done
- `POST /jobs/{run_id}/cancel` best-effort cancel

## Health

`GET /health` returns `{ "status": "ok" }`.

## n8n usage

1) Node A: HTTP Request (POST)
   - URL: `/run`
   - JSON body: `include_steps=true`, `include_step_screenshots="paths"`
2) Node B: Split Out Items on `response.steps`
3) Node C: HTTP Request (GET)
   - URL: `/runs/{{$json.run_id}}/steps/{{$json.step_number}}/screenshot`
   - Enable "Download" to store the PNG in the binary output

Each item will carry the step metadata and a binary screenshot for preview in n8n.

To fetch the HTML report, add a HTTP Request node:
- `GET /runs/{{$json.run_id}}/report`

For async jobs, POST `/jobs` and poll `GET /jobs/{{$json.run_id}}` until `status` is `completed`.

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
