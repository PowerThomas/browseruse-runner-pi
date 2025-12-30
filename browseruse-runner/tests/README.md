# Tests

This folder contains smoke tests for the browseruse-runner API. The goal is to
exercise real endpoints (health, run, steps, report, jobs, profiles) from
outside the service so regressions are caught early.

All tests live in `test_api_smoke.py`.

## Test list

- `test_health`
  - Why: verify the service is up.
  - Does: `GET /health`.
  - Expects: HTTP 200 and `{ "status": "ok" }`.

- `test_profiles_list`
  - Why: ensure profile listing works and always includes `default`.
  - Does: `GET /profiles`.
  - Expects: HTTP 200 and list containing `default`.

- `test_run_steps_report`
  - Why: validate the main sync run flow and report generation.
  - Does: `POST /run`, then `GET /runs/{run_id}/steps`,
    `GET /runs/{run_id}/steps/{step}/screenshot` (if present),
    and `GET /runs/{run_id}/report`.
  - Expects: HTTP 200, steps list, optional PNG screenshot, and HTML report.

- `test_interactive_run_live_view`
  - Why: ensure interactive mode returns live view metadata.
  - Does: `POST /run` with `interactive=true` and `keep_open_seconds=1`.
  - Expects: HTTP 200 and `live_view` with `display=:99` and a noVNC URL.

- `test_run_busy_when_parallel`
  - Why: confirm /run enforces single concurrency.
  - Does: starts a run with `keep_open_seconds=5` then attempts another /run.
  - Expects: second call returns HTTP 429 while the first is active.

- `test_jobs`
  - Why: validate async job lifecycle.
  - Does: `POST /jobs`, then `GET /jobs/{run_id}` until finished.
  - Expects: job `status` becomes `completed` with a `response`.

- `test_jobs_busy_when_parallel`
  - Why: confirm jobs are queued (FIFO).
  - Does: `POST /jobs` twice; second should be queued; poll until both finish.
  - Expects: second job returns `status: queued`, both finish.

- `test_cleanup_optional` (optional)
  - Why: verify artifact cleanup deletes old runs.
  - Does: creates fake old run dirs under `/app/artifacts`, then
    `POST /maintenance/cleanup`.
  - Expects: HTTP 200 and old dirs removed.

- `test_profile_clone_reset_optional` (optional)
  - Why: verify profile clone/reset flows.
  - Does: clone default profile to a temp name, then reset it.
  - Expects: HTTP 200 for both operations.

## How to run

Run inside the container:

```
docker compose exec -T browseruse-runner python -m unittest /app/tests/test_api_smoke.py
```

## Configuration

Required env vars:
- `RUNNER_API_KEY` must be set in the container.

Optional env vars:
- `RUN_TEST_CLEANUP=1` enables `test_cleanup_optional`.
- `RUN_TEST_PROFILE_MUTATION=1` enables `test_profile_clone_reset_optional`.

Example:

```
docker compose exec -T browseruse-runner sh -c \
  'RUN_TEST_CLEANUP=1 RUN_TEST_PROFILE_MUTATION=1 python -m unittest /app/tests/test_api_smoke.py'
```
