import json
import os
import time
import unittest
import urllib.error
import urllib.request
import threading


BASE_URL = os.environ.get("RUNNER_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("RUNNER_API_KEY")
REAL_SITE_TESTS = os.environ.get("REAL_SITE_TESTS", "")
REQUEST_TIMEOUT = int(os.environ.get("RUNNER_REQUEST_TIMEOUT", "60"))
REAL_SITE_TIMEOUT = int(os.environ.get("REAL_SITE_TIMEOUT", "180"))


def _request(method, path, body=None, headers=None, timeout=None):
    url = f"{BASE_URL}{path}"
    req_headers = {"X-API-Key": API_KEY} if API_KEY else {}
    if headers:
        req_headers.update(headers)
    data = None
    if body is not None:
        data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or REQUEST_TIMEOUT) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _post_json(path, payload, timeout=None):
    body = json.dumps(payload)
    return _request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


class RunnerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not API_KEY:
            raise unittest.SkipTest("RUNNER_API_KEY not set")

    def test_health(self):
        status, _, body = _request("GET", "/health")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload.get("status"), "ok")

    def test_profiles_list(self):
        status, _, body = _request("GET", "/profiles")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertIsInstance(payload, list)
        names = {item.get("name") for item in payload if isinstance(item, dict)}
        self.assertIn("default", names)

    def test_llms_list(self):
        status, _, body = _request("GET", "/llms")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        providers = payload.get("providers", [])
        self.assertTrue(any(p.get("provider") == "google" for p in providers))

    def test_run_steps_report(self):
        status, _, raw = _post_json(
            "/run",
            {
                "task": "Open https://example.com and report the title.",
                "url": "https://example.com",
                "include_steps": True,
                "include_step_screenshots": "paths",
            },
        )
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        run_id = payload["run_id"]
        steps = payload.get("steps") or []
        self.assertTrue(len(steps) >= 1)

        status, _, raw = _request("GET", f"/runs/{run_id}/steps")
        self.assertEqual(status, 200)
        step_list = json.loads(raw.decode("utf-8"))
        self.assertIsInstance(step_list, list)

        shot_step = next((s for s in step_list if s.get("screenshot_file")), None)
        if shot_step:
            step_number = shot_step["step_number"]
            status, headers, data = _request("GET", f"/runs/{run_id}/steps/{step_number}/screenshot")
            self.assertEqual(status, 200)
            self.assertTrue(headers.get("Content-Type", "").startswith("image/png"))
            self.assertGreater(len(data), 0)

        status, headers, data = _request("GET", f"/runs/{run_id}/report")
        self.assertEqual(status, 200)
        self.assertTrue(headers.get("Content-Type", "").startswith("text/html"))
        self.assertIn(b"<html", data.lower())

    def test_interactive_run_live_view(self):
        status, _, raw = _post_json(
            "/run",
            {
                "task": "Open https://example.com and report the title.",
                "url": "https://example.com",
                "interactive": True,
                "keep_open_seconds": 1,
                "include_steps": False,
            },
        )
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        live_view = payload.get("live_view")
        self.assertIsInstance(live_view, dict)
        self.assertEqual(live_view.get("display"), ":99")
        self.assertIn("novnc_url", live_view)

    def test_run_busy_when_parallel(self):
        results = {}

        def _run_long():
            status, _, raw = _post_json(
                "/run",
                {
                    "task": "Open https://example.com and wait.",
                    "url": "https://example.com",
                    "keep_open_seconds": 5,
                    "include_steps": False,
                },
            )
            results["status"] = status
            results["raw"] = raw

        thread = threading.Thread(target=_run_long, daemon=True)
        thread.start()
        time.sleep(0.5)

        busy = False
        deadline = time.time() + 6
        while time.time() < deadline and not busy:
            status, _, _ = _post_json(
                "/run",
                {
                    "task": "Open https://example.com and report the title.",
                    "url": "https://example.com",
                    "include_steps": False,
                },
            )
            busy = status == 429
            if not busy:
                time.sleep(0.5)

        thread.join(timeout=30)
        self.assertEqual(results.get("status"), 200)
        self.assertTrue(busy, "expected 429 when another run is active")

    def test_jobs(self):
        status, _, raw = _post_json(
            "/jobs",
            {
                "task": "Open https://example.com and report the title.",
                "url": "https://example.com",
                "include_steps": True,
                "include_step_screenshots": "paths",
            },
        )
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        run_id = payload["run_id"]

        deadline = time.time() + 120
        while time.time() < deadline:
            status, _, raw = _request("GET", f"/jobs/{run_id}")
            self.assertEqual(status, 200)
            job = json.loads(raw.decode("utf-8"))
            if job.get("status") in {"completed", "failed", "canceled"}:
                break
            time.sleep(2)
        else:
            self.fail("job did not finish before timeout")

        self.assertEqual(job.get("status"), "completed")
        response = job.get("response") or {}
        self.assertEqual(response.get("run_id"), run_id)

    def test_llm_requires_model(self):
        status, _, _ = _post_json(
            "/run",
            {
                "task": "Open https://example.com and report the title.",
                "llm": {"provider": "openai"},
                "include_steps": False,
            },
        )
        self.assertEqual(status, 422)

    def test_jobs_busy_when_parallel(self):
        status, _, raw = _post_json(
            "/jobs",
            {
                "task": "Open https://example.com and report the title.",
                "url": "https://example.com",
                "include_steps": False,
            },
        )
        self.assertEqual(status, 200)
        first = json.loads(raw.decode("utf-8"))
        first_id = first["run_id"]

        status, _, raw = _post_json(
            "/jobs",
            {
                "task": "Open https://example.com and report the title.",
                "url": "https://example.com",
                "include_steps": False,
            },
        )
        self.assertEqual(status, 200)
        second = json.loads(raw.decode("utf-8"))
        second_id = second["run_id"]
        self.assertEqual(second.get("status"), "queued")

        deadline = time.time() + 180
        finished = set()
        while time.time() < deadline and len(finished) < 2:
            for run_id in (first_id, second_id):
                if run_id in finished:
                    continue
                status, _, raw = _request("GET", f"/jobs/{run_id}")
                self.assertEqual(status, 200)
                job = json.loads(raw.decode("utf-8"))
                if job.get("status") in {"completed", "failed", "canceled"}:
                    finished.add(run_id)
            time.sleep(2)
        if len(finished) < 2:
            self.fail("queued jobs did not finish before timeout")

    def test_cleanup_optional(self):
        if os.environ.get("RUN_TEST_CLEANUP") != "1":
            self.skipTest("RUN_TEST_CLEANUP not set")
        old_dirs = []
        for idx in range(2):
            run_id = f"test-old-{int(time.time())}-{idx}"
            path = f"/app/artifacts/{run_id}"
            os.makedirs(path, exist_ok=True)
            old_time = time.time() - 10 * 86400
            os.utime(path, (old_time, old_time))
            old_dirs.append(path)
        status, _, raw = _request("POST", "/maintenance/cleanup", body="")
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertGreaterEqual(payload.get("deleted", 0), 1)
        for path in old_dirs:
            self.assertFalse(os.path.exists(path))

    def test_profile_clone_reset_optional(self):
        if os.environ.get("RUN_TEST_PROFILE_MUTATION") != "1":
            self.skipTest("RUN_TEST_PROFILE_MUTATION not set")
        profile_name = f"test-clone-{int(time.time())}"
        body = json.dumps({"to": profile_name})
        status, _, raw = _request(
            "POST",
            "/profiles/default/clone",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload.get("to"), profile_name)

        status, _, raw = _request("POST", f"/profiles/{profile_name}/reset", body="")
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload.get("profile"), profile_name)

    def test_human_in_loop_optional(self):
        if os.environ.get("RUN_TEST_HITL") != "1":
            self.skipTest("RUN_TEST_HITL not set")

        status, _, raw = _post_json(
            "/jobs",
            {
                "task": "Open https://example.com, scroll slowly, then report the title.",
                "url": "https://example.com",
                "include_steps": False,
            },
        )
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        run_id = payload["run_id"]

        deadline = time.time() + 60
        while time.time() < deadline:
            status, _, raw = _request("GET", f"/jobs/{run_id}")
            self.assertEqual(status, 200)
            job = json.loads(raw.decode("utf-8"))
            if job.get("status") == "running":
                break
            if job.get("status") in {"completed", "failed", "canceled"}:
                self.skipTest("job finished before pause could be issued")
            time.sleep(1)
        else:
            self.fail("job did not reach running state before timeout")

        pause_status, _, _ = _request("POST", f"/runs/{run_id}/pause", body="")
        if pause_status == 409:
            self.skipTest("agent not ready for pause in time")
        self.assertEqual(pause_status, 200)

        resume_body = json.dumps({"text": "Continue the task after the manual pause."})
        resume_status, _, _ = _request(
            "POST",
            f"/runs/{run_id}/resume",
            body=resume_body,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resume_status, 200)

        deadline = time.time() + 180
        while time.time() < deadline:
            status, _, raw = _request("GET", f"/jobs/{run_id}")
            self.assertEqual(status, 200)
            job = json.loads(raw.decode("utf-8"))
            if job.get("status") in {"completed", "failed", "canceled"}:
                break
            time.sleep(2)
        else:
            self.fail("job did not finish after resume")

    def test_human_in_loop_auto_pause_optional(self):
        if os.environ.get("RUN_TEST_HITL_AUTO") != "1":
            self.skipTest("RUN_TEST_HITL_AUTO not set")

        status, _, raw = _post_json(
            "/jobs",
            {
                "task": "Open https://example.com and report the title.",
                "url": "https://example.com",
                "include_steps": False,
                "hitl": {
                    "mode": "manual",
                    "pause_on_action_types": ["navigate"],
                },
            },
        )
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        run_id = payload["run_id"]

        deadline = time.time() + 60
        paused = False
        while time.time() < deadline and not paused:
            status, _, raw = _request("GET", f"/runs/{run_id}/status")
            self.assertEqual(status, 200)
            run_status = json.loads(raw.decode("utf-8"))
            paused = run_status.get("paused") is True
            if not paused:
                time.sleep(1)
        if not paused:
            self.fail("run did not auto-pause before timeout")

        self.assertIn("pause_reason", run_status)
        resume_body = json.dumps({"text": "Continue after auto-pause."})
        resume_status, _, _ = _request(
            "POST",
            f"/runs/{run_id}/resume",
            body=resume_body,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resume_status, 200)

        deadline = time.time() + 180
        while time.time() < deadline:
            status, _, raw = _request("GET", f"/jobs/{run_id}")
            self.assertEqual(status, 200)
            job = json.loads(raw.decode("utf-8"))
            if job.get("status") in {"completed", "failed", "canceled"}:
                break
            time.sleep(2)
        else:
            self.fail("job did not finish after auto-pause resume")

    def test_real_sites_optional(self):
        if os.environ.get("RUN_TEST_REAL_SITES") != "1":
            self.skipTest("RUN_TEST_REAL_SITES not set")
        if not REAL_SITE_TESTS:
            self.skipTest("REAL_SITE_TESTS not set")
        try:
            sites = json.loads(REAL_SITE_TESTS)
        except json.JSONDecodeError as exc:
            self.fail(f"REAL_SITE_TESTS must be JSON: {exc}")
        if not isinstance(sites, list) or not sites:
            self.fail("REAL_SITE_TESTS must be a non-empty list")

        for entry in sites:
            if not isinstance(entry, dict):
                self.fail("REAL_SITE_TESTS entries must be objects")
            url = entry.get("url")
            task = entry.get("task")
            if not url or not task:
                self.fail("REAL_SITE_TESTS entries require url and task")

            status, _, raw = _post_json(
                "/run",
                {
                    "task": task,
                    "url": url,
                    "include_steps": True,
                    "include_step_screenshots": "paths",
                },
                timeout=REAL_SITE_TIMEOUT,
            )
            self.assertEqual(status, 200)
            payload = json.loads(raw.decode("utf-8"))
            steps = payload.get("steps") or []
            self.assertTrue(len(steps) >= 1)


if __name__ == "__main__":
    unittest.main()
