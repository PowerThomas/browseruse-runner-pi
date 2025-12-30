import asyncio
import html
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from browser_use import Agent, ChatBrowserUse
from browser_use.browser.profile import BrowserProfile
from browser_use.agent.views import AgentHistoryList

RUNNER_API_KEY = os.environ.get("RUNNER_API_KEY")
BROWSER_USE_API_KEY = os.environ.get("BROWSER_USE_API_KEY")

PROFILES_DIR = Path(os.environ.get("PROFILES_DIR", "/app/profiles"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))

PROFILES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="browseruse-runner", version="0.1.0")
_lock = asyncio.Lock()
_job_tasks: Dict[str, asyncio.Task] = {}
_job_lock_owner: Optional[str] = None
_job_active_agents: Dict[str, Dict[str, Optional[Agent]]] = {}
_job_canceled: set[str] = set()
_job_queue: list[Dict[str, Any]] = []
_active_run_id: Optional[str] = None
_active_run_agent_ref: Optional[Dict[str, Optional[Agent]]] = None
_hitl_pause_info: Dict[str, Dict[str, str]] = {}


class RunRequest(BaseModel):
    task: str = Field(..., description="High-level instruction for Browser Use to accomplish.")
    url: Optional[str] = Field(
        None, description="Optional starting URL to open before executing the task."
    )
    profile_id: str = Field(
        default="default",
        description="Persistent profile bucket; keeps cookies/logins alive between runs.",
    )
    record_trace: bool = Field(
        default=True,
        description="Record Playwright trace for the run (stored in artifacts).",
    )
    interactive: bool = Field(
        default=False,
        description="Launch a headed browser for live viewing via VNC/noVNC.",
    )
    keep_open_seconds: int = Field(
        default=0,
        ge=0,
        description="Keep the browser open after completion for live viewing.",
    )
    include_steps: bool = Field(
        default=True,
        description="Include step metadata in the response payload.",
    )
    include_step_screenshots: Literal["none", "paths"] = Field(
        default="none",
        description="Include step screenshot file paths (none|paths).",
    )
    hitl: Optional["HitlConfig"] = None


class StepInfo(BaseModel):
    step_number: int
    url: Optional[str]
    title: Optional[str]
    action: Optional[Any]
    screenshot_file: Optional[str]


class ProfileCloneRequest(BaseModel):
    to: str = Field(..., description="Target profile name.")


class RunResponse(BaseModel):
    run_id: str
    artifacts_path: str
    profile_path: str
    result: Dict[str, Any]
    steps: Optional[list[StepInfo]] = None
    live_view: Optional[Dict[str, str]] = None


class HitlConfig(BaseModel):
    mode: Literal["off", "manual", "auto"] = Field(
        default="off",
        description="HITL mode: off, manual (no auto-pause), auto (pause on error patterns).",
    )
    pause_on_action_types: Optional[list[str]] = Field(
        default=None,
        description="Pause after steps that include any of these action types.",
    )
    pause_on_url_contains: Optional[list[str]] = Field(
        default=None,
        description="Pause if the current URL contains any of these substrings.",
    )
    pause_on_error_substrings: Optional[list[str]] = Field(
        default=None,
        description="Pause if an action error contains any of these substrings.",
    )


class HumanInputRequest(BaseModel):
    text: str = Field(..., description="Additional user guidance for the agent.")


RunRequest.model_rebuild()


async def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if not RUNNER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Runner API key not configured",
        )
    if not x_api_key or x_api_key != RUNNER_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _get_run_dir(run_id: str) -> Path:
    run_dir = (ARTIFACTS_DIR / run_id).resolve()
    artifacts_root = ARTIFACTS_DIR.resolve()
    if not run_dir.is_relative_to(artifacts_root):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run id")
    if not run_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run_dir


def _get_profile_dir(profile_name: str) -> Path:
    profile_dir = (PROFILES_DIR / profile_name).resolve()
    profiles_root = PROFILES_DIR.resolve()
    if not profile_dir.is_relative_to(profiles_root):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid profile name")
    return profile_dir


def _screenshot_relative_path(step_number: int) -> str:
    return f"screenshots/step_{step_number}.png"


def _screenshot_abs_path(artifacts_path: Path, step_number: int) -> Path:
    return artifacts_path / _screenshot_relative_path(step_number)


def _copy_screenshot(source_path: Path, destination_path: Path) -> bool:
    if not source_path.exists():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return True


async def _capture_screenshot(agent: Agent, destination_path: Path) -> bool:
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        await agent.browser_session.take_screenshot(path=str(destination_path))
        return True
    except Exception:  # noqa: BLE001
        return False


async def _persist_step_screenshot(
    agent: Optional[Agent],
    history_item: Any,
    artifacts_path: Path,
    step_number: int,
    step_screenshots: Dict[int, Optional[str]],
) -> None:
    if step_number in step_screenshots and step_screenshots[step_number]:
        return

    destination_path = _screenshot_abs_path(artifacts_path, step_number)
    source_path = None
    if getattr(history_item, "state", None) and getattr(history_item.state, "screenshot_path", None):
        source_path = Path(history_item.state.screenshot_path)

    if source_path and _copy_screenshot(source_path, destination_path):
        step_screenshots[step_number] = _screenshot_relative_path(step_number)
        return

    if agent and await _capture_screenshot(agent, destination_path):
        step_screenshots[step_number] = _screenshot_relative_path(step_number)
    else:
        step_screenshots[step_number] = None


def _build_steps(
    history: AgentHistoryList,
    step_screenshots: Dict[int, Optional[str]],
    include_step_screenshots: bool,
) -> list[StepInfo]:
    steps: list[StepInfo] = []
    for idx, item in enumerate(history.history, start=1):
        action = None
        if item.model_output and item.model_output.action:
            action = [a.model_dump(exclude_none=True, mode="json") for a in item.model_output.action]
        screenshot_file = step_screenshots.get(idx) if include_step_screenshots else None
        steps.append(
            StepInfo(
                step_number=idx,
                url=item.state.url if item.state else None,
                title=item.state.title if item.state else None,
                action=action,
                screenshot_file=screenshot_file,
            )
        )
    return steps


def _write_steps_file(artifacts_path: Path, steps: list[StepInfo]) -> None:
    steps_path = artifacts_path / "steps.json"
    steps_path.write_text(json.dumps([step.model_dump() for step in steps], indent=2))


def _summarize_action(action: Optional[Any]) -> str:
    if not action:
        return ""
    try:
        return json.dumps(action, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(action)


def _write_report(artifacts_path: Path, run_id: str, steps: list[StepInfo]) -> None:
    report_path = artifacts_path / "report.html"
    items = []
    for step in steps:
        action_summary = _summarize_action(step.action)
        screenshot_url = (
            f"/runs/{run_id}/steps/{step.step_number}/screenshot"
            if step.screenshot_file
            else ""
        )
        screenshot_html = (
            f'<img src="{screenshot_url}" alt="step {step.step_number} screenshot" />'
            if screenshot_url
            else '<div class="no-shot">No screenshot</div>'
        )
        items.append(
            f"""
            <div class="step">
              <div class="step-header">
                <div class="step-number">Step {step.step_number}</div>
                <div class="step-meta">
                  <div class="step-title">{html.escape(step.title or '')}</div>
                  <div class="step-url">{html.escape(step.url or '')}</div>
                </div>
              </div>
              <div class="step-body">
                <pre class="step-action">{html.escape(action_summary)}</pre>
                {screenshot_html}
              </div>
            </div>
            """
        )

    html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Run Report {run_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ font-size: 20px; margin-bottom: 16px; }}
    .step {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
    .step-header {{ display: flex; gap: 12px; align-items: center; }}
    .step-number {{ font-weight: bold; font-size: 14px; }}
    .step-title {{ font-weight: 600; }}
    .step-url {{ font-size: 12px; color: #555; word-break: break-all; }}
    .step-body {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 8px; }}
    .step-action {{ background: #f7f7f7; padding: 8px; border-radius: 6px; overflow-x: auto; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
    .no-shot {{ color: #888; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Run Report {run_id}</h1>
  {items}
</body>
</html>
""".format(
        run_id=html.escape(run_id),
        items="".join(items) if items else "<div>No steps available.</div>",
    )
    report_path.write_text(html_doc)


def _is_chrome_running() -> bool:
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.isdigit():
            continue
        comm_path = entry / "comm"
        cmd_path = entry / "cmdline"
        try:
            comm = comm_path.read_text(errors="ignore").strip().lower()
        except Exception:  # noqa: BLE001
            comm = ""
        if comm in {"chrome", "chromium", "chromium-browser", "headless_shell"}:
            return True
        try:
            cmdline = cmd_path.read_bytes().decode(errors="ignore").lower()
        except Exception:  # noqa: BLE001
            cmdline = ""
        if any(token in cmdline for token in ("chromium", "chrome", "headless_shell")):
            return True
    return False


def _clear_profile_locks(profile_path: Path) -> None:
    lock_names = ("SingletonLock", "SingletonCookie", "SingletonSocket")
    for name in lock_names:
        lock_path = profile_path / name
        try:
            if lock_path.is_symlink() or lock_path.exists():
                lock_path.unlink()
        except Exception:  # noqa: BLE001
            pass


def _get_retention_policy() -> tuple[int, int]:
    max_days = int(os.environ.get("ARTIFACTS_MAX_DAYS", "7"))
    max_runs = int(os.environ.get("ARTIFACTS_MAX_RUNS", "100"))
    return max_days, max_runs


def _cleanup_artifacts() -> tuple[int, int]:
    max_days, max_runs = _get_retention_policy()
    now = time.time()
    max_age_seconds = max_days * 86400 if max_days >= 0 else 0

    run_dirs = []
    for entry in ARTIFACTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        run_dirs.append((entry, mtime))

    run_dirs.sort(key=lambda item: item[1], reverse=True)
    keep = set()

    for idx, (path, mtime) in enumerate(run_dirs):
        if idx < max_runs:
            keep.add(path)
            continue
        if max_age_seconds and now - mtime <= max_age_seconds:
            keep.add(path)

    deleted = 0
    for path, _ in run_dirs:
        if path in keep:
            continue
        try:
            shutil.rmtree(path)
            deleted += 1
        except OSError:
            continue

    kept = len(run_dirs) - deleted
    return deleted, kept


def _profile_size_bytes(profile_dir: Path) -> Optional[int]:
    total = 0
    try:
        for root, _, files in os.walk(profile_dir):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
        return total
    except OSError:
        return None


def _get_active_agent(run_id: str) -> Optional[Agent]:
    if _active_run_id == run_id and _active_run_agent_ref:
        return _active_run_agent_ref.get("agent")
    agent_ref = _job_active_agents.get(run_id)
    if agent_ref:
        return agent_ref.get("agent")
    return None


def _get_run_status(run_id: str) -> Optional[str]:
    if _active_run_id == run_id:
        return "running"
    payload = _read_job_status(run_id)
    if payload:
        return payload.get("status")
    return None


def _extract_action_types(history_item: Any) -> list[str]:
    action_types: list[str] = []
    model_output = getattr(history_item, "model_output", None)
    if not model_output or not getattr(model_output, "action", None):
        return action_types
    for action in model_output.action:
        try:
            payload = action.model_dump(exclude_none=True, mode="json")
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        for key in payload.keys():
            action_types.append(key)
    return action_types


def _extract_errors(history_item: Any) -> list[str]:
    errors: list[str] = []
    for result in getattr(history_item, "result", []) or []:
        error = getattr(result, "error", None)
        if error:
            errors.append(str(error))
    return errors


def _should_pause_hitl(req: RunRequest, history_item: Any) -> Optional[Dict[str, str]]:
    hitl = req.hitl
    if not hitl or hitl.mode == "off":
        return None

    url = None
    if getattr(history_item, "state", None):
        url = getattr(history_item.state, "url", None)

    if hitl.pause_on_url_contains and url:
        for token in hitl.pause_on_url_contains:
            if token and token in url:
                return {"reason": "url_match", "message": f"Paused on URL match: {token}"}

    action_types = _extract_action_types(history_item)
    if hitl.pause_on_action_types:
        for action_type in action_types:
            if action_type in hitl.pause_on_action_types:
                return {"reason": "action_match", "message": f"Paused on action: {action_type}"}

    error_substrings = hitl.pause_on_error_substrings
    if hitl.mode == "auto" and not error_substrings:
        error_substrings = [
            "login",
            "sign in",
            "captcha",
            "credential",
            "2fa",
            "two factor",
            "otp",
            "verification code",
            "authorize",
            "approval",
            "permission",
        ]
    if error_substrings:
        for error in _extract_errors(history_item):
            lowered = error.lower()
            for token in error_substrings:
                if token and token.lower() in lowered:
                    return {"reason": "error_match", "message": f"Paused on error: {token}"}

    return None


def _job_file_path(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id / "job.json"


def _write_job_status(
    run_id: str,
    status_value: str,
    response: Optional[RunResponse] = None,
    error: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "status": status_value,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if response is not None:
        payload["response"] = response.model_dump()
    if error:
        payload["error"] = error
    job_path = _job_file_path(run_id)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(payload, indent=2))


def _read_job_status(run_id: str) -> Optional[Dict[str, Any]]:
    job_path = _job_file_path(run_id)
    if not job_path.exists():
        return None
    return json.loads(job_path.read_text())


def _prepare_run_paths(req: RunRequest, run_id: str) -> tuple[Path, Path]:
    profile_path = PROFILES_DIR / req.profile_id
    artifacts_path = ARTIFACTS_DIR / run_id
    if _is_chrome_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile is in use; chrome is running in the container.",
        )
    _clear_profile_locks(profile_path)
    return profile_path, artifacts_path


async def _build_run_response(
    req: RunRequest,
    run_id: str,
    profile_path: Path,
    artifacts_path: Path,
    active_agent_ref: Optional[Dict[str, Optional[Agent]]] = None,
) -> RunResponse:
    result, history, step_screenshots = await _run_browseruse_task(
        req, run_id, profile_path, artifacts_path, active_agent_ref=active_agent_ref
    )
    steps: Optional[list[StepInfo]] = None
    live_view: Optional[Dict[str, str]] = None
    if history:
        steps_with_paths = _build_steps(history, step_screenshots, include_step_screenshots=True)
        _write_steps_file(artifacts_path, steps_with_paths)
        _write_report(artifacts_path, run_id, steps_with_paths)
        if req.include_steps:
            include_paths = req.include_step_screenshots == "paths"
            steps = _build_steps(history, step_screenshots, include_step_screenshots=include_paths)
    else:
        _write_steps_file(artifacts_path, [])
        if req.include_steps:
            steps = []
    if req.interactive:
        live_view = {
            "vnc_host": "127.0.0.1",
            "novnc_url": "http://127.0.0.1:7900/vnc.html",
            "display": ":99",
        }
    return RunResponse(
        run_id=run_id,
        artifacts_path=str(artifacts_path),
        profile_path=str(profile_path),
        result=result,
        steps=steps,
        live_view=live_view,
    )


async def _start_job(run_id: str, req: RunRequest) -> None:
    global _job_lock_owner
    try:
        profile_path, artifacts_path = _prepare_run_paths(req, run_id)
    except HTTPException as exc:
        _write_job_status(run_id, "failed", error=str(exc.detail))
        return
    await _lock.acquire()
    _job_lock_owner = run_id
    task = asyncio.create_task(_run_job_task(run_id, req, profile_path, artifacts_path))
    _job_tasks[run_id] = task


def _dequeue_job(run_id: str) -> bool:
    for idx, job in enumerate(_job_queue):
        if job["run_id"] == run_id:
            _job_queue.pop(idx)
            return True
    return False


async def _start_next_job_if_available() -> None:
    if _lock.locked() or not _job_queue:
        return
    job = _job_queue.pop(0)
    await _start_job(job["run_id"], job["req"])


async def _run_job_task(
    run_id: str,
    req: RunRequest,
    profile_path: Path,
    artifacts_path: Path,
) -> None:
    global _job_lock_owner
    try:
        _write_job_status(run_id, "running")
        agent_ref: Dict[str, Optional[Agent]] = {"agent": None}
        _job_active_agents[run_id] = agent_ref
        response = await _build_run_response(req, run_id, profile_path, artifacts_path, active_agent_ref=agent_ref)
        if run_id not in _job_canceled:
            _write_job_status(run_id, "completed", response=response)
    except asyncio.CancelledError:
        if run_id not in _job_canceled:
            _write_job_status(run_id, "canceled")
        raise
    except Exception as exc:  # noqa: BLE001
        if run_id not in _job_canceled:
            _write_job_status(run_id, "failed", error=str(exc))
    finally:
        _job_active_agents.pop(run_id, None)
        task = _job_tasks.pop(run_id, None)
        if task and task.cancelled():
            _job_canceled.discard(run_id)
        _hitl_pause_info.pop(run_id, None)
        if _job_lock_owner == run_id and _lock.locked():
            _lock.release()
            _job_lock_owner = None
        await _start_next_job_if_available()


async def _run_browseruse_task(
    req: RunRequest,
    run_id: str,
    profile_path: Path,
    artifacts_path: Path,
    active_agent_ref: Optional[Dict[str, Optional[Agent]]] = None,
) -> tuple[Dict[str, Any], Optional[AgentHistoryList], Dict[int, Optional[str]]]:
    """
    Execute a Browser Use Agent run using the hosted LLM.
    Falls back to a light Playwright warmup when Browser Use is unavailable so we still
    produce trace/video artifacts.
    """
    artifacts_path.mkdir(parents=True, exist_ok=True)
    profile_path.mkdir(parents=True, exist_ok=True)

    if not BROWSER_USE_API_KEY:
        return {"error": "BROWSER_USE_API_KEY is not set"}, None, {}

    # Lazy import to keep startup fast and allow easier troubleshooting.
    try:
        video_dir = artifacts_path / "video"
        trace_dir = artifacts_path / "trace"
        downloads_dir = profile_path / "downloads"
        for d in (video_dir, trace_dir, downloads_dir):
            d.mkdir(parents=True, exist_ok=True)
        profile_path.mkdir(parents=True, exist_ok=True)

        llm = ChatBrowserUse()

        if req.interactive:
            os.environ.setdefault("DISPLAY", ":99")

        profile = BrowserProfile(
            user_data_dir=str(profile_path),
            traces_dir=str(trace_dir),
            record_video_dir=str(video_dir),
            downloads_path=str(downloads_dir),
            headless=not req.interactive,
            enable_default_extensions=False,
            chromium_sandbox=False,
            disable_security=False,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
            keep_alive=req.keep_open_seconds > 0,
        )

        task = req.task if req.url is None else f"{req.task}\nStart URL: {req.url}"
        agent = Agent(task=task, llm=llm, browser_profile=profile)
        if active_agent_ref is not None:
            active_agent_ref["agent"] = agent
        step_screenshots: Dict[int, Optional[str]] = {}

        async def on_step_end(agent_instance: Agent) -> None:
            if not agent_instance.history.history:
                return
            step_number = len(agent_instance.history.history)
            await _persist_step_screenshot(
                agent=agent_instance,
                history_item=agent_instance.history.history[-1],
                artifacts_path=artifacts_path,
                step_number=step_number,
                step_screenshots=step_screenshots,
            )
            pause_info = _should_pause_hitl(req, agent_instance.history.history[-1])
            if pause_info:
                _hitl_pause_info[run_id] = pause_info
                agent_instance.pause()

        try:
            result = await agent.run(on_step_end=on_step_end)
            if req.keep_open_seconds > 0 and agent.browser_session is not None:
                await asyncio.sleep(req.keep_open_seconds)
                await agent.browser_session.kill()
            if isinstance(result, AgentHistoryList):
                for idx, item in enumerate(result.history, start=1):
                    if idx not in step_screenshots:
                        await _persist_step_screenshot(
                            agent=None,
                            history_item=item,
                            artifacts_path=artifacts_path,
                            step_number=idx,
                            step_screenshots=step_screenshots,
                        )
                return result.model_dump(), result, step_screenshots
            if hasattr(result, "model_dump"):
                return result.model_dump(), None, {}
            if hasattr(result, "to_dict"):
                return result.to_dict(), None, {}  # type: ignore[attr-defined]
            return {"result": str(result)}, None, {}
        finally:
            if active_agent_ref is not None:
                active_agent_ref["agent"] = None
    except Exception as exc:  # noqa: BLE001
        return {"error": f"browser_use execution failed: {exc}"}, None, {}


@app.post("/run", response_model=RunResponse)
async def run(
    req: RunRequest, _: None = Depends(verify_api_key)
) -> RunResponse:  # pragma: no cover - minimal service surface
    global _active_run_id, _active_run_agent_ref
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    async with _lock:
        run_id = str(uuid.uuid4())
        profile_path, artifacts_path = _prepare_run_paths(req, run_id)
        _active_run_id = run_id
        _active_run_agent_ref = {"agent": None}
        try:
            return await _build_run_response(
                req, run_id, profile_path, artifacts_path, active_agent_ref=_active_run_agent_ref
            )
        finally:
            _active_run_id = None
            _active_run_agent_ref = None
            _hitl_pause_info.pop(run_id, None)


@app.get("/runs/{run_id}/steps", response_model=list[StepInfo])
async def list_run_steps(run_id: str, _: None = Depends(verify_api_key)) -> list[StepInfo]:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    run_dir = _get_run_dir(run_id)
    steps_path = run_dir / "steps.json"
    if not steps_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Steps not found")
    steps_payload = json.loads(steps_path.read_text())
    return [StepInfo(**step) for step in steps_payload]


@app.get("/runs/{run_id}/steps/{step_number}/screenshot")
async def get_step_screenshot(
    run_id: str, step_number: int, _: None = Depends(verify_api_key)
) -> Response:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    run_dir = _get_run_dir(run_id)
    screenshot_path = run_dir / _screenshot_relative_path(step_number)
    if not screenshot_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screenshot not found")
    return FileResponse(screenshot_path, media_type="image/png")


@app.get("/runs/{run_id}/report")
async def get_run_report(run_id: str, _: None = Depends(verify_api_key)) -> Response:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    run_dir = _get_run_dir(run_id)
    report_path = run_dir / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return FileResponse(report_path, media_type="text/html")


@app.get("/runs/{run_id}/status")
async def get_run_status(run_id: str, _: None = Depends(verify_api_key)) -> Dict[str, Any]:
    status_value = _get_run_status(run_id)
    if status_value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    agent = _get_active_agent(run_id)
    payload: Dict[str, Any] = {"run_id": run_id, "status": status_value}
    if agent is not None:
        payload["paused"] = agent.state.paused
        payload["step_number"] = agent.state.n_steps
    pause_info = _hitl_pause_info.get(run_id)
    if pause_info:
        payload["pause_reason"] = pause_info.get("reason")
        payload["pause_message"] = pause_info.get("message")
    return payload


@app.post("/runs/{run_id}/pause")
async def pause_run(run_id: str, _: None = Depends(verify_api_key)) -> Dict[str, str]:
    status_value = _get_run_status(run_id)
    if status_value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if status_value in {"queued"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run is queued")
    if status_value in {"completed", "failed", "canceled"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run already finished")
    agent = _get_active_agent(run_id)
    if agent is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Agent is not ready")
    if not agent.state.paused:
        agent.pause()
    return {"run_id": run_id, "status": "paused"}


@app.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    req: Optional[HumanInputRequest] = None,
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    status_value = _get_run_status(run_id)
    if status_value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if status_value in {"queued"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run is queued")
    if status_value in {"completed", "failed", "canceled"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run already finished")
    agent = _get_active_agent(run_id)
    if agent is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Agent is not ready")
    if req and not agent.state.paused:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Agent is not paused")
    if req:
        agent.add_new_task(req.text)
    agent.resume()
    _hitl_pause_info.pop(run_id, None)
    return {"run_id": run_id, "status": "running"}


@app.post("/maintenance/cleanup")
async def cleanup_artifacts(_: None = Depends(verify_api_key)) -> Dict[str, int]:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    deleted, kept = _cleanup_artifacts()
    return {"deleted": deleted, "kept": kept}


@app.get("/profiles")
async def list_profiles(_: None = Depends(verify_api_key)) -> list[Dict[str, Any]]:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    profiles = []
    for entry in PROFILES_DIR.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = None
        profiles.append(
            {
                "name": entry.name,
                "last_modified": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                if mtime
                else None,
                "size_bytes": _profile_size_bytes(entry),
            }
        )
    return profiles


@app.post("/profiles/{name}/reset")
async def reset_profile(name: str, _: None = Depends(verify_api_key)) -> Dict[str, str]:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    if _is_chrome_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile is in use; chrome is running in the container.",
        )
    profile_dir = _get_profile_dir(name)
    if not profile_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    shutil.rmtree(profile_dir)
    return {"status": "reset", "profile": name}


@app.post("/profiles/{name}/clone")
async def clone_profile(
    name: str, req: ProfileCloneRequest, _: None = Depends(verify_api_key)
) -> Dict[str, str]:
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    if _is_chrome_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile is in use; chrome is running in the container.",
        )
    source_dir = _get_profile_dir(name)
    target_dir = _get_profile_dir(req.to)
    if not source_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    if target_dir.exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Target profile exists")
    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("SingletonLock", "SingletonCookie", "SingletonSocket"),
    )
    return {"status": "cloned", "from": name, "to": req.to}


@app.post("/jobs")
async def create_job(req: RunRequest, _: None = Depends(verify_api_key)) -> Dict[str, str]:
    run_id = str(uuid.uuid4())
    _write_job_status(run_id, "queued")
    if _lock.locked() or _job_queue:
        _job_queue.append({"run_id": run_id, "req": req})
        return {"run_id": run_id, "status": "queued"}
    await _start_job(run_id, req)
    return {"run_id": run_id, "status": "running"}


@app.get("/jobs/{run_id}")
async def get_job(run_id: str, _: None = Depends(verify_api_key)) -> Dict[str, Any]:
    payload = _read_job_status(run_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if payload.get("status") == "running":
        agent = _get_active_agent(run_id)
        if agent is not None:
            payload["paused"] = agent.state.paused
            pause_info = _hitl_pause_info.get(run_id)
            if pause_info:
                payload["pause_reason"] = pause_info.get("reason")
                payload["pause_message"] = pause_info.get("message")
    return payload


@app.post("/jobs/{run_id}/cancel")
async def cancel_job(run_id: str, _: None = Depends(verify_api_key)) -> Dict[str, str]:
    global _job_lock_owner
    payload = _read_job_status(run_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if payload.get("status") in {"completed", "failed", "canceled"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job already finished")
    if payload.get("status") == "queued" and _dequeue_job(run_id):
        _job_canceled.add(run_id)
        _write_job_status(run_id, "canceled")
        return {"status": "canceled"}
    _job_canceled.add(run_id)
    _write_job_status(run_id, "canceled")
    task = _job_tasks.get(run_id)
    if task and not task.done():
        task.cancel()
    agent_ref = _job_active_agents.get(run_id)
    agent = agent_ref.get("agent") if agent_ref else None
    if agent and agent.browser_session:
        try:
            await agent.browser_session.kill()
        except Exception:  # noqa: BLE001
            pass
    if _job_lock_owner == run_id and _lock.locked():
        _lock.release()
        _job_lock_owner = None
    return {"status": "canceled"}
