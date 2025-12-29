import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
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
    include_steps: bool = Field(
        default=True,
        description="Include step metadata in the response payload.",
    )
    include_step_screenshots: Literal["none", "paths"] = Field(
        default="none",
        description="Include step screenshot file paths (none|paths).",
    )


class StepInfo(BaseModel):
    step_number: int
    url: Optional[str]
    title: Optional[str]
    action: Optional[Any]
    screenshot_file: Optional[str]


class RunResponse(BaseModel):
    run_id: str
    artifacts_path: str
    profile_path: str
    result: Dict[str, Any]
    steps: Optional[list[StepInfo]] = None


async def verify_api_key(x_api_key: str = Header(alias="X-API-Key")) -> None:
    if not RUNNER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Runner API key not configured",
        )
    if x_api_key != RUNNER_API_KEY:
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


async def _run_browseruse_task(
    req: RunRequest, profile_path: Path, artifacts_path: Path
) -> tuple[Dict[str, Any], Optional[AgentHistoryList], Dict[int, Optional[str]]]:
    """
    Execute a Browser Use Agent run using the hosted LLM.
    Falls back to a light Playwright warmup when Browser Use is unavailable so we still
    produce trace/video artifacts.
    """
    artifacts_path.mkdir(parents=True, exist_ok=True)
    profile_path.mkdir(parents=True, exist_ok=True)

    if not BROWSER_USE_API_KEY:
        fallback = await _playwright_fallback(req, profile_path, artifacts_path)
        return {"error": "BROWSER_USE_API_KEY is not set", "playwright": fallback}, None, {}

    # Lazy import to keep startup fast and allow easier troubleshooting.
    try:
        video_dir = artifacts_path / "video"
        trace_dir = artifacts_path / "trace"
        downloads_dir = profile_path / "downloads"
        for d in (video_dir, trace_dir, downloads_dir):
            d.mkdir(parents=True, exist_ok=True)
        profile_path.mkdir(parents=True, exist_ok=True)

        llm = ChatBrowserUse()

        profile = BrowserProfile(
            user_data_dir=str(profile_path),
            traces_dir=str(trace_dir),
            record_video_dir=str(video_dir),
            downloads_path=str(downloads_dir),
            headless=True,
            enable_default_extensions=False,
            chromium_sandbox=False,
            disable_security=False,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )

        task = req.task if req.url is None else f"{req.task}\nStart URL: {req.url}"
        agent = Agent(task=task, llm=llm, browser_profile=profile)
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

        result = await agent.run(on_step_end=on_step_end)
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
    except Exception as exc:  # noqa: BLE001
        fallback = await _playwright_fallback(req, profile_path, artifacts_path)
        return {"error": f"browser_use execution failed: {exc}", "playwright": fallback}, None, {}


async def _playwright_fallback(
    req: RunRequest, profile_path: Path, artifacts_path: Path
) -> Dict[str, Any]:
    """Minimal Playwright warmup to keep profiles/artifacts active even if Browser Use fails."""
    video_dir = artifacts_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    trace_file = artifacts_path / "trace.zip"
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_path),
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
            record_video_dir=str(video_dir),
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()
        if req.record_trace:
            await browser.tracing.start(screenshots=True, snapshots=True, sources=True)
        if req.url:
            await page.goto(req.url)
        if req.record_trace:
            await browser.tracing.stop(path=str(trace_file))
        await browser.close()
    return {
        "opened_url": req.url,
        "trace": str(trace_file) if req.record_trace else None,
        "video_dir": str(video_dir),
    }


@app.post("/run", response_model=RunResponse)
async def run(
    req: RunRequest, _: None = Depends(verify_api_key)
) -> RunResponse:  # pragma: no cover - minimal service surface
    if _lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Runner is busy; try again later.",
        )
    async with _lock:
        run_id = str(uuid.uuid4())
        profile_path = PROFILES_DIR / req.profile_id
        artifacts_path = ARTIFACTS_DIR / run_id
        if _is_chrome_running():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Profile is in use; chrome is running in the container.",
            )
        _clear_profile_locks(profile_path)
        result, history, step_screenshots = await _run_browseruse_task(req, profile_path, artifacts_path)
        steps: Optional[list[StepInfo]] = None
        if history:
            steps_with_paths = _build_steps(history, step_screenshots, include_step_screenshots=True)
            _write_steps_file(artifacts_path, steps_with_paths)
            if req.include_steps:
                include_paths = req.include_step_screenshots == "paths"
                steps = _build_steps(history, step_screenshots, include_step_screenshots=include_paths)
        else:
            _write_steps_file(artifacts_path, [])
            if req.include_steps:
                steps = []
        return RunResponse(
            run_id=run_id,
            artifacts_path=str(artifacts_path),
            profile_path=str(profile_path),
            result=result,
            steps=steps,
        )


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
