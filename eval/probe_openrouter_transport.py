"""Cheap ADK/LiteLLM vision and tool-call transport probes.

For OpenRouter proxy models, run after starting the local LiteLLM proxy:

    litellm --config configs/litellm/openrouter-qwen36.yaml --port 4000
    python eval/probe_openrouter_transport.py

Local Ollama models can be probed directly:

    python eval/probe_openrouter_transport.py --model ollama:gemma4:26b
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps.app import App  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from langslice.harness.estimation.adk_plugins import RequestCapturePlugin  # noqa: E402
from langslice.harness.estimation.model_resolver import resolve_adk_model  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

APP_NAME = "langslice_transport_probe"
USER_ID = "probe-user"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="litellm-proxy:langslice-qwen36-plus",
        help="LangSlice ADK model string to probe",
    )
    parser.add_argument(
        "--proxy-base",
        default=os.getenv("LANGSLICE_LITELLM_PROXY_BASE", "http://127.0.0.1:4000/v1"),
        help="LiteLLM proxy OpenAI-compatible /v1 base URL",
    )
    parser.add_argument(
        "--proxy-key",
        default=os.getenv("LANGSLICE_LITELLM_PROXY_KEY", "sk-langslice-local"),
        help="LiteLLM proxy bearer key",
    )
    parser.add_argument(
        "--capture-dir",
        default=str(REPO_ROOT / "eval_outputs" / "openrouter_transport_probe"),
        help="Directory for redacted ADK request captures",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    return parser.parse_args()


def _proxy_model_info(proxy_base: str, proxy_key: str) -> dict[str, Any]:
    url = proxy_base.rstrip("/") + "/model/info"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {proxy_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {
                "ok": True,
                "status": response.status,
                "body": json.loads(response.read().decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _needs_proxy_info(model: str) -> bool:
    return model.strip().lower().startswith("litellm-proxy:")


def _synthetic_probe_image_part() -> types.Part:
    img = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 45, 260, 175), fill=(220, 20, 30))
    draw.text((92, 92), "RED", fill="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(mime_type="image/jpeg", data=buf.getvalue())


async def _run_agent_once(
    *,
    model: str,
    instruction: str,
    message: types.Content,
    capture_dir: Path,
    run_label: str,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    agent = LlmAgent(
        model=resolve_adk_model(model),  # type: ignore[arg-type]
        name=f"{run_label}_agent",
        instruction=instruction,
        tools=tools or [],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
        ),
    )
    app = App(
        name=APP_NAME,
        root_agent=agent,
        plugins=[RequestCapturePlugin(capture_dir, run_label=run_label)],
    )
    runner = InMemoryRunner(app=app)
    assert runner.session_service is not None
    session_id = f"{run_label}_session"
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={},
    )

    final_text: list[str] = []
    function_calls: list[str] = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
    ):
        for call in event.get_function_calls() or []:
            function_calls.append(str(getattr(call, "name", "")))
        content = getattr(event, "content", None)
        if content is not None:
            for part in content.parts or []:
                if getattr(part, "text", None):
                    final_text.append(part.text or "")

    return {
        "text": "\n".join(final_text).strip(),
        "function_calls": function_calls,
    }


def choose_number(value: float) -> dict[str, float]:
    """Record a chosen numeric value for the transport probe."""
    return {"value": value}


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["LANGSLICE_LITELLM_PROXY_BASE"] = args.proxy_base
    os.environ["LANGSLICE_LITELLM_PROXY_KEY"] = args.proxy_key
    capture_dir = Path(args.capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)

    model_info = (
        _proxy_model_info(args.proxy_base, args.proxy_key)
        if _needs_proxy_info(args.model)
        else {"ok": True, "skipped": "model does not use the LiteLLM proxy"}
    )

    vision = await _run_agent_once(
        model=args.model,
        instruction=(
            "You are a vision transport probe. Answer in one short sentence. "
            "Mention the main color you see."
        ),
        message=types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="What color is the large rectangle?"),
                _synthetic_probe_image_part(),
            ],
        ),
        capture_dir=capture_dir,
        run_label="vision",
    )

    tool = await _run_agent_once(
        model=args.model,
        instruction=(
            "You are a tool transport probe. You must call choose_number once "
            "with value=4.2, then stop."
        ),
        message=types.Content(
            role="user",
            parts=[types.Part.from_text(text="Call choose_number with value 4.2.")],
        ),
        tools=[choose_number],
        capture_dir=capture_dir,
        run_label="tool",
    )

    return {
        "model": args.model,
        "proxy_base": args.proxy_base,
        "model_info": model_info,
        "vision_probe": vision,
        "tool_probe": tool,
        "capture_dir": str(capture_dir),
        "passed": bool(vision["text"]) and "choose_number" in tool["function_calls"],
    }


def main() -> int:
    args = _parse_args()
    result = asyncio.run(_main_async(args))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
