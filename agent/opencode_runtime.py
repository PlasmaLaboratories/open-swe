"""OpenCode runtime adapter for Open SWE.

This keeps the existing Open SWE webhook/thread/sandbox plumbing but replaces
the inner agent loop with an OpenCode session running inside the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import textwrap
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_config, get_store
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel import Pregel
from langgraph_sdk import get_client

from .tools.commit_and_open_pr import commit_and_open_pr
from .utils.github_app import get_github_app_installation_token
from .utils.github_comments import post_github_comment
from .utils.linear import comment_on_linear_issue
from .utils.messages import extract_text_content
from .utils.multimodal import fetch_image_block
from .utils.slack import post_slack_thread_reply

logger = logging.getLogger(__name__)

DEFAULT_OPENCODE_MODEL = "anthropic/claude-opus-4-1-20250805"
OPENCODE_SERVER_PORT = 4096
OPENCODE_RUNTIME_DIR = "/tmp/open-swe-opencode"
OPENCODE_REQUEST_PATH = f"{OPENCODE_RUNTIME_DIR}/request.json"
OPENCODE_SERVER_LOG_PATH = f"{OPENCODE_RUNTIME_DIR}/server.log"
OPENCODE_SERVER_PID_PATH = f"{OPENCODE_RUNTIME_DIR}/server.pid"

_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "opencode": "OPENCODE_API_KEY",
}


def parse_opencode_model(model: str | None = None) -> tuple[str, str]:
    """Return `(provider_id, model_id)` for OpenCode."""
    model_value = (model or os.environ.get("OPENCODE_MODEL") or DEFAULT_OPENCODE_MODEL).strip()

    if "/" in model_value:
        provider_id, model_id = model_value.split("/", 1)
    elif ":" in model_value:
        provider_id, model_id = model_value.split(":", 1)
    else:
        msg = (
            "Invalid OpenCode model. Set OPENCODE_MODEL as 'provider/model', "
            f"got: {model_value!r}"
        )
        raise ValueError(msg)

    provider_id = provider_id.strip()
    model_id = model_id.strip()
    if not provider_id or not model_id:
        raise ValueError(f"Invalid OpenCode model: {model_value!r}")

    return provider_id, model_id


def extract_pending_user_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return user messages added since the last assistant reply."""
    pending: list[BaseMessage] = []
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        if message_type in {"ai", "assistant"}:
            break
        if message_type in {"human", "user"}:
            pending.append(message)
    pending.reverse()
    return pending


def content_to_opencode_parts(content: Any) -> list[dict[str, Any]]:
    """Convert LangChain message content into OpenCode `parts`."""
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "text", "text": text}] if text else []

    if not isinstance(content, list):
        text = extract_text_content(content)
        return [{"type": "text", "text": text}] if text else []

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue

        part_type = item.get("type")
        if part_type == "text":
            text = item.get("text", "").strip()
            if text:
                parts.append({"type": "text", "text": text})
            continue

        if part_type != "image":
            continue

        source = item.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            data = source.get("data")
            media_type = source.get("media_type")
            if isinstance(data, str) and isinstance(media_type, str):
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            continue

        data = item.get("base64")
        media_type = item.get("mime_type")
        if isinstance(data, str) and isinstance(media_type, str):
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )

    return parts


def build_default_pr_payload(summary: str) -> tuple[str, str]:
    """Build a deterministic PR title/body for the OpenCode path."""
    config = get_config()
    configurable = config.get("configurable", {})

    linear_issue = configurable.get("linear_issue", {}) or {}
    github_issue = configurable.get("github_issue", {}) or {}

    source_title = linear_issue.get("title") or github_issue.get("title") or "requested changes"
    normalized_title = " ".join(str(source_title).lower().split()) or "requested changes"
    normalized_title = normalized_title[:60].rstrip(" -_:,.;")
    pr_title = f"chore: {normalized_title}"

    description = (summary or "Implements the requested changes using the OpenCode runtime.").strip()
    description = " ".join(description.split())

    lines: list[str] = []
    identifier = linear_issue.get("identifier")
    linear_url = linear_issue.get("url")
    if identifier and linear_url:
        lines.append(f"Closes [{identifier}]({linear_url})")
        lines.append("")

    lines.extend(
        [
            "## Description",
            description,
            "",
            "## Test Plan",
            "- [ ] Verify the requested workflow behaves as expected",
        ]
    )

    return pr_title, "\n".join(lines)


async def _queued_payload_to_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    text = payload.get("text", "")
    image_urls = payload.get("image_urls", []) or []

    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})

    if not image_urls:
        return parts

    async with httpx.AsyncClient() as http_client:
        for image_url in image_urls:
            image_block = await fetch_image_block(image_url, http_client)
            if not image_block:
                continue
            parts.extend(content_to_opencode_parts([image_block]))
    return parts


async def drain_queued_opencode_parts() -> list[list[dict[str, Any]]]:
    """Drain queued follow-up messages for the current thread."""
    config = get_config()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return []

    try:
        store = get_store()
    except Exception:
        logger.exception("Could not get store from context while draining queue")
        return []

    if store is None:
        return []

    namespace = ("queue", thread_id)
    try:
        queued_item = await store.aget(namespace, "pending_messages")
    except Exception:
        logger.exception("Failed to fetch queued messages for thread %s", thread_id)
        return []

    if queued_item is None:
        return []

    await store.adelete(namespace, "pending_messages")
    queued_messages = queued_item.value.get("messages", [])
    if not queued_messages:
        return []

    drained_parts: list[list[dict[str, Any]]] = []
    for message in queued_messages:
        content = message.get("content")
        if isinstance(content, dict) and ("text" in content or "image_urls" in content):
            parts = await _queued_payload_to_parts(content)
        else:
            parts = content_to_opencode_parts(content)

        if parts:
            drained_parts.append(parts)

    return drained_parts


async def _run_in_sandbox(
    sandbox_backend: Any,
    command: str,
    *,
    timeout: int = 300,
) -> Any:
    return await asyncio.to_thread(sandbox_backend.execute, command, timeout=timeout)


async def _write_sandbox_file(sandbox_backend: Any, path: str, content: str) -> None:
    result = await asyncio.to_thread(sandbox_backend.write, path, content)
    error = getattr(result, "error", None)
    if error:
        raise RuntimeError(str(error))


async def _ensure_runtime_dir(sandbox_backend: Any) -> None:
    result = await _run_in_sandbox(
        sandbox_backend,
        f"mkdir -p {shlex.quote(OPENCODE_RUNTIME_DIR)}",
    )
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to create OpenCode runtime dir: {result.output}")


async def _ensure_opencode_installed(sandbox_backend: Any) -> None:
    install_cmd = (
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "(command -v opencode >/dev/null 2>&1 || curl -fsSL https://opencode.ai/install | bash) && "
        "command -v opencode >/dev/null 2>&1"
    )
    result = await _run_in_sandbox(sandbox_backend, install_cmd, timeout=900)
    if result.exit_code != 0:
        raise RuntimeError(
            "Failed to install or locate `opencode` in the sandbox. "
            f"Output: {result.output.strip()}"
        )


async def _sandbox_http_request(
    sandbox_backend: Any,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: int = 300,
) -> tuple[int, Any]:
    await _ensure_runtime_dir(sandbox_backend)
    payload = {
        "method": method,
        "url": f"http://127.0.0.1:{OPENCODE_SERVER_PORT}{path}",
        "body": body,
        "headers": {"Content-Type": "application/json"},
    }
    await _write_sandbox_file(sandbox_backend, OPENCODE_REQUEST_PATH, json.dumps(payload))

    script = textwrap.dedent(
        f"""
        import json
        import sys
        import urllib.error
        import urllib.request

        with open({OPENCODE_REQUEST_PATH!r}, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        data = None
        if payload.get("body") is not None:
            data = json.dumps(payload["body"]).encode("utf-8")

        request = urllib.request.Request(
            payload["url"],
            data=data,
            headers=payload.get("headers") or {{}},
            method=payload["method"],
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
                print(json.dumps({{"status": response.getcode(), "body": body}}))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            print(json.dumps({{"status": exc.code, "body": body}}))
        except Exception as exc:
            print(json.dumps({{"status": 0, "error": str(exc)}}))
            sys.exit(1)
        """
    )
    command = f"python3 - <<'PY'\n{script}\nPY"
    result = await _run_in_sandbox(sandbox_backend, command, timeout=timeout)
    if result.exit_code != 0:
        raise RuntimeError(f"Sandbox HTTP request failed: {result.output.strip()}")

    response = json.loads(result.output.strip() or "{}")
    status = int(response.get("status", 0))
    body_text = response.get("body", "")
    if not status:
        raise RuntimeError(response.get("error", "Unknown sandbox HTTP error"))

    try:
        parsed_body = json.loads(body_text) if body_text else None
    except json.JSONDecodeError:
        parsed_body = body_text
    return status, parsed_body


async def _server_healthy(sandbox_backend: Any) -> bool:
    try:
        status, body = await _sandbox_http_request(
            sandbox_backend,
            "GET",
            "/global/health",
            timeout=30,
        )
    except Exception:
        return False
    return status == 200 and isinstance(body, dict) and body.get("healthy") is True


async def _ensure_opencode_server(sandbox_backend: Any, repo_dir: str) -> None:
    if await _server_healthy(sandbox_backend):
        return

    await _ensure_runtime_dir(sandbox_backend)

    start_cmd = (
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        f"cd {shlex.quote(repo_dir)} && "
        f"nohup opencode serve --hostname 127.0.0.1 --port {OPENCODE_SERVER_PORT} "
        f">{shlex.quote(OPENCODE_SERVER_LOG_PATH)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(OPENCODE_SERVER_PID_PATH)}"
    )
    result = await _run_in_sandbox(sandbox_backend, start_cmd, timeout=60)
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to start OpenCode server: {result.output.strip()}")

    for _ in range(20):
        if await _server_healthy(sandbox_backend):
            break
        await asyncio.sleep(1)
    else:
        raise RuntimeError("OpenCode server did not become healthy in time")

    status, _body = await _sandbox_http_request(
        sandbox_backend,
        "PATCH",
        "/config",
        body={"permission": "allow"},
    )
    if status not in {200, 204}:
        raise RuntimeError("Failed to configure OpenCode permissions")


async def _ensure_provider_auth(sandbox_backend: Any, provider_id: str) -> None:
    key_env = _PROVIDER_API_KEY_ENV.get(provider_id)
    if not key_env:
        raise ValueError(
            "OpenCode runtime does not know which API key env var to use for provider "
            f"{provider_id!r}. Set OPENCODE_MODEL to a supported provider."
        )

    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing {key_env} for OpenCode runtime. "
            f"Set {key_env} in the Open SWE environment."
        )

    status, body = await _sandbox_http_request(
        sandbox_backend,
        "PUT",
        f"/auth/{provider_id}",
        body={"type": "api", "key": api_key},
    )
    if status not in {200, 204} or body is False:
        raise RuntimeError(f"Failed to configure OpenCode auth for provider {provider_id}")


async def ensure_opencode_ready(sandbox_backend: Any, repo_dir: str) -> tuple[str, str]:
    provider_id, model_id = parse_opencode_model()
    await _ensure_opencode_installed(sandbox_backend)
    await _ensure_opencode_server(sandbox_backend, repo_dir)
    await _ensure_provider_auth(sandbox_backend, provider_id)
    return provider_id, model_id


async def _get_or_create_session_id(sandbox_backend: Any) -> tuple[str, bool]:
    config = get_config()
    thread_id = config.get("configurable", {}).get("thread_id")
    metadata = config.get("metadata", {})
    existing_session_id = metadata.get("opencode_session_id")

    if existing_session_id:
        status, _body = await _sandbox_http_request(
            sandbox_backend,
            "GET",
            f"/session/{existing_session_id}",
        )
        if status == 200:
            return str(existing_session_id), False

    title = f"Open SWE {thread_id}" if thread_id else "Open SWE"
    status, body = await _sandbox_http_request(
        sandbox_backend,
        "POST",
        "/session",
        body={"title": title},
    )
    if status != 200 or not isinstance(body, dict) or not body.get("id"):
        raise RuntimeError(f"Failed to create OpenCode session: {body!r}")

    session_id = str(body["id"])
    if thread_id:
        client = get_client()
        await client.threads.update(thread_id=thread_id, metadata={"opencode_session_id": session_id})
    return session_id, True


def _extract_response_text(response: dict[str, Any]) -> str:
    parts = response.get("parts", [])
    texts: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n\n".join(texts).strip()


async def _prompt_session(
    sandbox_backend: Any,
    session_id: str,
    provider_id: str,
    model_id: str,
    parts: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": {"providerID": provider_id, "modelID": model_id},
        "agent": "build",
        "parts": parts,
    }
    if system_prompt:
        body["system"] = system_prompt

    status, response = await _sandbox_http_request(
        sandbox_backend,
        "POST",
        f"/session/{session_id}/message",
        body=body,
        timeout=1800,
    )
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"OpenCode prompt failed: {response!r}")
    return response


def _build_external_reply(text: str, *, pr_url: str | None, source: str) -> str:
    normalized = text.strip() or "Completed."
    if source == "slack":
        if pr_url:
            return f"*Pull Request Created*\n\n{normalized}\n\n{pr_url}"
        return f"*Agent Response*\n\n{normalized}"

    if pr_url:
        return f"✅ **Pull Request Created**\n\n{normalized}\n\nPR: {pr_url}"
    return f"🤖 **Agent Response**\n\n{normalized}"


async def _post_external_reply(text: str, *, pr_url: str | None = None) -> None:
    config = get_config()
    configurable = config.get("configurable", {})
    source = configurable.get("source")
    if not source:
        return

    message = _build_external_reply(text, pr_url=pr_url, source=source)

    if source == "slack":
        slack_thread = configurable.get("slack_thread", {}) or {}
        channel_id = slack_thread.get("channel_id")
        thread_ts = slack_thread.get("thread_ts")
        if channel_id and thread_ts:
            await post_slack_thread_reply(channel_id, thread_ts, message)
        return

    if source == "linear":
        linear_issue = configurable.get("linear_issue", {}) or {}
        issue_id = linear_issue.get("id")
        if issue_id:
            await comment_on_linear_issue(issue_id, message)
        return

    if source == "github":
        repo_config = configurable.get("repo", {}) or {}
        issue_number = configurable.get("pr_number") or configurable.get("github_issue", {}).get("number")
        if repo_config and issue_number:
            token = await get_github_app_installation_token()
            if token:
                await post_github_comment(repo_config, int(issue_number), message, token=token)


async def _post_external_error(error_text: str) -> None:
    config = get_config()
    source = config.get("configurable", {}).get("source")
    if not source:
        return

    if source == "slack":
        message = f"*Agent Error*\n\n{error_text}"
        configurable = config.get("configurable", {})
        slack_thread = configurable.get("slack_thread", {}) or {}
        channel_id = slack_thread.get("channel_id")
        thread_ts = slack_thread.get("thread_ts")
        if channel_id and thread_ts:
            await post_slack_thread_reply(channel_id, thread_ts, message)
    else:
        message = f"❌ **Agent Error**\n\n{error_text}"
        configurable = config.get("configurable", {})
        if source == "linear":
            linear_issue = configurable.get("linear_issue", {}) or {}
            issue_id = linear_issue.get("id")
            if issue_id:
                await comment_on_linear_issue(issue_id, message)
            return

        if source == "github":
            repo_config = configurable.get("repo", {}) or {}
            issue_number = configurable.get("pr_number") or configurable.get("github_issue", {}).get("number")
            if repo_config and issue_number:
                token = await get_github_app_installation_token()
                if token:
                    await post_github_comment(repo_config, int(issue_number), message, token=token)


async def maybe_commit_and_open_pr(summary: str) -> str | None:
    """Commit changes and open a PR when changes exist."""
    title, body = build_default_pr_payload(summary)
    result = await asyncio.to_thread(commit_and_open_pr, title, body, title)
    if not isinstance(result, dict):
        return None

    if result.get("success") and result.get("pr_url"):
        return str(result["pr_url"])

    error = (result.get("error") or "").lower()
    if "no changes detected" in error:
        return None

    if error:
        logger.warning("OpenCode PR automation did not complete: %s", result.get("error"))
    return None


async def run_opencode_once(
    sandbox_backend: Any,
    repo_dir: str,
    system_prompt: str,
    messages: list[BaseMessage],
) -> list[AIMessage]:
    """Run OpenCode for the pending user messages plus queued follow-ups."""
    pending_messages = extract_pending_user_messages(messages)
    pending_parts_batches = [
        parts
        for message in pending_messages
        for parts in [content_to_opencode_parts(message.content)]
        if parts
    ]

    if not pending_parts_batches:
        return []

    provider_id, model_id = await ensure_opencode_ready(sandbox_backend, repo_dir)
    session_id, is_new_session = await _get_or_create_session_id(sandbox_backend)

    assistant_messages: list[AIMessage] = []
    prompt_queue = list(pending_parts_batches)
    include_system_prompt = is_new_session

    while prompt_queue:
        parts = prompt_queue.pop(0)
        response = await _prompt_session(
            sandbox_backend,
            session_id,
            provider_id,
            model_id,
            parts,
            system_prompt=system_prompt if include_system_prompt else None,
        )
        include_system_prompt = False

        response_text = _extract_response_text(response) or "Completed."
        assistant_messages.append(AIMessage(content=response_text))

        queued_parts = await drain_queued_opencode_parts()
        prompt_queue.extend(queued_parts)

    final_summary = extract_text_content(assistant_messages[-1].content) if assistant_messages else ""
    pr_url = await maybe_commit_and_open_pr(final_summary)
    await _post_external_reply(final_summary, pr_url=pr_url)
    return assistant_messages


def create_opencode_agent(
    *,
    system_prompt: str,
    sandbox_backend: Any | None,
    repo_dir: str | None,
) -> Pregel:
    """Create a LangGraph wrapper that delegates execution to OpenCode."""

    async def run_opencode(state: MessagesState) -> dict[str, Any]:
        if not sandbox_backend or not repo_dir:
            return {}

        try:
            messages = state.get("messages", [])
            assistant_messages = await run_opencode_once(
                sandbox_backend,
                repo_dir,
                system_prompt,
                messages,
            )
            return {"messages": assistant_messages} if assistant_messages else {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenCode runtime failed")
            error_text = f"OpenCode runtime error: {type(exc).__name__}: {exc}"
            await _post_external_error(error_text)
            return {"messages": [AIMessage(content=error_text)]}

    graph = StateGraph(MessagesState)
    graph.add_node("run_opencode", run_opencode)
    graph.add_edge(START, "run_opencode")
    graph.add_edge("run_opencode", END)
    return graph.compile(name="opencode_agent")
