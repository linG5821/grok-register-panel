#!/usr/bin/env python3
"""Batch chat quality probe: 降智 / 风控 via real model replies.

SSO grok.com botFlag is no longer a reliable risk signal. This module asks each
CPA (or Grok2API) account to generate a real streamed reply through a configured
residential (家宽) proxy, then classifies:

  - risk     : HTTP 401/403 / permission-denied (account cannot chat)
  - hard     : missing thinking, or Token/s >= hard_tps
  - soft     : Token/s >= soft_tps
  - burst    : short generation window with inflated Token/s
  - healthy  : thinking present and Token/s below soft
  - error    : transport / proxy / parse failure
  - ignored  : reply too short to judge
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from curl_cffi import requests
from secure_files import append_private_text, ensure_private_dir
from sso_to_auth_json import CPA_GROK_BASE_URL, CPA_GROK_HEADERS, CPA_PROBE_MODEL
from webui.security_utils import mask_email, redact_log_line, redact_proxy

DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how TCP slow start works, "
    "at least 12 sentences, plain text only."
)
CHAT_PATH = "/chat/completions"
THINKING_KEYS = (
    "thinking_content",
    "ThinkingContent",
    "thinkingContent",
    "reasoning_content",
    "ReasoningContent",
    "reasoningContent",
    "thinking",
    "Thinking",
)
ACCOUNT_ERROR_MARKERS = (
    "permission-denied",
    "permission denied",
    "forbidden",
    "invalid token",
    "expired",
    "no auth",
    "quota",
    "rate limit",
    "ratelimit",
    "too many requests",
    "unauthorized",
)
SOFT_TPS = 200.0
HARD_TPS = 1000.0
MIN_OUTPUT_TOKENS = 32
MIN_GENERATION_MS = 1000
MAX_OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT = 90
DEFAULT_WORKERS = 2
MAX_CONTENT_CHARS = 400


def classify_failure_kind(status: int, body: str) -> str:
    lower = str(body or "").lower()
    if status in (401, 403, 400, 404, 409, 422, 429):
        return "account_error"
    for marker in ACCOUNT_ERROR_MARKERS:
        if marker in lower:
            return "account_error"
    if status in (407,):
        return "transport_error"
    if status >= 500:
        return "upstream_error"
    return "error"


def classify_sample(
    tps: float,
    output_tokens: int,
    has_thinking: bool,
    gen_ms: int,
    *,
    soft_tps: float = SOFT_TPS,
    hard_tps: float = HARD_TPS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
    min_generation_ms: int = MIN_GENERATION_MS,
    require_thinking: bool = True,
) -> str:
    if output_tokens <= 0 or tps <= 0:
        return "unknown"
    if min_output_tokens > 0 and output_tokens < min_output_tokens:
        return "ignored"
    if require_thinking and not has_thinking:
        return "hard"
    if min_generation_ms > 0 and gen_ms < min_generation_ms and tps >= soft_tps:
        return "burst"
    if tps >= hard_tps:
        return "hard"
    if tps >= soft_tps:
        return "soft"
    return "healthy"


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def map_has_thinking(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in THINKING_KEYS:
        if _nonempty_text(payload.get(key)):
            return True
    return False


def _int_field(payload: dict | None, *keys: str) -> int:
    if not isinstance(payload, dict):
        return 0
    for key in keys:
        raw = payload.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value:
            return value
    return 0


def output_tokens_from_usage(usage: dict | None) -> int:
    if not isinstance(usage, dict):
        return 0
    return max(
        _int_field(usage, "output_tokens", "completion_tokens", "completionTokens"),
        _int_field(usage, "reasoning_tokens", "reasoningTokens"),
    )


def parse_sse_quality(lines) -> dict:
    """Parse an OpenAI-style SSE stream into quality signals."""
    has_thinking = False
    content_chars = 0
    usage_out = 0
    usage_reason = 0
    visible: list[str] = []
    first_token = False
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if not isinstance(chunk, dict):
            continue
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            usage_out = max(usage_out, output_tokens_from_usage(usage))
            usage_reason = max(
                usage_reason,
                _int_field(usage, "reasoning_tokens", "reasoningTokens"),
            )
            if usage_reason > 0:
                has_thinking = True
        choices = chunk.get("choices") or []
        if not isinstance(choices, list):
            choices = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                message = choice.get("message")
                delta = message if isinstance(message, dict) else {}
            if map_has_thinking(delta):
                has_thinking = True
            text = delta.get("content")
            if isinstance(text, str) and text:
                first_token = True
                content_chars += len(text)
                if sum(len(part) for part in visible) < MAX_CONTENT_CHARS:
                    visible.append(text)
            for key in THINKING_KEYS:
                think = delta.get(key)
                if isinstance(think, str) and think:
                    first_token = True
                    content_chars += len(think)
    preview = "".join(visible).replace("\n", " ").strip()[:MAX_CONTENT_CHARS]
    return {
        "has_thinking": has_thinking,
        "content_chars": content_chars,
        "usage_out": usage_out,
        "usage_reason": usage_reason,
        "first_token": first_token,
        "preview": preview,
    }


def _chat_url(record: dict) -> str:
    base = str(record.get("base_url") or CPA_GROK_BASE_URL).strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}{CHAT_PATH}"


def _auth_headers(record: dict) -> dict:
    access = str(record.get("access_token") or record.get("key") or "").strip()
    headers = dict(CPA_GROK_HEADERS)
    extra = record.get("headers")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value:
                headers[str(key)] = str(value)
    headers["Authorization"] = f"Bearer {access}"
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "text/event-stream"
    return headers


def probe_account(
    record: dict,
    proxy: str = "",
    *,
    timeout: int = DEFAULT_TIMEOUT,
    model: str = CPA_PROBE_MODEL,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.7,
    prompt: str = DEFAULT_PROMPT,
    soft_tps: float = SOFT_TPS,
    hard_tps: float = HARD_TPS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
    min_generation_ms: int = MIN_GENERATION_MS,
    require_thinking: bool = True,
    post_fn: Callable | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict:
    """Send one streamed chat completion and classify the account."""
    clock = monotonic or time.monotonic
    email = str(record.get("email") or "").strip()
    access = str(record.get("access_token") or record.get("key") or "").strip()
    result = {
        "email": email,
        "verdict": "error",
        "tps": 0.0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "has_thinking": False,
        "duration_ms": 0,
        "first_token_ms": 0,
        "gen_ms": 0,
        "status_code": 0,
        "error": "",
        "error_kind": "",
        "preview": "",
        "proxy": proxy,
        "model": model,
    }
    if not access:
        result["error"] = "missing access_token"
        result["error_kind"] = "account_error"
        result["verdict"] = "risk"
        return result

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max(8, int(max_tokens or MAX_OUTPUT_TOKENS)),
        "temperature": float(temperature),
    }
    kwargs = {
        "headers": _auth_headers(record),
        "json": payload,
        "impersonate": "chrome",
        "timeout": timeout,
        "stream": True,
    }
    if proxy:
        kwargs["proxy"] = proxy

    start = clock()
    first_token_at = 0.0
    try:
        http_post = post_fn or requests.post
        resp = http_post(_chat_url(record), **kwargs)
    except Exception as exc:
        result["duration_ms"] = int((clock() - start) * 1000)
        result["error"] = redact_log_line(str(exc))[:240]
        result["error_kind"] = "transport_error"
        result["verdict"] = "error"
        return result

    status = int(getattr(resp, "status_code", 0) or 0)
    result["status_code"] = status
    try:
        if status >= 400:
            body = str(getattr(resp, "text", "") or "")[:400]
            kind = classify_failure_kind(status, body)
            result["error_kind"] = kind
            result["error"] = redact_log_line(f"HTTP {status}: {body}")[:240]
            result["duration_ms"] = int((clock() - start) * 1000)
            result["verdict"] = "risk" if kind == "account_error" else "error"
            return result

        parsed_lines = []
        iterator = resp.iter_lines() if hasattr(resp, "iter_lines") else []
        for line in iterator:
            if first_token_at <= 0:
                raw = line.decode("utf-8", "replace") if isinstance(line, (bytes, bytearray)) else str(line or "")
                if '"content"' in raw or any(key in raw for key in THINKING_KEYS):
                    first_token_at = clock()
            parsed_lines.append(line)
        parsed = parse_sse_quality(parsed_lines)
    finally:
        closer = getattr(resp, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    end = clock()
    duration_ms = max(1, int((end - start) * 1000))
    first_token_ms = max(0, int((first_token_at - start) * 1000)) if first_token_at else 0
    gen_ms = max(0, duration_ms - first_token_ms) if first_token_ms else duration_ms
    out_tokens = int(parsed.get("usage_out") or 0)
    reason_tokens = int(parsed.get("usage_reason") or 0)
    if reason_tokens > out_tokens:
        out_tokens = reason_tokens
    if out_tokens <= 0:
        chars = int(parsed.get("content_chars") or 0)
        out_tokens = max(1, chars // 4) if chars else 0
    tps = (out_tokens * 1000.0 / gen_ms) if gen_ms > 0 and out_tokens > 0 else 0.0
    has_thinking = bool(parsed.get("has_thinking")) or reason_tokens > 0
    lower_preview = str(parsed.get("preview") or "").lower()
    if any(marker in lower_preview for marker in ACCOUNT_ERROR_MARKERS) and out_tokens < min_output_tokens:
        result.update(
            {
                "verdict": "risk",
                "error_kind": "account_error",
                "error": "reply looks like a permission/quota denial",
                "duration_ms": duration_ms,
                "preview": parsed.get("preview") or "",
            }
        )
        return result

    verdict = classify_sample(
        tps,
        out_tokens,
        has_thinking,
        gen_ms,
        soft_tps=soft_tps,
        hard_tps=hard_tps,
        min_output_tokens=min_output_tokens,
        min_generation_ms=min_generation_ms,
        require_thinking=require_thinking,
    )
    error = ""
    if verdict == "hard" and require_thinking and not has_thinking:
        error = "响应缺少 thinking_content（降智）"
    elif verdict in {"hard", "soft", "burst"}:
        error = f"Token/s={tps:.1f}（降智）"
    result.update(
        {
            "verdict": verdict,
            "tps": round(tps, 2),
            "output_tokens": out_tokens,
            "reasoning_tokens": reason_tokens,
            "has_thinking": has_thinking,
            "duration_ms": duration_ms,
            "first_token_ms": first_token_ms,
            "gen_ms": gen_ms,
            "error": error,
            "preview": parsed.get("preview") or "",
        }
    )
    return result


def load_auth_records(dirs: list[Path], *, limit: int = 0) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for folder in dirs:
        if not folder or not folder.is_dir():
            continue
        paths = sorted(folder.glob("*.json"))
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            # Grok2API nested issuer::client_id
            if "access_token" not in data and "key" not in data:
                nested = None
                for value in data.values():
                    if isinstance(value, dict) and (
                        value.get("access_token") or value.get("key")
                    ):
                        nested = value
                        break
                if nested is None:
                    continue
                data = dict(nested)
            access = str(data.get("access_token") or data.get("key") or "").strip()
            if not access:
                continue
            email = str(data.get("email") or "").strip()
            ident = email or path.name
            if ident in seen:
                continue
            seen.add(ident)
            item = dict(data)
            item["_path"] = str(path)
            item["_file"] = path.name
            if email and not item.get("email"):
                item["email"] = email
            records.append(item)
            if limit and len(records) >= limit:
                return records
    return records


def public_row(row: dict) -> dict:
    return {
        "index": int(row.get("index") or 0),
        "email": mask_email(str(row.get("email") or "")),
        "file": str(row.get("file") or "")[:180],
        "verdict": str(row.get("verdict") or ""),
        "tps": row.get("tps"),
        "output_tokens": row.get("output_tokens"),
        "reasoning_tokens": row.get("reasoning_tokens"),
        "has_thinking": bool(row.get("has_thinking")),
        "duration_ms": row.get("duration_ms"),
        "first_token_ms": row.get("first_token_ms"),
        "gen_ms": row.get("gen_ms"),
        "status_code": row.get("status_code"),
        "error": redact_log_line(str(row.get("error") or ""))[:240],
        "error_kind": str(row.get("error_kind") or ""),
        "proxy": redact_proxy(str(row.get("proxy") or "")),
        "model": str(row.get("model") or ""),
    }


def run_quality_scan(
    records: list[dict],
    proxies: list[str],
    *,
    workers: int = DEFAULT_WORKERS,
    delay: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
    model: str = CPA_PROBE_MODEL,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.7,
    prompt: str = DEFAULT_PROMPT,
    soft_tps: float = SOFT_TPS,
    hard_tps: float = HARD_TPS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
    min_generation_ms: int = MIN_GENERATION_MS,
    require_thinking: bool = True,
    export: str | Path | None = None,
    risk_export: str | Path | None = None,
    log=print,
    on_item=None,
    cancel_callback=None,
    post_fn: Callable | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict:
    """Probe many accounts. Export files are redacted jsonl (no tokens)."""
    pool = [str(item or "").strip() for item in (proxies or [])]
    if not pool:
        pool = [""]
    summary: dict = {
        "ok": True,
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": 0,
        "healthy_count": 0,
        "soft_count": 0,
        "hard_count": 0,
        "burst_count": 0,
        "risk_count": 0,
        "error_count": 0,
        "ignored_count": 0,
        "unknown_count": 0,
        "degraded_count": 0,
        "items": [],
        "export_path": "",
        "export_count": 0,
        "risk_export_path": "",
        "risk_export_count": 0,
        "cancelled": False,
        "workers": max(1, int(workers or 1)),
        "proxy_count": len([item for item in pool if item]),
        "model": model,
    }
    export_path = Path(export) if export else None
    risk_path = Path(risk_export) if risk_export else None
    if export_path:
        export_path.unlink(missing_ok=True)
        ensure_private_dir(export_path.parent)
    if risk_path:
        risk_path.unlink(missing_ok=True)
        ensure_private_dir(risk_path.parent)

    total = len(records)
    worker_n = max(1, min(int(workers or 1), total or 1, 8))

    def _probe_one(index: int, record: dict) -> dict:
        if delay and index > 1:
            time.sleep(float(delay))
        proxy = pool[(index - 1) % len(pool)]
        probed = probe_account(
            record,
            proxy=proxy,
            timeout=timeout,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt=prompt,
            soft_tps=soft_tps,
            hard_tps=hard_tps,
            min_output_tokens=min_output_tokens,
            min_generation_ms=min_generation_ms,
            require_thinking=require_thinking,
            post_fn=post_fn,
            monotonic=monotonic,
        )
        row = {
            "index": index,
            "email": str(record.get("email") or ""),
            "file": str(record.get("_file") or ""),
            **probed,
        }
        return row

    pending = list(enumerate(records, 1))
    completed: list[dict] = []
    if worker_n == 1:
        for index, record in pending:
            if cancel_callback and cancel_callback():
                summary["cancelled"] = True
                break
            completed.append(_probe_one(index, record))
    else:
        with ThreadPoolExecutor(max_workers=worker_n) as pool_ex:
            futures = {
                pool_ex.submit(_probe_one, index, record): index
                for index, record in pending
            }
            for future in as_completed(futures):
                if cancel_callback and cancel_callback():
                    summary["cancelled"] = True
                    for leftover in futures:
                        leftover.cancel()
                    break
                try:
                    completed.append(future.result())
                except Exception as exc:
                    index = futures[future]
                    completed.append(
                        {
                            "index": index,
                            "email": str(records[index - 1].get("email") or ""),
                            "file": str(records[index - 1].get("_file") or ""),
                            "verdict": "error",
                            "error": redact_log_line(str(exc))[:240],
                            "error_kind": "error",
                            "tps": 0,
                            "has_thinking": False,
                            "status_code": 0,
                            "proxy": pool[(index - 1) % len(pool)],
                            "model": model,
                        }
                    )

    completed.sort(key=lambda item: int(item.get("index") or 0))
    for row in completed:
        summary["total"] += 1
        verdict = str(row.get("verdict") or "error")
        key = f"{verdict}_count"
        if key in summary:
            summary[key] += 1
        else:
            summary["unknown_count"] += 1
        if verdict in {"hard", "soft", "burst"}:
            summary["degraded_count"] += 1
            if export_path:
                append_private_text(
                    export_path, json.dumps(public_row(row), ensure_ascii=False) + "\n"
                )
                summary["export_count"] += 1
        if verdict == "risk" and risk_path:
            append_private_text(
                risk_path, json.dumps(public_row(row), ensure_ascii=False) + "\n"
            )
            summary["risk_export_count"] += 1
        summary["items"].append(row)
        tag = {
            "healthy": "✅",
            "soft": "⚠️",
            "hard": "❌",
            "burst": "❌",
            "risk": "⛔",
        }.get(verdict, "⚠️")
        log(
            f"{tag} [{row.get('index')}/{total}] {row.get('email') or '(no email)'} "
            f"tps={row.get('tps')} think={int(bool(row.get('has_thinking')))} "
            f"status={row.get('status_code')} -> {verdict}"
        )
        if on_item:
            on_item(row, summary)

    summary["export_path"] = str(export_path) if export_path else ""
    summary["risk_export_path"] = str(risk_path) if risk_path else ""
    if summary["total"] == 0:
        summary["ok"] = False
    return summary
