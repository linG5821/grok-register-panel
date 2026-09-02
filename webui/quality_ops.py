"""Batch 降智 / chat-quality scan for the live panel."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_probe import (
    DEFAULT_WORKERS,
    HARD_TPS,
    MIN_GENERATION_MS,
    MIN_OUTPUT_TOKENS,
    SOFT_TPS,
    load_auth_records,
    public_row,
    run_quality_scan,
)
from secure_files import (
    atomic_write_json,
    create_private_text,
    ensure_private_dir,
    exclusive_file_lock,
)

try:
    from webui.proxy_store import worker_proxy_details
    from webui.security_utils import redact_log_line, redact_proxy
except ImportError:
    from proxy_store import worker_proxy_details  # type: ignore
    from security_utils import redact_log_line, redact_proxy  # type: ignore

CPA_DIR = Path(os.environ.get("CPA_AUTH_DIR", str(ROOT / "cpa_auth")))
G2A_DIR = Path(os.environ.get("GROK2API_AUTH_DIR", str(ROOT / "grok2api_auth")))
CONFIG_FILE = ROOT / "config.json"
LOG_DIR = ROOT / "log"
REPORT_FILE = LOG_DIR / "quality_scan_report.json"
DEGRADED_EXPORT = LOG_DIR / "quality_degraded.jsonl"
RISK_EXPORT = LOG_DIR / "quality_risk.jsonl"

MAX_RECORDS = 2000
VALID_SOURCES = ("cpa", "g2a", "all")

_lock = threading.Lock()
_cancel = threading.Event()
_thread: threading.Thread | None = None
_state: dict = {
    "running": False,
    "started_at": "",
    "finished_at": "",
    "error": "",
    "cancelled": False,
    "source": "",
    "proxy_mode": "",
    "progress": 0,
    "total": 0,
    "summary": {},
    "items": [],
    "run_id": "",
}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _run_claim_file() -> Path:
    return LOG_DIR / "quality_scan.claim"


def _run_claim_lock() -> Path:
    return LOG_DIR / "quality_scan.claim.lock"


def _claim_run(run_id: str) -> bool:
    ensure_private_dir(LOG_DIR)
    claim_file = _run_claim_file()
    with exclusive_file_lock(_run_claim_lock()):
        if claim_file.is_file():
            try:
                current = json.loads(claim_file.read_text(encoding="utf-8"))
                pid = int(current.get("pid") or 0)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pid = 0
            if _pid_alive(pid):
                return False
            claim_file.unlink(missing_ok=True)
        try:
            create_private_text(
                claim_file,
                json.dumps({"run_id": run_id, "pid": os.getpid()}) + "\n",
            )
        except FileExistsError:
            return False
    return True


def _release_run(run_id: str) -> None:
    claim_file = _run_claim_file()
    with exclusive_file_lock(_run_claim_lock()):
        if not claim_file.is_file():
            return
        try:
            current = json.loads(claim_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        if str(current.get("run_id") or "") == run_id:
            claim_file.unlink(missing_ok=True)


def _config() -> dict:
    if not CONFIG_FILE.is_file():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _config_proxy() -> str:
    return str(_config().get("proxy") or "").strip()


def _resolve_auth_dirs(source: str) -> list[Path]:
    cfg = _config()
    base = CONFIG_FILE.parent

    def _add(raw: object, fallback: Path, bucket: list[Path], seen: set[str]) -> None:
        text = str(raw or "").strip()
        path = Path(text) if text else fallback
        if not path.is_absolute():
            path = base / path
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        bucket.append(path)

    dirs: list[Path] = []
    seen: set[str] = set()
    if source in {"cpa", "all"}:
        _add(os.environ.get("CPA_AUTH_DIR") or cfg.get("cpa_auth_dir"), CPA_DIR, dirs, seen)
    if source in {"g2a", "all"}:
        _add(
            os.environ.get("GROK2API_AUTH_DIR") or cfg.get("grok2api_auth_dir"),
            G2A_DIR,
            dirs,
            seen,
        )
    return dirs


def resolve_probe_proxies(explicit: str = "", *, prefer_home: bool = True) -> list[str]:
    text = str(explicit or "").strip()
    if text:
        return [text]
    homes: list[str] = []
    others: list[str] = []
    try:
        for row in worker_proxy_details() or []:
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            if row.get("home"):
                homes.append(url)
            else:
                others.append(url)
    except Exception:
        pass
    if prefer_home and homes:
        return homes
    if homes:
        return homes
    if others:
        return others
    fallback = _config_proxy()
    return [fallback] if fallback else [""]


def source_counts() -> dict:
    cpa = load_auth_records(_resolve_auth_dirs("cpa"))
    g2a = load_auth_records(_resolve_auth_dirs("g2a"))
    return {"cpa": len(cpa), "g2a": len(g2a), "all": len(cpa) + len(g2a)}


def _public_summary(summary: dict) -> dict:
    if not isinstance(summary, dict):
        return {}
    return {
        "ok": bool(summary.get("ok", True)),
        "scanned_at": summary.get("scanned_at") or "",
        "total": int(summary.get("total") or 0),
        "healthy_count": int(summary.get("healthy_count") or 0),
        "soft_count": int(summary.get("soft_count") or 0),
        "hard_count": int(summary.get("hard_count") or 0),
        "burst_count": int(summary.get("burst_count") or 0),
        "risk_count": int(summary.get("risk_count") or 0),
        "error_count": int(summary.get("error_count") or 0),
        "ignored_count": int(summary.get("ignored_count") or 0),
        "unknown_count": int(summary.get("unknown_count") or 0),
        "degraded_count": int(summary.get("degraded_count") or 0),
        "export_path": str(summary.get("export_path") or ""),
        "export_count": int(summary.get("export_count") or 0),
        "risk_export_path": str(summary.get("risk_export_path") or ""),
        "risk_export_count": int(summary.get("risk_export_count") or 0),
        "cancelled": bool(summary.get("cancelled")),
        "workers": int(summary.get("workers") or 0),
        "proxy_count": int(summary.get("proxy_count") or 0),
        "model": str(summary.get("model") or ""),
    }


def _load_saved_report() -> dict:
    if not REPORT_FILE.is_file():
        return {}
    try:
        data = json.loads(REPORT_FILE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def quality_status() -> dict:
    saved = _load_saved_report()
    with _lock:
        running = bool(_state["running"])
        live_items = [public_row(it) for it in _state.get("items") or []]
        live_summary = _public_summary(_state.get("summary") or {})
        live_run_id = str(_state.get("run_id") or "")
        use_live = running or bool(live_run_id)
        snapshot = {
            "ok": True,
            "running": running,
            "started_at": _state.get("started_at") or "",
            "finished_at": _state.get("finished_at") or "",
            "error": _state.get("error") or "",
            "cancelled": bool(_state.get("cancelled")),
            "source": (_state.get("source") if use_live else saved.get("source")) or "",
            "proxy_mode": (
                _state.get("proxy_mode") if use_live else saved.get("proxy_mode")
            )
            or "",
            "progress": int(_state.get("progress") or 0),
            "total": int(_state.get("total") or 0),
            "run_id": live_run_id or str(saved.get("run_id") or ""),
            "historical": not use_live and bool(saved),
            "summary": live_summary
            if use_live
            else {
                key: saved.get(key)
                for key in (
                    "ok",
                    "scanned_at",
                    "total",
                    "healthy_count",
                    "soft_count",
                    "hard_count",
                    "burst_count",
                    "risk_count",
                    "error_count",
                    "ignored_count",
                    "unknown_count",
                    "degraded_count",
                    "export_path",
                    "export_count",
                    "risk_export_path",
                    "risk_export_count",
                    "cancelled",
                    "workers",
                    "proxy_count",
                    "model",
                )
                if key in saved
            },
            "items": live_items if use_live else saved.get("items") or [],
        }
    snapshot["sources"] = source_counts()
    snapshot["default_proxy"] = redact_proxy(_config_proxy())
    snapshot["home_proxy_count"] = len(
        [item for item in resolve_probe_proxies(prefer_home=True) if item]
    )
    snapshot["report_path"] = str(REPORT_FILE) if REPORT_FILE.exists() else ""
    snapshot["degraded_export"] = str(DEGRADED_EXPORT) if DEGRADED_EXPORT.exists() else ""
    snapshot["risk_export"] = str(RISK_EXPORT) if RISK_EXPORT.exists() else ""
    snapshot["thresholds"] = {
        "soft_tps": SOFT_TPS,
        "hard_tps": HARD_TPS,
        "min_output_tokens": MIN_OUTPUT_TOKENS,
        "min_generation_ms": MIN_GENERATION_MS,
        "require_thinking": True,
    }
    return snapshot


def _persist_report(summary: dict, *, source: str, proxy_mode: str, run_id: str) -> None:
    ensure_private_dir(LOG_DIR)
    payload = dict(summary)
    payload["source"] = source
    payload["proxy_mode"] = proxy_mode
    payload["run_id"] = run_id
    payload["items"] = [public_row(it) for it in (summary.get("items") or [])]
    atomic_write_json(REPORT_FILE, payload)


def _run_job(
    records: list[dict],
    proxies: list[str],
    *,
    source: str,
    proxy_mode: str,
    workers: int,
    delay: float,
    run_id: str,
) -> None:
    def _on_item(row, summary):
        with _lock:
            _state["progress"] = int(summary.get("total") or 0)
            _state["summary"] = dict(summary)
            _state["items"] = list(summary.get("items") or [])
        _persist_report(summary, source=source, proxy_mode=proxy_mode, run_id=run_id)

    try:
        summary = run_quality_scan(
            records,
            proxies,
            workers=workers,
            delay=delay,
            export=DEGRADED_EXPORT,
            risk_export=RISK_EXPORT,
            log=lambda _message: None,
            on_item=_on_item,
            cancel_callback=_cancel.is_set,
        )
        _persist_report(summary, source=source, proxy_mode=proxy_mode, run_id=run_id)
        with _lock:
            _state["summary"] = dict(summary)
            _state["items"] = list(summary.get("items") or [])
            _state["progress"] = int(summary.get("total") or 0)
            _state["cancelled"] = bool(summary.get("cancelled"))
            _state["error"] = ""
    except Exception as exc:
        with _lock:
            _state["error"] = redact_log_line(str(exc))[:240]
            _state["cancelled"] = _cancel.is_set()
    finally:
        _release_run(run_id)
        with _lock:
            _state["running"] = False
            _state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def start_quality_scan(
    *,
    source: str = "cpa",
    proxy: str = "",
    prefer_home: bool = True,
    workers: int = DEFAULT_WORKERS,
    delay: float = 0.0,
    limit: int = 0,
) -> dict:
    normalized = str(source or "cpa").strip().lower()
    if normalized not in VALID_SOURCES:
        return {"ok": False, "error": f"unknown source: {normalized}"}
    try:
        wait = max(0.0, min(10.0, float(delay)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid delay"}
    try:
        worker_n = max(1, min(8, int(workers or DEFAULT_WORKERS)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid workers"}
    try:
        cap = max(0, min(MAX_RECORDS, int(limit or 0)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid limit"}

    records = load_auth_records(_resolve_auth_dirs(normalized), limit=cap)
    if not records:
        return {"ok": False, "error": "没有可用的 CPA / Grok2API auth"}
    if len(records) > MAX_RECORDS:
        return {
            "ok": False,
            "error": f"一次最多检查 {MAX_RECORDS} 条，当前 {len(records)}",
        }

    proxies = resolve_probe_proxies(proxy, prefer_home=bool(prefer_home))
    proxy_mode = "explicit" if str(proxy or "").strip() else ("home" if prefer_home else "pool")
    run_id = secrets.token_hex(8)
    with _lock:
        if _state["running"]:
            return {"ok": False, "error": "降智测试已在运行", "running": True}
        if not _claim_run(run_id):
            return {"ok": False, "error": "另一面板进程正在执行降智测试", "running": True}
        _cancel.clear()
        _state.update(
            {
                "running": True,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "finished_at": "",
                "error": "",
                "cancelled": False,
                "source": normalized,
                "proxy_mode": proxy_mode,
                "progress": 0,
                "total": len(records),
                "summary": {},
                "items": [],
                "run_id": run_id,
            }
        )

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "records": records,
            "proxies": proxies,
            "source": normalized,
            "proxy_mode": proxy_mode,
            "workers": worker_n,
            "delay": wait,
            "run_id": run_id,
        },
        name="quality-scan",
        daemon=True,
    )
    global _thread
    _thread = thread
    thread.start()
    return {
        "ok": True,
        "running": True,
        "total": len(records),
        "source": normalized,
        "delay": wait,
        "workers": worker_n,
        "proxy_mode": proxy_mode,
        "proxy_count": len([item for item in proxies if item]),
        "run_id": run_id,
    }


def stop_quality_scan() -> dict:
    _cancel.set()
    thread = _thread
    return {"ok": True, "running": bool(thread and thread.is_alive())}


def read_quality_export(kind: str = "degraded") -> dict:
    path = RISK_EXPORT if str(kind or "") == "risk" else DEGRADED_EXPORT
    if not path.is_file():
        return {"ok": False, "error": "还没有导出文件", "content": "", "lines": 0}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": redact_log_line(str(exc)), "content": "", "lines": 0}
    lines = [line for line in content.splitlines() if line.strip()]
    return {"ok": True, "content": content, "lines": len(lines), "path": str(path)}
