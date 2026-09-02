#!/usr/bin/env python3
"""Batch-test Grok accounts for 降智 / chat risk through a residential proxy.

Usage:
  python scripts/check_quality.py --dir cpa_auth --from-config config.json
  python scripts/check_quality.py --dir cpa_auth --proxy http://127.0.0.1:8001 --workers 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_probe import (  # noqa: E402
    DEFAULT_WORKERS,
    load_auth_records,
    run_quality_scan,
)
from secure_files import atomic_write_json  # noqa: E402
from webui.quality_ops import resolve_probe_proxies  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Probe CPA/Grok2API accounts with a real streamed reply (家宽)"
    )
    ap.add_argument("--dir", action="append", default=[], help="auth directory (repeatable)")
    ap.add_argument("--from-config", metavar="FILE", help="read proxy / auth dirs from config.json")
    ap.add_argument("--proxy", default="", help="explicit HTTP/SOCKS proxy; empty = 家宽池")
    ap.add_argument("--no-home", action="store_true", help="do not prefer 家宽 ports")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds before each probe")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--export", metavar="FILE", help="degraded jsonl (no token)")
    ap.add_argument("--risk-export", metavar="FILE", help="risk/denied jsonl (no token)")
    ap.add_argument("--report-json", metavar="FILE", help="full summary JSON")
    args = ap.parse_args()

    dirs: list[Path] = []
    if args.from_config:
        cfg_path = Path(args.from_config).expanduser()
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            cfg = {}
        base = cfg_path.parent
        for key in ("cpa_auth_dir", "grok2api_auth_dir"):
            raw = str(cfg.get(key) or "").strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = base / path
            dirs.append(path)
        if not args.proxy:
            args.proxy = str(cfg.get("proxy") or "").strip() if args.no_home else ""
    for raw in args.dir:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        dirs.append(path)
    if not dirs:
        dirs = [ROOT / "cpa_auth"]

    records = load_auth_records(dirs, limit=max(0, int(args.limit or 0)))
    if not records:
        ap.error("没有可用的 auth 记录")

    proxies = resolve_probe_proxies(args.proxy, prefer_home=not args.no_home)
    summary = run_quality_scan(
        records,
        proxies,
        workers=args.workers,
        delay=args.delay,
        export=args.export,
        risk_export=args.risk_export,
    )
    if args.report_json:
        atomic_write_json(args.report_json, summary)
        print(f"报告 → {args.report_json}")
    print(
        f"降智测试: total={summary['total']} healthy={summary['healthy_count']} "
        f"degraded={summary['degraded_count']} (soft={summary['soft_count']} "
        f"hard={summary['hard_count']} burst={summary['burst_count']}) "
        f"risk={summary['risk_count']} err={summary['error_count']} "
        f"proxies={summary['proxy_count']}"
    )
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
