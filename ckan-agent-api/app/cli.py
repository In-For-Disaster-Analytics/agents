from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agents.ckan_registration.graph import get_runner
from app.agents.ckan_registration.schemas import CkanResumeRequest, CkanRunRequest


def load_payload(path: str | None) -> dict:
    if not path or path == "-":
        text = sys.stdin.read()
        return json.loads(text) if text.strip() else {}
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CKAN registration LangGraph runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start or continue a CKAN registration action.")
    run_parser.add_argument("--input", "-i", help="JSON request file. Use '-' or omit to read stdin.")

    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted registration thread.")
    resume_parser.add_argument("thread_id")
    resume_parser.add_argument("--input", "-i", help="JSON resume payload. Use '-' or omit to read stdin.")

    args = parser.parse_args(argv)
    runner = get_runner()

    if args.command == "run":
        result = runner.invoke(CkanRunRequest.model_validate(load_payload(args.input)))
    elif args.command == "resume":
        result = runner.resume(args.thread_id, CkanResumeRequest.model_validate(load_payload(args.input)))
    else:
        parser.error(f"Unsupported command: {args.command}")

    print(json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
