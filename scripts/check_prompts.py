"""CI check: prompts referenced by agent configs exist, are non-empty, and still contain
the markers the code depends on."""

from __future__ import annotations

import sys
from pathlib import Path

from gateway.config.loader import load_agent_config

REQUIRED = {
    "orchestrator": ["[[TITLE:", "<title_requested/>", "<previous_sources>", "progress_title"],
    "followups": ["[[FOLLOWUPS]]"],
    "guardrails": ["allowed", "prompt_injection", "harmful", "sensitive_data", "off_topic"],
}


def main() -> int:
    errors = []
    for cfg_path in sorted(Path("configs/agents").glob("*.yaml")):
        cfg = load_agent_config(cfg_path, environ={})
        for kind, path in cfg.prompts.model_dump().items():
            p = Path(path)
            if not p.is_file() or not p.read_text().strip():
                errors.append(f"{cfg_path}: prompt {kind} missing or empty at {path}")
                continue
            text = p.read_text()
            errors += [
                f"{path}: missing required marker {m!r}" for m in REQUIRED[kind] if m not in text
            ]
    for e in errors:
        print("ERROR", e)
    print("prompt checks:", "failed" if errors else "ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
