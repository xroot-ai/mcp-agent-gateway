"""Run the eval suites against a running gateway and write a report.

    python -m evals.run --base-url http://localhost:8000 --token "$EVAL_TOKEN" --suite all

Answers are graded by an LLM judge (structured verdict) plus deterministic checks.
Every run calls the model API for both the gateway and the judge, so it costs tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
JUDGE_MODEL = "claude-opus-5-5"

JUDGE_SYSTEM = """You grade answers from a Kubernetes documentation assistant.
You get the question, the facts a correct answer must contain, the assistant's answer, and the
source passages it cited. Judge only against the expected facts and the cited sources.
- answer_correct: the answer states every expected fact (wording may differ) and states nothing
  that contradicts them.
- grounded: every substantive claim in the answer is supported by the cited source snippets or
  is general knowledge needed to connect them.
Keep the reason to one sentence."""


class Grade(BaseModel):
    answer_correct: bool
    grounded: bool
    reason: str = Field(max_length=400)


@dataclass
class RunResult:
    events: list[dict[str, Any]]
    conversation_id: str | None = None

    @property
    def final(self) -> dict[str, Any]:
        return self.events[-1] if self.events else {}

    @property
    def kind(self) -> str:
        return self.final.get("type", "none")


@dataclass
class SuiteReport:
    name: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_jsonl(name: str) -> list[dict[str, Any]]:
    path = HERE / "datasets" / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def ask(
    http: httpx.AsyncClient, message: str, conversation_id: str | None = None
) -> RunResult:
    body: dict[str, Any] = {"message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    events: list[dict[str, Any]] = []
    async with http.stream("POST", "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    conv = next((e["conversation_id"] for e in events if e["type"] == "run.started"), None)
    return RunResult(events, conv)


async def judge(
    client: AsyncAnthropic, question: str, expected: list[str], result: RunResult
) -> Grade:
    final = result.final
    sources = "\n".join(
        f"[{s['number']}] {s['title']} > {s.get('section', '')}: {s.get('snippet', '')}"
        for s in final.get("sources", [])
    )
    content = (
        f"<question>{question}</question>\n"
        f"<expected_facts>\n" + "\n".join(f"- {f}" for f in expected) + "\n</expected_facts>\n"
        f"<answer>{final.get('answer', '')}</answer>\n<sources>\n{sources}\n</sources>"
    )
    resp = await client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=2048,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": content}],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        output_format=Grade,
    )
    if resp.parsed_output is None:
        return Grade(answer_correct=False, grounded=False, reason=f"judge stop={resp.stop_reason}")
    return resp.parsed_output


def cites(result: RunResult, fragment: str | None) -> bool:
    if not fragment:
        return True
    return any(fragment in s["url"] for s in result.final.get("sources", []))


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0


async def rag_suite(
    http: httpx.AsyncClient, judge_client: AsyncAnthropic, limit: int | None
) -> SuiteReport:
    report = SuiteReport("rag")
    for case in load_jsonl("rag.jsonl")[:limit]:
        result = await ask(http, case["question"])
        grade = (
            await judge(judge_client, case["question"], case["expected_facts"], result)
            if result.kind == "run.completed"
            else Grade(answer_correct=False, grounded=False, reason=result.kind)
        )
        report.rows.append(
            {
                "id": case["id"],
                "status": result.kind,
                "answer_correct": grade.answer_correct,
                "grounded": grade.grounded,
                "cites_expected_source": cites(result, case.get("source_url")),
                "reason": grade.reason,
            }
        )
    for key in ("answer_correct", "grounded", "cites_expected_source"):
        report.metrics[key] = rate(report.rows, key)
    return report


async def multiturn_suite(
    http: httpx.AsyncClient, judge_client: AsyncAnthropic, limit: int | None
) -> SuiteReport:
    report = SuiteReport("multiturn")
    for case in load_jsonl("multiturn.jsonl")[:limit]:
        conv = None
        result = RunResult([])
        for turn in case["turns"]:
            result = await ask(http, turn, conv)
            conv = result.conversation_id
        grade = (
            await judge(judge_client, " / ".join(case["turns"]), case["expected_facts"], result)
            if result.kind == "run.completed"
            else Grade(answer_correct=False, grounded=False, reason=result.kind)
        )
        report.rows.append(
            {
                "id": case["id"],
                "status": result.kind,
                "answer_correct": grade.answer_correct,
                "reason": grade.reason,
            }
        )
    report.metrics["answer_correct"] = rate(report.rows, "answer_correct")
    return report


async def guardrails_suite(http: httpx.AsyncClient, limit: int | None) -> SuiteReport:
    report = SuiteReport("guardrails")
    for case in load_jsonl("guardrails.jsonl")[:limit]:
        result = await ask(http, case["message"])
        blocked = result.kind == "run.blocked"
        expect_block = case["expect"] == "block"
        report.rows.append(
            {
                "id": case["id"],
                "expect": case["expect"],
                "status": result.kind,
                "correct": blocked == expect_block,
                "false_block": blocked and not expect_block,
                "category": result.final.get("category", ""),
            }
        )
    report.metrics["accuracy"] = rate(report.rows, "correct")
    allowed = [r for r in report.rows if r["expect"] == "allow"]
    report.metrics["false_block_rate"] = rate(allowed, "false_block")
    return report


def check(reports: list[SuiteReport]) -> list[str]:
    bars = yaml.safe_load((HERE / "thresholds.yaml").read_text())
    failures = []
    for rep in reports:
        for key, bar in bars.get(rep.name, {}).items():
            if key.endswith("_max"):
                value = rep.metrics.get(key.removesuffix("_max"), 1.0)
                if value > bar:
                    failures.append(f"{rep.name}.{key.removesuffix('_max')}={value:.2f} > {bar}")
            elif rep.metrics.get(key, 0.0) < bar:
                failures.append(f"{rep.name}.{key}={rep.metrics.get(key, 0.0):.2f} < {bar}")
    return failures


def write_report(reports: list[SuiteReport], failures: list[str]) -> Path:
    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    lines = [f"# Eval report {stamp}", "", "| suite | metric | value |", "|---|---|---|"]
    for rep in reports:
        for k, v in rep.metrics.items():
            lines.append(f"| {rep.name} | {k} | {v:.0%} |")
    lines += [
        "",
        "## Failures" if failures else "## All thresholds met",
        *[f"- {f}" for f in failures],
    ]
    for rep in reports:
        lines += ["", f"## {rep.name} cases", ""]
        lines += [
            f"- `{r['id']}` {json.dumps({k: v for k, v in r.items() if k != 'id'})}"
            for r in rep.rows
        ]
    path = out_dir / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    (out_dir / f"{stamp}.json").write_text(
        json.dumps({r.name: {"metrics": r.metrics, "rows": r.rows} for r in reports}, indent=2)
    )
    return path


async def main_async(args: argparse.Namespace) -> int:
    headers = {"Authorization": f"Bearer {args.token}"}
    suites = ["rag", "multiturn", "guardrails"] if args.suite == "all" else [args.suite]
    judge_client = AsyncAnthropic()
    reports = []
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=180) as http:
        if "rag" in suites:
            reports.append(await rag_suite(http, judge_client, args.limit))
        if "multiturn" in suites:
            reports.append(await multiturn_suite(http, judge_client, args.limit))
        if "guardrails" in suites:
            reports.append(await guardrails_suite(http, args.limit))
    failures = check(reports)
    path = write_report(reports, failures)
    for rep in reports:
        print(rep.name, {k: f"{v:.0%}" for k, v in rep.metrics.items()})
    print(f"report: {path}")
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--token", required=True)
    ap.add_argument("--suite", choices=["all", "rag", "multiturn", "guardrails"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases per suite")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
