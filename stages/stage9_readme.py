"""Stage 9 — README auto-generation.

Renders README.md (overwriting any previous version) with live statistics read
from the input files, taxonomy files, concept graph files, and output files.
No LLM calls.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

import config
from utils.io import load_json, load_json_obj

logger = logging.getLogger(__name__)

STAGES_TABLE = """| Stage | Description | Input | Output | Estimated LLM Calls per 100 Problems |
| --- | --- | --- | --- | --- |
| 1 | Reasoning generation | *.json | *_with_reasoning.json | 100 |
| 2 | Raw concept extraction | *_with_reasoning.json | *_raw_concepts.json | 100 |
| 3 | Taxonomy normalization | *_raw_concepts.json | *_taxonomy.json | 2 (batch, once) |
| 4 | Failure mode extraction (2-pass) | *_with_reasoning.json | *_with_failure_modes.json | 200 |
| 5 | Concept graph construction | *_taxonomy.json | *_concept_graph.json | ~200 (once per concept) |
| 6 | Question generation | *_with_failure_modes.json | *_questions_raw.json | ~400 |
| 7 | Validation and dedup | *_questions_raw.json | *_questions_validated.json | ~400 |
| 8 | Output assembly | *_questions_validated.json | *_prerequisite_questions.json | 0 |
| 9 | README generation | all stats | README.md | 0 |"""

INPUT_FORMAT = """### Record Format

Each input file is a JSON **array of records**. All five fields below are read
by the pipeline; `problem_id` is the resume key, so it must be unique within a
file (and ideally unique across both files).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `benchmark` | string | yes | Benchmark name, e.g. `"HumanEval"`, `"AIME"`, `"GPQA"` |
| `sub_benchmark` | string or null | no | Subset/partition of the benchmark, e.g. `"2024-I"`; use `null` when there is none |
| `problem_id` | string | yes | Unique problem identifier — convention: `"<benchmark>/<id>"` |
| `question` | string | yes | The full problem statement |
| `answer` | string | yes | The gold answer — for coding problems the solution implementation (one-liner or multi-line function body/snippet), for reasoning problems the exact answer. It grounds and verifies Stage 1: the trace ends with "Therefore, the answer is: <answer>" and is verified against the gold answer |

Example `coding.json`:

```json
[
  {
    "benchmark": "HumanEval",
    "sub_benchmark": null,
    "problem_id": "HumanEval/0",
    "question": "def has_close_elements(solution: list[float], threshold: float) -> bool:\\n    # Write a function that returns true if the given list has two elements\\n    # that are closer to each other than threshold.",
    "answer": "any(abs(a - b) < threshold for i, a in enumerate(solution) for b in solution[i + 1:])"
  }
]
```

Example `reasoning.json`:

```json
[
  {
    "benchmark": "AIME",
    "sub_benchmark": "2024-I",
    "problem_id": "AIME/2024-I-1",
    "question": "Let X be the number of ordered triples (a, b, c) of positive integers ...",
    "answer": "208"
  }
]
```
"""

BUDGET_BLOCK = """Per 100 source problems:
  Stage 1 (reasoning generation):       100 calls
  Stage 2 (concept extraction):         100 calls
  Stage 3 (taxonomy normalization):       2 calls (batched, run once)
  Stage 4 (failure modes, 2-pass):       200 calls (2 per problem)
  Stage 5 (concept graph):              ~200 calls (once per unique concept)
  Stage 6 (question generation):        ~400 calls (avg 4 failure modes per problem)
  Stage 7 (validation):                 ~400 calls
  Stage 8 (output assembly):              0 calls
  Stage 9 (README):                       0 calls
  Total:                              ~1,400 calls per 100 source problems"""

CODING_SCHEMA = '''```json
[
  {
    "id": "uuid4",
    "source_benchmark": "HumanEval",
    "source_sub_benchmark": null,
    "source_problem_id": "HumanEval/42",
    "domain": "coding",
    "failure_type": "COMMON_MISTAKE",
    "failure_source": "reasoning_anchored | anticipatory",
    "failure_description": "...",
    "concept_involved": "algorithms.technique.two-pointer",
    "prerequisite_depth": 1,
    "question": "...",
    "what_it_tests": "...",
    "trap": "...",
    "why_trap_is_tempting": "...",
    "difficulty": "beginner | intermediate | advanced",
    "tags": ["algorithms.technique.two-pointer", "data-structures.array.indexing"],
    "generated_answer": null
  }
]
```

`generated_answer` is null for coding — answers will be generated separately.
`prerequisite_depth` is the shortest path in the concept graph from
`concept_involved` to the source problem's primary concept (1 = direct
prerequisite, 2 = prerequisite of a prerequisite, 0 = same level / no path).'''

REASONING_SCHEMA = '''```json
[
  {
    "id": "uuid4",
    "source_benchmark": "AIME",
    "source_sub_benchmark": null,
    "source_problem_id": "AIME/2024-I-1",
    "domain": "reasoning",
    "failure_type": "COMMON_MISTAKE",
    "failure_source": "reasoning_anchored | anticipatory",
    "failure_description": "...",
    "concept_involved": "combinatorics.counting.inclusion-exclusion",
    "prerequisite_depth": 1,
    "question": "...",
    "answer": "the exact correct answer",
    "answer_explanation": "2-3 sentence explanation",
    "what_it_tests": "...",
    "trap": "...",
    "why_trap_is_tempting": "...",
    "difficulty": "beginner | intermediate | advanced",
    "tags": ["combinatorics.counting.inclusion-exclusion", "logic.proof-technique.contradiction"],
    "generated_answer": null
  }
]
```

Reasoning questions include `answer` and `answer_explanation` inline since they
are verifiable during generation. All other fields match the coding schema.'''


def _input_row(name: str, path: str | Path | None) -> str:
    """One row of the Input Files table."""
    if path is None:
        path = config.PIPELINE_DIR / f"{name}.json"
    path = Path(path)
    if not path.exists():
        return f"| {name} | - | - |"
    records = load_json(path)
    benchmarks = sorted(
        {str(r.get("benchmark")) for r in records if r.get("benchmark")}
    )
    if not benchmarks:
        bench_cell = "-"
    else:
        shown = ", ".join(benchmarks[:5])
        if len(benchmarks) > 5:
            shown += f" (+{len(benchmarks) - 5} more)"
        bench_cell = f"{shown} ({len(benchmarks)})"
    return f"| {name} | {len(records)} | {bench_cell} |"


def _taxonomy_section(domain: str, label: str) -> str:
    doc = load_json_obj(config.taxonomy_file(domain))
    if not isinstance(doc, dict) or not doc.get("taxonomy"):
        return f"### {label}\n\n(Taxonomy not generated yet.)"
    taxonomy = doc["taxonomy"]
    summary = doc.get("category_summary") or {}
    categories = sorted(
        ((str(k), int(v)) for k, v in summary.items() if str(v).isdigit()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    lines = [
        f"### {label}",
        "",
        f"{label} domain: {len(taxonomy)} concepts across {len(categories)} "
        f"top-level categories",
        "",
        "| Category | Count |",
        "| --- | --- |",
    ]
    for category, count in categories:
        lines.append(f"| {category} | {count} |")
    return "\n".join(lines)


def _graph_line(domain: str, label: str) -> str:
    graph = load_json_obj(config.concept_graph_file(domain))
    if not isinstance(graph, dict):
        return f"{label}: (concept graph not generated yet)"
    return (
        f"{label}: {len(graph.get('nodes', []))} nodes, "
        f"{len(graph.get('edges', []))} edges, "
        f"{len(graph.get('leaves', []))} leaf concepts, "
        f"{len(graph.get('removed_cycles', []))} cycles removed"
    )


def _stats_section(domain: str, label: str) -> str:
    raw = load_json(config.questions_raw_file(domain))
    final = load_json(config.final_output_file(domain))
    if not raw and not final:
        return f"### {label}\n\n(No questions generated yet.)"

    total = len(raw)
    passed = len(final)
    pct = (100.0 * passed / total) if total else 0.0
    anchored = sum(1 for q in final if q.get("failure_source") == "reasoning_anchored")
    anticipatory = sum(1 for q in final if q.get("failure_source") == "anticipatory")

    by_benchmark = Counter(str(q.get("source_benchmark") or "unknown") for q in final)
    by_failure_type = Counter(str(q.get("failure_type") or "unknown") for q in final)
    by_difficulty = Counter(str(q.get("difficulty") or "unknown") for q in final)

    def fmt(counter: Counter) -> str:
        return "; ".join(f"{k}: {v}" for k, v in sorted(counter.items())) or "-"

    return "\n".join(
        [
            f"### {label}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total questions generated | {total} |",
            f"| Passed validation | {passed} ({pct:.1f}%) |",
            f"| From reasoning-anchored failures | {anchored} |",
            f"| From anticipatory failures | {anticipatory} |",
            f"| By benchmark | {fmt(by_benchmark)} |",
            f"| By failure_type | {fmt(by_failure_type)} |",
            f"| By difficulty | {fmt(by_difficulty)} |",
        ]
    )


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def build_readme(input_paths: dict[str, str | Path | None]) -> str:
    """Render the full README.md content from live pipeline artifacts."""
    lines: list[str] = []
    lines.append("# Prerequisite Question Bank — Generation Pipeline")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This pipeline converts raw benchmark problems (coding and reasoning) into a bank "
        "of targeted diagnostic prerequisite questions, each designed to expose one specific "
        "reason a model or learner would fail the source problem. For every source problem it "
        "generates a gold-grounded reasoning trace, extracts the concepts the solution "
        "requires, normalizes them into a domain-wide controlled vocabulary, builds a "
        "prerequisite concept graph, and then writes an easier, standalone question per "
        "identified failure mode. Failure modes are extracted in two passes — "
        "reasoning-anchored and anticipatory wrong-solver simulation — because on hard "
        "benchmarks models typically fail by going off track from the first step, a failure "
        "class that cannot be discovered by analyzing the correct solution alone."
    )
    lines.append("")
    lines.append("## Input Files")
    lines.append("")
    lines.append("| File | Records | Benchmarks Covered |")
    lines.append("| --- | --- | --- |")
    lines.append(_input_row("coding.json", input_paths.get("coding")))
    lines.append(_input_row("reasoning.json", input_paths.get("reasoning")))
    lines.append("")
    lines.append(INPUT_FORMAT)
    lines.append("")
    lines.append("## How to Run")
    lines.append("")
    lines.append("Place your input files `coding.json` and `reasoning.json` in the")
    lines.append("project root (record format above), then run:")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -r requirements.txt")
    lines.append("python run_pipeline.py")
    lines.append("```")
    lines.append("")
    lines.append("Optional flags:")
    lines.append("")
    lines.append("```text")
    lines.append("  --stages 1,2,3      # run only specific stages (comma-separated)")
    lines.append("  --domain coding     # run for one domain only")
    lines.append("  --concurrency 30    # max concurrent LLM requests (default: 30)")
    lines.append("  --resume            # default: always on; completed records are always skipped")
    lines.append("```")
    lines.append("")
    lines.append("## Pipeline Stages")
    lines.append("")
    lines.append(STAGES_TABLE)
    lines.append("")
    lines.append("## Concept Taxonomy")
    lines.append("")
    lines.append(_taxonomy_section("coding", "Coding"))
    lines.append("")
    lines.append(_taxonomy_section("reasoning", "Reasoning"))
    lines.append("")
    lines.append("## Concept Graph")
    lines.append("")
    lines.append(_graph_line("coding", "Coding"))
    lines.append("")
    lines.append(_graph_line("reasoning", "Reasoning"))
    lines.append("")
    lines.append("## Output Statistics")
    lines.append("")
    lines.append(_stats_section("coding", "Coding"))
    lines.append("")
    lines.append(_stats_section("reasoning", "Reasoning"))
    lines.append("")
    lines.append("## Output Schema")
    lines.append("")
    lines.append("### coding_prerequisite_questions.json")
    lines.append("")
    lines.append(CODING_SCHEMA)
    lines.append("")
    lines.append("### reasoning_prerequisite_questions.json")
    lines.append("")
    lines.append(REASONING_SCHEMA)
    lines.append("")
    lines.append("## Estimated LLM Call Budget")
    lines.append("")
    lines.append(BUDGET_BLOCK)
    lines.append("")
    return "\n".join(lines)


async def run(
    input_paths: dict[str, str | Path | None],
    concurrency: int | None = None,
) -> None:
    """Render and write README.md from the current pipeline state."""
    content = build_readme(input_paths)
    _write_text_atomic(config.README_PATH, content)
    logger.info("Stage 9 complete: README.md generated at %s", config.README_PATH)
