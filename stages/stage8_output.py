"""Stage 8 — Final output assembly.

Filters to validation_passed=true records and computes ``prerequisite_depth``:
the shortest path length from the question's concept_involved up to the source
problem's primary concept (normalized_concepts[0]) in the Stage 5 concept
graph. 1 = direct prerequisite, 2 = prerequisite of a prerequisite, 3 = deepest
tracked. If the concept is the primary concept itself or no path exists within
depth 3, depth is 0 (same level / unknown).

coding_prerequisite_questions.json carries ``generated_answer: null`` (coding
answers are generated separately). reasoning_prerequisite_questions.json
additionally carries ``answer`` and ``answer_explanation`` (included inline
because they are verifiable at generation time).

This stage is deterministic and idempotent (no LLM calls), so it simply
recomputes and rewrites the final file on every run.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import config
from utils.io import load_json, load_json_obj, require_file, save_json

logger = logging.getLogger(__name__)

MAX_DEPTH = 3


def _depths_from(
    start: str, adj: dict[str, list[str]], max_depth: int = MAX_DEPTH
) -> dict[str, int]:
    """BFS distance map from *start* following prerequisite edges (1..max_depth)."""
    dist = {start: 0}
    frontier = {start}
    for depth in range(1, max_depth + 1):
        level: set[str] = set()
        for current in frontier:
            for prereq in adj.get(current, []):
                if prereq not in dist:
                    dist[prereq] = depth
                    level.add(prereq)
        if not level:
            break
        frontier = level
    return dist


async def run(
    validated_path: Path,
    raw_concepts_path: Path,
    graph_path: Path,
    output_path: Path,
    domain: str,
    concurrency: int | None = None,
) -> None:
    """Run Stage 8 for one domain."""
    require_file(
        validated_path,
        f"(run stage 7 first for domain '{domain}')",
    )
    validated = load_json(validated_path)
    passed = [q for q in validated if q.get("validation_passed") is True]
    logger.info(
        "Stage 8 [%s]: %d validated record(s), %d passed, assembling final output",
        domain, len(validated), len(passed),
    )

    # Primary concept per source problem = first normalized concept (Stage 2/3).
    records = load_json(raw_concepts_path)
    primary_by_pid: dict[str, str] = {}
    for record in records:
        normalized = record.get("normalized_concepts") or []
        if normalized:
            primary_by_pid[str(record.get("problem_id"))] = str(normalized[0])

    graph = load_json_obj(graph_path) or {}
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and edge.get("from") and edge.get("to"):
            adj[str(edge["from"])].add(str(edge["to"]))
    adj = {k: sorted(v) for k, v in adj.items()}

    depth_cache: dict[str, dict[str, int]] = {}
    final: list[dict] = []
    for q in passed:
        pid = str(q.get("source_problem_id"))
        concept = str(q.get("concept_involved", "")).lower()
        primary = primary_by_pid.get(pid)

        depth = 0
        if primary and concept and concept != primary:
            if primary not in depth_cache:
                depth_cache[primary] = _depths_from(primary, adj, MAX_DEPTH)
            depth = depth_cache[primary].get(concept, 0)

        out = {
            "id": q.get("id"),
            "source_benchmark": q.get("source_benchmark"),
            "source_sub_benchmark": q.get("source_sub_benchmark"),
            "source_problem_id": q.get("source_problem_id"),
            "domain": domain,
            "failure_type": q.get("failure_type"),
            "failure_source": q.get("failure_source"),
            "failure_description": q.get("failure_description"),
            "concept_involved": concept,
            "prerequisite_depth": depth,
            "question": q.get("question"),
            "what_it_tests": q.get("what_it_tests"),
            "trap": q.get("trap"),
            "why_trap_is_tempting": q.get("why_trap_is_tempting"),
            "difficulty": q.get("difficulty"),
            "tags": q.get("tags", []),
        }
        if domain == "coding":
            # Coding answers are generated separately.
            out["generated_answer"] = None
        else:
            out["generated_answer"] = None
            out["answer"] = q.get("answer", "")
            out["answer_explanation"] = q.get("answer_explanation", "")
        final.append(out)

    save_json(output_path, final)  # deterministic: rewritten (possibly empty) each run
    logger.info(
        "Stage 8 [%s] complete: %d question(s) written to %s",
        domain, len(final), output_path,
    )
