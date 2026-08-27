"""Stage 4 — Failure mode extraction (per problem, TWO-PASS).

Pass A (reasoning-anchored): given the correct reasoning trace, identify what can
go wrong ALONG the correct solution path.

Pass B (anticipatory, wrong-solver simulation): WITHOUT the reasoning trace,
simulate plausible-but-wrong first attempts. This is the pass that catches
failures that never appear on the correct path at all (wrong frame, wrong
algorithm, pattern overfitting) — critical for hard benchmarks.

Both passes are tagged with the Stage 3 normalized vocabulary for
``concept_involved``. Results are merged and deduplicated on
(failure_type, concept_involved): higher severity wins; on equal severity the
anticipatory (Pass B) entry wins. New concepts (is_new_concept=true) are
registered into the taxonomy file so Stage 5's graph includes them.

Resumability: problem_ids already present in the output file are skipped.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from string import Template
from typing import Any

from tqdm import tqdm

from utils.io import load_json, load_json_obj, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1}
REQUIRED_FM_FIELDS = (
    "failure_type",
    "description",
    "concept_involved",
    "what_correct_understanding_looks_like",
)

PROMPT_PASS_A = Template(r"""You are an expert AI evaluator and curriculum designer.

You have a problem, its correct reasoning trace, and its answer. Your task is to
identify the specific reasons why a capable but imperfect model or learner would
FAIL this problem — specifically, failures that occur ALONG or NEAR the correct
solution path.

Think carefully about each of these failure types:

MISSING_PREREQUISITE:
  A foundational concept required by the solution is not known at all.
  The solver cannot even begin the correct approach because they lack a building block.
  Example: Does not know modular inverse exists → cannot complete a number theory solution.

WRONG_MENTAL_MODEL:
  The concept is known but the learner has an incorrect or incomplete understanding
  of how it works in practice.
  Example: Knows binary search exists but believes it only works on strictly increasing
  arrays, so rejects it when duplicates are present.

MISSING_TRICK_OR_INSIGHT:
  A non-obvious insight is required that you either know or you do not — it cannot
  be derived by brute force reasoning alone in reasonable time.
  Example: "The answer is always the XOR of all elements" — you need to have seen
  this trick; a solver without it will spin indefinitely.

COMMON_MISTAKE:
  A mistake that is easy to make and looks almost right, that passes most test cases
  but fails on specific ones.
  Example: Using < instead of <= in a boundary check, causing an off-by-one that
  only manifests on inputs where the boundary is exactly hit.

FALSE_ASSUMPTION:
  The solver assumes something about the input or problem structure that is not
  guaranteed, producing a solution that passes most cases but fails when the
  assumption is violated.
  Example: Assuming the input array is always non-empty; assuming values are always
  positive; assuming the graph is always connected.

MISREAD_CONSTRAINTS:
  The problem statement contains a constraint that is easy to overlook or misread,
  and missing it leads to a fundamentally different (and wrong) solution.
  Example: "Return the indices, not the values" — solver returns values.
  Example: "All elements are distinct" is NOT stated, but solver assumes it.

MISSING_DOMAIN_KNOWLEDGE:
  Specialized knowledge outside of core algorithms is required.
  Example: Knowing Python float has 53-bit mantissa precision.
  Example: Knowing a specific mathematical identity or theorem by name.
  Example: Knowing that a particular graph structure guarantees a property.

SHORTCUT_ATTEMPT:
  The solver tries a simpler approach (greedy, brute force, heuristic) that seems
  to work on the examples provided but fails on edge cases or at scale.
  Example: Greedy interval selection when problem weights require DP.

Problem:
${question}

Correct Reasoning Trace:
${reasoning}

Correct Answer:
${answer}

Normalized concept vocabulary (use ONLY these for concept_involved, pick closest match;
if genuinely new, use same dot-notation format and set is_new_concept=true):
${normalized_taxonomy}

Return a JSON array of failure modes. Each entry:
{
  "failure_type": "MISSING_PREREQUISITE | WRONG_MENTAL_MODEL | MISSING_TRICK_OR_INSIGHT | COMMON_MISTAKE | FALSE_ASSUMPTION | MISREAD_CONSTRAINTS | MISSING_DOMAIN_KNOWLEDGE | SHORTCUT_ATTEMPT",
  "description": "1-2 sentences: exactly what the failure is for THIS specific problem, not generic",
  "concept_involved": "canonical.concept.from.taxonomy",
  "is_new_concept": true | false,
  "severity": "critical | major | minor",
  "what_correct_understanding_looks_like": "1 sentence: what the solver needs to know or do instead",
  "source": "reasoning_anchored"
}

Return 2-4 failure modes. Prioritize severity (critical first).
Return ONLY a valid JSON array. No explanation outside the array.""")

PROMPT_PASS_B = Template(r"""You are simulating a capable but imperfect AI model or learner attempting a ${domain}
problem cold — without any hints about the correct approach.

Your goal is to generate PLAUSIBLE WRONG SOLUTION ATTEMPTS: the kinds of approaches
that a model or learner would confidently try, often without realizing they are wrong,
before seeing the correct solution.

IMPORTANT: This is NOT about finding edge cases in the correct solution.
This IS about: what completely different, wrong approaches would seem reasonable to
try from the beginning?

You will be shown the correct answer ONLY for reference to verify your wrong attempts
are actually wrong. Do NOT let the correct answer anchor your thinking — generate
the wrong attempts first, then check they are wrong.

Problem:
${question}

Correct Answer (for reference only — do not anchor to this):
${answer}

Generate 3-5 distinct wrong solution attempts. Each attempt must represent a solver
who starts from a plausible but wrong frame or strategy. At least ONE attempt must
represent a solver who never gets close to the correct approach — not someone who
got 90% of the way there.

For each wrong attempt, identify its root cause from these categories:

WRONG_PROBLEM_FRAME:
  Solver misunderstood what the problem is asking at a fundamental level.
  They are solving a related but different problem than the one stated.
  Example: Problem asks for count of valid pairs; solver finds the pairs themselves.
  Example: Problem asks for minimum cost; solver finds any valid solution ignoring cost.

PLAUSIBLE_WRONG_ALGORITHM:
  Solver correctly recognized the problem domain but chose an algorithm that works
  on most cases but fails on this problem's specific constraints or structure.
  Example: Uses greedy for a problem that requires DP because of overlapping subproblems.
  Example: Uses BFS for a weighted shortest path problem (should use Dijkstra).

KNOWLEDGE_ILLUSION:
  Solver believes they know a theorem, formula, or rule and applies it, but their
  understanding is subtly wrong — the rule has a condition they are not checking.
  Example: Applying AM-GM without verifying non-negativity of terms.
  Example: Using the formula for combinations but forgetting the ordering constraint.
  Example: Thinking binary search works on any sequence, not just sorted ones.

PATTERN_OVERFITTING:
  Solver has seen similar-looking problems and applies the pattern from those problems
  without verifying it applies here. Surface features of the problem trigger a
  memorized (wrong) template.
  Example: Seeing "subarray" and immediately applying sliding window template, which
  requires non-negative values that this problem does not guarantee.
  Example: Seeing a tree and applying in-order traversal template when problem needs
  post-order.

COMPLEXITY_BLINDNESS:
  Solver produces a logically correct approach that is computationally infeasible
  for the given constraints, not recognizing that their solution will TLE.
  Or: solver "optimizes" their approach in a way that inadvertently changes its semantics.
  Example: Correct O(n^3) solution for n=10^5 input.
  Example: Caching results incorrectly, making a correct algorithm return stale values.

PHANTOM_CONSTRAINT:
  Solver adds a constraint that is NOT stated in the problem (because it appears in
  similar problems they have seen), artificially restricting their solution space.
  Example: Assuming graph is undirected when it is directed.
  Example: Assuming values are distinct when duplicates are allowed.
  Example: Assuming input is 1-indexed when it is 0-indexed.

TERMINATION_ERROR:
  Solver's approach and algorithm are directionally correct but the solution terminates
  too early, too late, or on the wrong condition — missing or overcounting results.
  Example: Returning on first match instead of continuing to find the best match.
  Example: Loop runs to n instead of n-1, processing a phantom element.

REPRESENTATION_ERROR:
  Solver chooses the wrong data structure or representation for the problem, causing
  a correct conceptual algorithm to produce wrong results or be infeasible.
  Example: Using adjacency matrix for a sparse graph with 10^5 nodes (memory blows up).
  Example: Storing cumulative counts when the problem requires point values.

For each wrong attempt, return:
{
  "attempt_description": "2-3 sentences: what the wrong solver does, specifically",
  "why_it_seems_reasonable": "1-2 sentences: why a capable solver would confidently try this",
  "wrong_answer_or_behavior": "what this approach produces or how it fails",
  "failure_type": "WRONG_PROBLEM_FRAME | PLAUSIBLE_WRONG_ALGORITHM | KNOWLEDGE_ILLUSION | PATTERN_OVERFITTING | COMPLEXITY_BLINDNESS | PHANTOM_CONSTRAINT | TERMINATION_ERROR | REPRESENTATION_ERROR",
  "description": "1-2 sentences: the precise failure for THIS specific problem",
  "concept_involved": "canonical.concept.from.taxonomy or new concept in dot-notation",
  "is_new_concept": true | false,
  "severity": "critical | major | minor",
  "what_correct_understanding_looks_like": "1 sentence: what the solver needs to know or do instead",
  "source": "anticipatory"
}

Normalized concept vocabulary (prefer these for concept_involved):
${normalized_taxonomy}

Return ONLY a valid JSON array of wrong attempts. No explanation outside the array.""")


def _sanitize(entries: Any, source: str) -> list[dict]:
    """Validate/normalize raw failure-mode entries from one pass."""
    if not isinstance(entries, list):
        if entries is not None:
            logger.debug("failure mode response was not a list: %r", str(entries)[:200])
        return []
    clean: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if any(not str(entry.get(field, "")).strip() for field in REQUIRED_FM_FIELDS):
            logger.debug("dropping malformed failure mode entry: %r", str(entry)[:200])
            continue
        fm = dict(entry)
        fm["failure_type"] = str(fm["failure_type"]).strip().upper()
        fm["concept_involved"] = str(fm["concept_involved"]).strip().lower()
        severity = str(fm.get("severity", "major")).strip().lower()
        fm["severity"] = severity if severity in SEVERITY_RANK else "major"
        fm["is_new_concept"] = bool(fm.get("is_new_concept", False))
        fm["source"] = source
        clean.append(fm)
    return clean


def _merge(pass_a: list[dict], pass_b: list[dict]) -> list[dict]:
    """Merge both passes, deduplicating on (failure_type, concept_involved).

    Higher severity wins; on equal severity the anticipatory (Pass B) entry wins
    (anticipatory failures are harder to generate and more valuable).
    """
    best: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for fm in [*pass_a, *pass_b]:
        key = (fm["failure_type"], fm["concept_involved"])
        current = best.get(key)
        if current is None:
            best[key] = fm
            order.append(key)
            continue
        current_rank = SEVERITY_RANK[current["severity"]]
        challenger_rank = SEVERITY_RANK[fm["severity"]]
        if challenger_rank > current_rank:
            best[key] = fm
        elif challenger_rank == current_rank and fm["source"] == "anticipatory":
            best[key] = fm
    return [best[key] for key in order]


async def run(reasoning_path: Path, taxonomy_path: Path, output_path: Path, domain: str) -> None:
    """Run Stage 4 (two-pass) for one domain."""
    require_file(
        reasoning_path,
        f"(run stage 1 first for domain '{domain}')",
    )
    require_file(
        taxonomy_path,
        f"(run stage 3 first for domain '{domain}')",
    )
    records = load_json(reasoning_path)
    ok_records = [r for r in records if r.get("reasoning_status") == "ok"]

    taxonomy_doc = load_json_obj(taxonomy_path) or {}
    state: dict[str, Any] = {
        "taxonomy": [str(c).strip() for c in taxonomy_doc.get("taxonomy", []) if str(c).strip()],
        "merge_map": {str(k): str(v) for k, v in (taxonomy_doc.get("merge_map") or {}).items()},
        "doc": taxonomy_doc,
    }

    existing = load_json(output_path)
    by_id: dict[str, dict] = {
        str(r["problem_id"]): r
        for r in existing
        if r.get("problem_id") is not None
    }
    existing_ids = set(by_id)
    pending = [
        r for r in ok_records if str(r.get("problem_id")) not in existing_ids
    ]

    logger.info(
        "Stage 4 [%s]: %d ok record(s), %d already done, %d to process, "
        "%d taxonomy concept(s)",
        domain, len(ok_records), len(existing_ids), len(pending), len(state["taxonomy"]),
    )

    lock = asyncio.Lock()
    tax_lock = asyncio.Lock()
    counters = {"success": 0, "failed": 0}
    pbar = tqdm(total=len(pending), desc=f"Stage 4 [{domain}] failure modes (2-pass)", unit="prob")
    pbar.set_postfix(skip=len(existing_ids))

    async def vocabulary_string() -> str:
        async with tax_lock:
            if not state["taxonomy"]:
                return "(empty taxonomy)"
            return "\n".join(f"- {c}" for c in state["taxonomy"])

    async def register_new_concepts(fms: list[dict]) -> None:
        """Add is_new_concept=true concepts to the taxonomy file (Stage 5 needs them)."""
        new_names = [
            fm["concept_involved"]
            for fm in fms
            if fm.get("is_new_concept") and fm["concept_involved"]
        ]
        if not new_names:
            return
        added = 0
        async with tax_lock:
            for name in new_names:
                if name not in state["merge_map"]:
                    state["merge_map"][name] = name
                    state["taxonomy"].append(name)
                    added += 1
            if added:
                state["doc"]["taxonomy"] = state["taxonomy"]
                state["doc"]["merge_map"] = state["merge_map"]
                save_json(taxonomy_path, state["doc"])
        if added:
            logger.info(
                "Stage 4 [%s]: registered %d new concept(s) in taxonomy", domain, added
            )

    async def worker(record: dict) -> None:
        pid = str(record.get("problem_id"))
        vocabulary = await vocabulary_string()
        prompt_a = PROMPT_PASS_A.safe_substitute(
            question=str(record.get("question", "")),
            reasoning=str(record.get("reasoning", "")),
            answer=str(record.get("answer", "")),
            normalized_taxonomy=vocabulary,
        )
        prompt_b = PROMPT_PASS_B.safe_substitute(
            domain=domain,
            question=str(record.get("question", "")),
            answer=str(record.get("answer", "")),
            normalized_taxonomy=vocabulary,
        )
        try:
            raw_a, raw_b = await asyncio.gather(
                call_llm(prompt_a, expect_json=True),
                call_llm(prompt_b, expect_json=True),
            )
        except LLMError as exc:
            logger.error(
                "Stage 4 [%s] DISCARD problem %s: pass LLM call failed: %s",
                domain, pid, exc,
            )
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
            return

        pass_a = _sanitize(raw_a, "reasoning_anchored")
        pass_b = _sanitize(raw_b, "anticipatory")
        merged = _merge(pass_a, pass_b)
        if not merged:
            logger.error(
                "Stage 4 [%s] DISCARD problem %s: both passes produced no usable "
                "failure modes (pass A: %d, pass B: %d entries)",
                domain, pid, len(pass_a), len(pass_b),
            )
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
            return

        await register_new_concepts(merged)
        if not 4 <= len(merged) <= 8:
            logger.warning(
                "Stage 4 [%s] problem %s: %d failure mode(s) after merge (target 4-8)",
                domain, pid, len(merged),
            )

        out = {
            "benchmark": record.get("benchmark"),
            "sub_benchmark": record.get("sub_benchmark"),
            "problem_id": record.get("problem_id"),
            "question": record.get("question"),
            "answer": record.get("answer"),
            "reasoning": record.get("reasoning"),
            "reasoning_status": record.get("reasoning_status"),
            "failure_modes": merged,
        }
        async with lock:
            by_id[pid] = out
            save_json(output_path, list(by_id.values()))
            counters["success"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
        logger.debug(
            "Stage 4 [%s] problem %s: %d merged failure mode(s)", domain, pid, len(merged)
        )

    if pending:
        await asyncio.gather(*(worker(r) for r in pending))
    pbar.close()
    if not output_path.exists():
        save_json(output_path, [])  # keep downstream stages runnable on empty input
    logger.info(
        "Stage 4 [%s] complete: %d record(s) with failure modes in output",
        domain, len(by_id),
    )
