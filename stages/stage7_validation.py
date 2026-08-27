"""Stage 7 — Validation and deduplication.

Step 7.1 — Contamination check: token-level Jaccard similarity of each generated
question against ALL source benchmark questions (whitespace tokenization).
Similarity > SIMILARITY_CONTAMINATION_THRESHOLD (0.85) → discard (kept in the
validated file with validation_passed=false and the reason).

Step 7.2 — Deduplication within generated questions: for pairs sharing the same
concept_involved, Jaccard > SIMILARITY_DEDUP_THRESHOLD (0.92) → discard the one
from the lower-severity source failure mode; on equal severity discard the
reasoning_anchored one (keep anticipatory — harder to generate, more novel).

Step 7.3 — LLM validation (Prompt S7): discrimination / isolation / drift /
trap_validity. A question passes only if discrimination=pass AND isolation=pass
AND drift is no_drift|minor_drift AND trap_validity=valid.

All records (passed and failed) are kept in the validated file; Stage 8 filters
to validation_passed=true. Resumability: questions whose id is already in the
validated file are skipped; saved after every record.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path

import config
from tqdm import tqdm

from prompts.stage7_validation import PROMPT_VALIDATE
from utils.constants import SEVERITY_RANK
from utils.io import load_json, require_file, save_json
from utils.llm import LLMError, call_llm
from utils.similarity import max_jaccard_against, tokenize

logger = logging.getLogger(__name__)


def _sort_key(q: dict):
    """Deterministic ordering for dedup tie-breaks (earlier problem wins)."""
    return (
        str(q.get("source_problem_id", "")),
        int(q.get("failure_index", 0) or 0),
        str(q.get("id", "")),
    )


def _pick_loser(a: dict, b: dict) -> tuple[dict, dict]:
    """Return (loser, winner) for a near-duplicate pair.

    Higher severity wins; on equal severity the anticipatory source wins; on a
    full tie the earlier question (stable order) is kept.
    """
    rank_a = SEVERITY_RANK.get(str(a.get("failure_severity", "major")).lower(), 1)
    rank_b = SEVERITY_RANK.get(str(b.get("failure_severity", "major")).lower(), 1)
    if rank_a != rank_b:
        return (a, b) if rank_a < rank_b else (b, a)
    a_ant = a.get("failure_source") == "anticipatory"
    b_ant = b.get("failure_source") == "anticipatory"
    if a_ant != b_ant:
        return (b, a) if a_ant else (a, b)
    return (b, a) if _sort_key(a) > _sort_key(b) else (a, b)


def _contamination_discards(
    questions: list[dict],
    source_tokens: list[tuple[str, frozenset[str]]],
) -> dict[str, str]:
    """Step 7.1: Jaccard against every source benchmark question."""
    discards: dict[str, str] = {}
    for q in questions:
        best, label = max_jaccard_against(
            tokenize(q.get("question", "")),
            source_tokens,
            config.SIMILARITY_CONTAMINATION_THRESHOLD,
        )
        if best > config.SIMILARITY_CONTAMINATION_THRESHOLD:
            reason = (
                f"contamination: Jaccard {best:.3f} > "
                f"{config.SIMILARITY_CONTAMINATION_THRESHOLD} with source question "
                f"{label or 'unknown'}"
            )
            discards[str(q.get("id"))] = reason
            logger.warning("Stage 7 DISCARD (contamination) %s: %s", q.get("id"), reason)
    return discards


def _dedup_discards(questions: list[dict]) -> dict[str, str]:
    """Step 7.2: within-concept near-duplicate removal."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        concept = str(q.get("concept_involved", "")).lower()
        groups[concept].append(q)

    discards: dict[str, str] = {}
    for concept, members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=_sort_key)
        tokens = {str(q.get("id")): tokenize(q.get("question", "")) for q in members}
        alive: set[str] = {str(q.get("id")) for q in members}
        for i in range(len(members)):
            a = members[i]
            for j in range(i + 1, len(members)):
                b = members[j]
                id_a, id_b = str(a.get("id")), str(b.get("id"))
                if id_a not in alive or id_b not in alive:
                    continue
                tok_a, tok_b = tokens[id_a], tokens[id_b]
                if not tok_a or not tok_b:
                    continue
                shorter, longer = sorted((len(tok_a), len(tok_b)))
                if shorter <= config.SIMILARITY_DEDUP_THRESHOLD * longer:
                    continue  # Jaccard cannot exceed the threshold
                inter = len(tok_a & tok_b)
                union = len(tok_a | tok_b)
                sim = inter / union if union else 0.0
                if sim <= config.SIMILARITY_DEDUP_THRESHOLD:
                    continue
                loser, winner = _pick_loser(a, b)
                reason = (
                    f"deduplicated: Jaccard {sim:.3f} > "
                    f"{config.SIMILARITY_DEDUP_THRESHOLD} vs question "
                    f"{winner.get('id')} (same concept '{concept}')"
                )
                discards[str(loser.get("id"))] = reason
                alive.discard(str(loser.get("id")))
                logger.warning(
                    "Stage 7 DISCARD (dedup) %s in favor of %s: %s",
                    loser.get("id"), winner.get("id"), reason,
                )
    return discards


def _evaluate(data: object) -> tuple[bool, str]:
    """Compute the pass/fail verdict from the validator's JSON response."""
    if not isinstance(data, dict):
        return False, "validation response was not a JSON object"
    discrimination = str(data.get("discrimination", "")).strip().lower()
    isolation = str(data.get("isolation", "")).strip().lower()
    drift = str(data.get("drift", "")).strip().lower()
    trap_validity = str(data.get("trap_validity", "")).strip().lower()
    reason = str(data.get("reason", "")).strip() or "no reason provided by reviewer"
    passed = (
        discrimination == "pass"
        and isolation == "pass"
        and drift in ("no_drift", "minor_drift")
        and trap_validity == "valid"
    )
    if bool(data.get("passes")) != passed:
        logger.debug(
            "reviewer 'passes' flag disagrees with the four criteria; using criteria"
        )
    return passed, reason


def _ordered_records(
    validated_by_id: dict[str, dict], questions: list[dict]
) -> list[dict]:
    """Keep the validated file in the original raw-question order."""
    order = {str(q.get("id")): i for i, q in enumerate(questions)}
    return sorted(
        validated_by_id.values(),
        key=lambda r: order.get(str(r.get("id")), 10**9),
    )


async def run(
    questions_raw_path: Path,
    validated_path: Path,
    source_questions_path: Path,
    domain: str,
) -> None:
    """Run Stage 7 for one domain."""
    require_file(
        questions_raw_path,
        f"(run stage 6 first for domain '{domain}')",
    )
    require_file(
        source_questions_path,
        f"(place {domain}.json in the project root) for the contamination check",
    )
    questions = load_json(questions_raw_path)
    questions = [q for q in questions if q.get("id") is not None]

    validated = load_json(validated_path)
    validated_by_id: dict[str, dict] = {
        str(r["id"]): r for r in validated if r.get("id") is not None
    }

    sources = load_json(source_questions_path)
    source_tokens: list[tuple[str, frozenset[str]]] = []
    for s in sources:
        tokens = tokenize(str(s.get("question", "")))
        if tokens:
            source_tokens.append((str(s.get("problem_id")), tokens))

    logger.info(
        "Stage 7 [%s]: %d question(s), %d already validated, %d source question(s) "
        "in contamination corpus",
        domain, len(questions), len(validated_by_id), len(source_tokens),
    )

    # Deterministic checks run over the FULL set so resume re-runs stay consistent.
    contaminations = _contamination_discards(questions, source_tokens)
    dedups = _dedup_discards([q for q in questions if str(q.get("id")) not in contaminations])

    pending = [q for q in questions if str(q.get("id")) not in validated_by_id]
    n_contam_pending = sum(1 for q in pending if str(q.get("id")) in contaminations)
    n_dedup_pending = sum(1 for q in pending if str(q.get("id")) in dedups)
    n_llm_pending = len(pending) - n_contam_pending - n_dedup_pending
    logger.info(
        "Stage 7 [%s]: %d pending (%d contaminated, %d deduped, %d to LLM-review)",
        domain, len(pending), n_contam_pending, n_dedup_pending, n_llm_pending,
    )

    lock = asyncio.Lock()
    counters = {"pass": 0, "fail": 0, "contaminated": 0, "deduped": 0}
    initial_validated = len(validated_by_id)
    pbar = tqdm(total=len(pending), desc=f"Stage 7 [{domain}] validation", unit="q")
    pbar.set_postfix(skip=initial_validated)

    async def finalize(q: dict, passed: bool, reason: str) -> None:
        record = {**q, "validation_passed": passed, "validation_reason": reason}
        async with lock:
            validated_by_id[str(q["id"])] = record
            save_json(validated_path, _ordered_records(validated_by_id, questions))

    async def worker(q: dict) -> None:
        qid = str(q.get("id"))
        if qid in contaminations:
            await finalize(q, False, contaminations[qid])
            counters["contaminated"] += 1
        elif qid in dedups:
            await finalize(q, False, dedups[qid])
            counters["deduped"] += 1
        else:
            question_text = str(q.get("question", ""))
            if domain == "reasoning" and str(q.get("answer", "")).strip():
                question_text += f"\n\nAnswer: {q.get('answer')}"
            prompt_text = PROMPT_VALIDATE.safe_substitute(
                failure_type=str(q.get("failure_type", "")),
                description=str(q.get("failure_description", "")),
                what_correct_understanding_looks_like=str(
                    q.get("what_correct_understanding_looks_like", "")
                ),
                question=question_text,
                trap=str(q.get("trap", "")),
                why_trap_is_tempting=str(q.get("why_trap_is_tempting", "")),
            )
            try:
                data = await call_llm(prompt_text, expect_json=True)
                passed, reason = _evaluate(data)
            except LLMError as exc:
                passed = False
                reason = f"validation LLM call failed: {exc}"
                logger.error(
                    "Stage 7 [%s] question %s: validation LLM call failed: %s",
                    domain, qid, exc,
                )
            await finalize(q, passed, reason)
            counters["pass" if passed else "fail"] += 1
            if passed:
                logger.debug("Stage 7 [%s] question %s PASSED: %s", domain, qid, reason)
            else:
                logger.info("Stage 7 [%s] question %s failed validation: %s", domain, qid, reason)
        pbar.update(1)
        pbar.set_postfix(skip=initial_validated, **counters)

    if pending:
        await asyncio.gather(*(worker(q) for q in pending))
    pbar.close()
    if not validated_path.exists():
        save_json(validated_path, [])  # keep downstream stages runnable on empty input
    logger.info(
        "Stage 7 [%s] complete: %d validated record(s); totals: %d passed, %d failed "
        "(%d contaminated, %d deduped in this run)",
        domain, len(validated_by_id),
        sum(1 for r in validated_by_id.values() if r.get("validation_passed")),
        sum(1 for r in validated_by_id.values() if not r.get("validation_passed")),
        n_contam_pending, n_dedup_pending,
    )
