"""Stage 3 — Taxonomy normalization (done ONCE per domain).

Step 3.1: collect all unique raw concept strings across the whole domain.
Step 3.2: call the LLM (Prompt S3-NORMALIZE) to produce the canonical taxonomy,
          merge_map, removed list, and category summary.
Step 3.3: save the taxonomy file.
Step 3.4: update every record with "normalized_concepts" (canonical forms via
          the merge_map), saving after each record.

If the taxonomy file already exists, Steps 3.1-3.3 are skipped and the existing
merge_map is applied to any records still missing "normalized_concepts".
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any

from utils.io import load_json, load_json_obj, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

PROMPT_NORMALIZE = Template(r"""You are a taxonomy designer for a ${domain} education system.

Below are ${n} programming concept tags independently extracted from ${domain} problems.
There are duplicates, synonyms, and inconsistent naming. Produce a CLEAN, CANONICAL
taxonomy following these steps exactly.

STEP 1 — Merge synonyms (preserve the most standard, descriptive name):
Examples of correct merges:
  algorithms.pointers.two-pointer + patterns.array.two-pointers
    → algorithms.technique.two-pointer
  data-structures.hash.hashmap + mapping.dict.dictionary
    → data-structures.mapping.dictionary
  math.arithmetic.modular + algorithms.math.modular-arithmetic
    → math.modular-arithmetic.modular-inverse
  dp.dynamic-programming.knapsack + algorithms.dp.0-1-knapsack
    → dp.knapsack.zero-one

STEP 2 — Remove noise. Remove a concept if it is:
  - Too vague to be teachable: "algorithms.general.problem-solving"
  - Specific to one problem instance: "algorithms.array.find-max-of-three-numbers"
  - Pure syntax rather than a concept: "basics.syntax.for-loop", "basics.control.if-statement"
  - Non-domain concern: "practice.interview.preparation"

STEP 3 — Enforce exactly 3 levels for every surviving concept:
  2-level tag → infer and add the missing subcategory level:
    algorithms.recursion → algorithms.recursion.recursive-decomposition
    data-structures.stack → data-structures.stack.push-pop-operations
  4-level tag → flatten by merging the two most semantically related levels:
    algorithms.graph.traversal.bfs → graph.traversal.bfs
    functionality.data.processing.filtering → functionality.processing.filtering

STEP 4 — Standardize naming conventions:
  - All lowercase
  - Hyphens between words within a level (two-pointer not two_pointer or twopointer)
  - Dots between levels (algorithms.technique.two-pointer)
  - No trailing hyphens or dots
  - Consistent vocabulary — pick one term and use it everywhere:
      "dictionary" not "dict" / "hashmap" / "hash-table"
      "two-pointer" not "two-pointers" / "2-pointer"
      "memoization" not "memo" / "top-down-dp"
  - For coding domain, allowed top-level categories:
      algorithms, data-structures, functionality, analytics, graph, dp, string, math
  - For reasoning domain, allowed top-level categories:
      algebra, combinatorics, number-theory, geometry, logic, probability, calculus,
      proof-technique

Return ONLY valid JSON:
{
  "taxonomy": ["canonical.concept.one", ...],
  "merge_map": {
    "raw_concept_string": "canonical.concept.string"
  },
  "removed": {
    "removed_concept": "reason for removal"
  },
  "category_summary": {
    "top_level_category": count
  }
}

Raw tags (${n} total):
${tags}""")


def _normalize_list(raw_concepts: list[str], merge_map: dict[str, str]) -> list[str]:
    """Apply the merge map, dropping blanks and de-duplicating (order preserved)."""
    out: list[str] = []
    seen: set[str] = set()
    for concept in raw_concepts or []:
        concept = str(concept).strip()
        if not concept:
            continue
        canonical = merge_map.get(concept, concept)
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _write_taxonomy(
    path: Path,
    domain: str,
    taxonomy: list[str],
    merge_map: dict[str, str],
    removed: dict[str, str],
    category_summary: dict[str, Any],
) -> None:
    doc = {
        "domain": domain,
        "taxonomy": taxonomy,
        "merge_map": merge_map,
        "removed": removed,
        "category_summary": category_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(path, doc)


async def run(raw_concepts_path: Path, taxonomy_path: Path, domain: str) -> None:
    """Run Stage 3 for one domain."""
    require_file(
        raw_concepts_path,
        f"(run stage 2 first for domain '{domain}')",
    )
    records = load_json(raw_concepts_path)

    # --- Step 3.1: collect unique raw concepts (order-preserving) ----------
    raw_unique: list[str] = []
    seen: set[str] = set()
    for record in records:
        for concept in record.get("raw_concepts", []):
            concept = str(concept).strip()
            if concept and concept not in seen:
                seen.add(concept)
                raw_unique.append(concept)

    # --- If a taxonomy already exists, only apply the existing merge map ----
    existing = load_json_obj(taxonomy_path)
    if isinstance(existing, dict) and existing.get("merge_map") is not None:
        merge_map = {str(k): str(v) for k, v in existing["merge_map"].items()}
        logger.info(
            "Stage 3 [%s]: reusing existing taxonomy (%d merge-map entries)",
            domain, len(merge_map),
        )
        updated = 0
        for record in records:
            if "normalized_concepts" in record:
                continue
            record["normalized_concepts"] = _normalize_list(
                record.get("raw_concepts", []), merge_map
            )
            save_json(raw_concepts_path, records)
            updated += 1
        logger.info(
            "Stage 3 [%s]: applied merge map to %d record(s); %d already normalized",
            domain, updated, len(records) - updated,
        )
        return

    # --- No taxonomy yet -----------------------------------------------------
    if not raw_unique:
        logger.warning(
            "Stage 3 [%s]: no raw concepts found; writing empty taxonomy", domain
        )
        _write_taxonomy(taxonomy_path, domain, [], {}, {}, {})
        for record in records:
            record.setdefault("normalized_concepts", [])
        if records:
            save_json(raw_concepts_path, records)
        return

    # --- Steps 3.2 + 3.3: LLM normalization call, then save ------------------
    prompt_text = PROMPT_NORMALIZE.safe_substitute(
        domain=domain, n=len(raw_unique), tags="\n".join(raw_unique)
    )
    try:
        data = await call_llm(prompt_text, expect_json=True)
    except LLMError as exc:
        logger.error("Stage 3 [%s]: taxonomy normalization failed: %s", domain, exc)
        raise SystemExit(f"stage 3 [{domain}]: taxonomy normalization failed: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("taxonomy"), list):
        logger.error(
            "Stage 3 [%s]: taxonomy response missing a 'taxonomy' list: %r",
            domain, str(data)[:200],
        )
        raise SystemExit(f"stage 3 [{domain}]: invalid taxonomy response from LLM")

    taxonomy: list[str] = []
    seen_tax: set[str] = set()
    for item in data["taxonomy"]:
        item = str(item).strip()
        if item and item not in seen_tax:
            seen_tax.add(item)
            taxonomy.append(item)
    merge_map = {str(k): str(v) for k, v in (data.get("merge_map") or {}).items()}
    removed = {str(k): str(v) for k, v in (data.get("removed") or {}).items()}
    category_summary = {str(k): v for k, v in (data.get("category_summary") or {}).items()}

    _write_taxonomy(taxonomy_path, domain, taxonomy, merge_map, removed, category_summary)
    logger.info(
        "Stage 3 [%s]: taxonomy built: %d canonical concept(s), %d merge entr(ies), "
        "%d removed",
        domain, len(taxonomy), len(merge_map), len(removed),
    )

    # --- Step 3.4: normalize every record (save after each) ------------------
    for record in records:
        if "normalized_concepts" in record:
            continue
        record["normalized_concepts"] = _normalize_list(
            record.get("raw_concepts", []), merge_map
        )
        save_json(raw_concepts_path, records)
    logger.info("Stage 3 [%s]: all %d record(s) normalized", domain, len(records))
