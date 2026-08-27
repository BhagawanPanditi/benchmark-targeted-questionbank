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
from typing import Any

from prompts.stage3_taxonomy import PROMPT_NORMALIZE
from utils.io import load_json, load_json_obj, require_file, save_json
from utils.llm import LLMError, call_llm, set_concurrency

logger = logging.getLogger(__name__)


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


async def run(
    raw_concepts_path: Path,
    taxonomy_path: Path,
    domain: str,
    concurrency: int | None = None,
) -> None:
    """Run Stage 3 for one domain."""
    if concurrency is not None:
        set_concurrency(concurrency)
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
