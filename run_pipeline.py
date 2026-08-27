#!/usr/bin/env python3
"""Entry point for the prerequisite question bank pipeline.

Usage:
    python run_pipeline.py

Input files are read from the project root: ``coding.json`` and
``reasoning.json`` (see README.md for the record format).

Optional flags:
    --stages 1,2,3      run only specific stages (comma-separated, default: all)
    --domain coding     run for one domain only (default: both)
    --resume            accepted for compatibility — resuming is ALWAYS on:
                        completed records (matched by problem_id / id) are skipped
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import config

from stages import (
    stage1_reasoning,
    stage2_raw_concepts,
    stage3_taxonomy,
    stage4_failure_modes,
    stage5_concept_graph,
    stage6_question_gen,
    stage7_validation,
    stage8_output,
    stage9_readme,
)
from utils.io import load_json

logger = logging.getLogger("pipeline")

ALL_STAGES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def setup_logging() -> None:
    """Console at INFO, pipeline.log at DEBUG."""
    root = logging.getLogger()
    if root.handlers:  # already configured (re-entrant call)
        return
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )

    file_handler = logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    root.addHandler(console)
    root.addHandler(file_handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate targeted diagnostic prerequisite questions from the "
        "benchmark problem files coding.json and reasoning.json in the project root.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(str(n) for n in ALL_STAGES),
        help="Comma-separated stage numbers to run (default: 1,2,3,4,5,6,7,8,9)",
    )
    parser.add_argument(
        "--domain",
        choices=["coding", "reasoning", "both"],
        default="both",
        help="Run for one domain only (default: both)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Accepted for compatibility. Resuming is always on: completed "
        "records are always skipped and stages are safe to re-run.",
    )
    return parser.parse_args(argv)


def resolve_stages(raw: str) -> list[int]:
    stages: set[int] = set()
    for chunk in raw.replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            number = int(chunk)
        except ValueError as exc:
            raise SystemExit(f"error: invalid stage number in --stages: {chunk!r}") from exc
        if number not in ALL_STAGES:
            raise SystemExit(f"error: unknown stage {number} (valid: 1-9)")
        stages.add(number)
    return sorted(stages)


def validate_inputs(domains: list[str]) -> dict[str, Path]:
    """Resolve and sanity-check the fixed input files in the project root."""
    paths: dict[str, Path] = {}
    for domain in domains:
        path = config.input_file(domain)
        if not path.exists():
            raise SystemExit(
                f"error: input file not found: {path} "
                f"(place {domain}.json in the project root)"
            )
        records = load_json(path)
        for record in records:
            for field in ("problem_id", "question", "answer"):
                if record.get(field) is None:
                    logger.warning(
                        "input %s: record %r is missing field '%s'",
                        path, record.get("problem_id"), field,
                    )
        logger.info("input %s: %d record(s) loaded", path, len(records))
        paths[domain] = path
    return paths


async def run_stages(
    stages: list[int], domains: list[str], paths: dict[str, Path]
) -> None:
    for stage_no in stages:
        if stage_no == 9:
            continue  # runs once, after every domain finishes stage 8
        for domain in domains:
            if stage_no == 1:
                await stage1_reasoning.run(paths[domain], config.reasoning_file(domain), domain)
            elif stage_no == 2:
                await stage2_raw_concepts.run(
                    config.reasoning_file(domain), config.raw_concepts_file(domain), domain
                )
            elif stage_no == 3:
                await stage3_taxonomy.run(
                    config.raw_concepts_file(domain), config.taxonomy_file(domain), domain
                )
            elif stage_no == 4:
                await stage4_failure_modes.run(
                    config.reasoning_file(domain),
                    config.taxonomy_file(domain),
                    config.failure_modes_file(domain),
                    domain,
                )
            elif stage_no == 5:
                await stage5_concept_graph.run(
                    config.taxonomy_file(domain), config.concept_graph_file(domain), domain
                )
            elif stage_no == 6:
                await stage6_question_gen.run(
                    config.failure_modes_file(domain),
                    config.concept_graph_file(domain),
                    config.questions_raw_file(domain),
                    domain,
                )
            elif stage_no == 7:
                await stage7_validation.run(
                    config.questions_raw_file(domain),
                    config.questions_validated_file(domain),
                    paths[domain],
                    domain,
                )
            elif stage_no == 8:
                await stage8_output.run(
                    config.questions_validated_file(domain),
                    config.raw_concepts_file(domain),
                    config.concept_graph_file(domain),
                    config.final_output_file(domain),
                    domain,
                )
    if 9 in stages:
        await stage9_readme.run({d: paths.get(d) for d in config.DOMAINS})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()

    stages = resolve_stages(args.stages)
    domains = (
        [args.domain] if args.domain in ("coding", "reasoning") else list(config.DOMAINS)
    )
    paths = validate_inputs(domains)

    logger.info(
        "Pipeline start: domains=%s stages=%s base_url=%s model=%s",
        domains, stages, config.LLM_BASE_URL, config.LLM_MODEL,
    )
    try:
        asyncio.run(run_stages(stages, domains, paths))
    except SystemExit as exc:
        logger.error("pipeline aborted: %s", exc)
        raise
    except KeyboardInterrupt:
        logger.warning("interrupted — safe to re-run: completed records will be skipped")
        sys.exit(130)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
