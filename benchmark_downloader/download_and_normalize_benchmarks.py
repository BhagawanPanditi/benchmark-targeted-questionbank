#!/usr/bin/env python3
"""
download_and_normalize_benchmarks.py

Download coding + reasoning benchmarks from Hugging Face and normalize them into:
    coding.json
    reasoning.json
    download_manifest.json
    normalization_errors.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset


DEFAULT_OUTPUT_DIR = Path(
    "~/bgen_projects/concept_based_data_gen/benchmarks"
).expanduser()

JSON_INDENT = 2


# ============================================================
# Canonical Schema
# ============================================================

@dataclass
class Record:
    benchmark: str
    sub_benchmark: Optional[str]
    problem_id: str
    question: str
    answer: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "sub_benchmark": self.sub_benchmark,
            "problem_id": self.problem_id,
            "question": self.question,
            "answer": self.answer,
        }


@dataclass
class DatasetSpec:
    category: str
    benchmark: str
    hf_id: str
    config: Optional[str] = None
    split: Optional[str] = None
    sub_benchmark: Optional[str] = None
    notes: str = ""
    all_configs: bool = False


# ============================================================
# Utility Helpers
# ============================================================

def clean(value: Any) -> Optional[str]:
    """Convert arbitrary HF values to a string. None/empty returns None."""
    if value is None:
        return None

    if isinstance(value, str):
        v = value.strip()
        return v if v else None

    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    return str(value)


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            val = row[key]
            if isinstance(val, str) and not val.strip():
                continue
            return val
    return None


def first_text(row: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    return clean(first_present(row, keys))


def flatten_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def try_len(ds: Any) -> int:
    try:
        return len(ds)
    except Exception:
        return 0


def choose_split(dataset_obj: Any, preferred: Optional[str] = None) -> Dataset:
    if isinstance(dataset_obj, DatasetDict):
        if preferred and preferred in dataset_obj:
            return dataset_obj[preferred]
        for split in ["test", "train", "validation", "dev", "f2f"]:
            if split in dataset_obj:
                return dataset_obj[split]
        first_key = next(iter(dataset_obj.keys()))
        return dataset_obj[first_key]
    return dataset_obj


def load_one_dataset(
    hf_id: str,
    config: Optional[str],
    split: Optional[str],
    token: Optional[str] = None,
) -> Dataset:
    errors = []

    # 1. Try explicit config + split
    if config and split:
        try:
            return load_dataset(hf_id, config, split=split, token=token, trust_remote_code=True)
        except Exception as e:
            errors.append(f"config={config}, split={split}: {e}")

    # 2. Try explicit config, infer split
    if config:
        try:
            obj = load_dataset(hf_id, config, token=token, trust_remote_code=True)
            return choose_split(obj, split)
        except Exception as e:
            errors.append(f"config={config}: {e}")

    # 3. Try no config, explicit split
    if split:
        try:
            return load_dataset(hf_id, split=split, token=token, trust_remote_code=True)
        except Exception as e:
            errors.append(f"split={split}: {e}")

    # 4. Default fallback
    try:
        obj = load_dataset(hf_id, token=token, trust_remote_code=True)
        return choose_split(obj, split)
    except Exception as e:
        errors.append(f"default: {e}")

    raise RuntimeError(
        f"Unable to load {hf_id}"
        + (f" / {config}" if config else "")
        + "\n"
        + "\n".join(errors)
    )


def discover_configs(hf_id: str, token: Optional[str] = None) -> list[str]:
    try:
        configs = get_dataset_config_names(hf_id, token=token, trust_remote_code=True)
        return list(configs)
    except Exception:
        return []


def iterate_dataset(ds: Dataset) -> Iterable[dict[str, Any]]:
    for row in ds:
        yield flatten_row(row)


# ============================================================
# Generic Normalizers
# ============================================================

def generic_question(row: dict[str, Any]) -> Optional[str]:
    # Support 'turns' for LiveBench and conversational benchmarks
    if "turns" in row and isinstance(row["turns"], list) and row["turns"]:
        first_turn = row["turns"][0]
        if isinstance(first_turn, dict):
            return clean(first_turn.get("content") or first_turn.get("text"))
        return clean(first_turn)

    return first_text(
        row,
        [
            "question",
            "problem",
            "prompt",
            "text",
            "query",
            "input",
            "description",
            "instruction",
            "context",
            "entity_question",
            "complete_prompt",
            "instruct_prompt",
            "skeleton",
        ],
    )


def generic_answer(row: dict[str, Any]) -> Optional[str]:
    return first_text(
        row,
        [
            "ground_truth",
            "answer",
            "answers",
            "gold",
            "gold_answer",
            "reference_answer",
            "canonical_solution",
            "code",
            "solution",
            "solution_code",
            "output",
            "target",
            "Short Answer",
            "canonical",
        ],
    )


def generic_id(row: dict[str, Any], index: int, benchmark: str, sub_benchmark: Optional[str] = None) -> str:
    value = first_present(
        row,
        [
            "problem_id",
            "Problem ID",
            "task_id",
            "question_id",
            "problem_idx",
            "id",
            "idx",
            "name",
        ],
    )
    v_clean = clean(value)
    if v_clean:
        return v_clean
    sub_prefix = f"{sub_benchmark}/" if sub_benchmark else ""
    return f"{benchmark}/{sub_prefix}{index}"


def generic_normalizer(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    return Record(
        benchmark=spec.benchmark,
        sub_benchmark=spec.sub_benchmark,
        problem_id=generic_id(row, index, spec.benchmark, spec.sub_benchmark),
        question=generic_question(row) or "",
        answer=generic_answer(row),
    )


# ============================================================
# Benchmark-Specific Normalizers
# ============================================================

def normalize_humaneval(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    return Record(
        benchmark="HumanEval",
        sub_benchmark=spec.sub_benchmark,
        problem_id=clean(row.get("task_id")) or f"HumanEval/{index}",
        question=clean(row.get("prompt")) or "",
        answer=clean(row.get("canonical_solution")),
    )


def normalize_mbpp(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    pid = clean(row.get("task_id")) or f"MBPP/{index}"
    question = clean(row.get("text")) or ""
    # Append test signature if available to clarify the target function signature
    test_list = row.get("test_list")
    if isinstance(test_list, list) and test_list:
        question += f"\n\nTest example:\n{test_list[0]}"

    return Record(
        benchmark="MBPP",
        sub_benchmark=spec.sub_benchmark,
        problem_id=pid,
        question=question,
        answer=clean(row.get("code")),
    )


def normalize_cruxeval(row: dict[str, Any], index: int, spec: DatasetSpec) -> list[Record]:
    pid = clean(row.get("id")) or f"CRUXEval/{index}"
    code = clean(row.get("code")) or ""
    inp = clean(row.get("input")) or ""
    out = clean(row.get("output"))

    q = (
        "Given the following Python function:\n\n"
        f"{code}\n\n"
        "Input:\n"
        f"{inp}\n\n"
        "Predict the output."
    )
    return [
        Record(
            benchmark="CruxEval",
            sub_benchmark=spec.sub_benchmark or "Output Prediction",
            problem_id=pid,
            question=q,
            answer=out,
        )
    ]


def normalize_classeval(row: dict[str, Any], index: int, spec: DatasetSpec) -> list[Record]:
    pid = clean(row.get("task_id")) or f"ClassEval/{index}"
    class_description = clean(row.get("class_description")) or ""
    skeleton = clean(row.get("skeleton")) or ""
    solution = clean(row.get("solution_code"))

    parts = []
    if class_description:
        parts.append(f"Class description:\n{class_description}")
    if skeleton:
        parts.append(f"Class skeleton:\n{skeleton}")

    return [
        Record(
            benchmark="ClassEval",
            sub_benchmark=spec.sub_benchmark,
            problem_id=pid,
            question="\n\n".join(parts),
            answer=solution,
        )
    ]


def normalize_livecodebench(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    pid = (
        clean(row.get("question_id"))
        or clean(row.get("task_id"))
        or clean(row.get("id"))
        or f"LiveCodeBench/{index}"
    )
    question = (
        generic_question(row)
        or clean(row.get("question_content"))
        or ""
    )
    answer = (
        generic_answer(row)
        or clean(row.get("starter_code"))
    )
    return Record(
        benchmark="LiveCodeBench",
        sub_benchmark=spec.sub_benchmark,
        problem_id=pid,
        question=question,
        answer=answer,
    )


def normalize_multipl_e(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    language = clean(row.get("language"))
    name = clean(row.get("name")) or f"MultiPL-E/{index}"
    sub = spec.sub_benchmark

    if language and sub and language not in sub:
        sub = f"{sub}/{language}"
    elif language and not sub:
        sub = language

    return Record(
        benchmark="MultiPL-E",
        sub_benchmark=sub,
        problem_id=name,
        question=clean(row.get("prompt")) or "",
        answer=clean(row.get("canonical_solution")) or clean(row.get("original")) or clean(row.get("tests")),
    )


def normalize_gpqa(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    pid = clean(row.get("Record ID")) or clean(row.get("id")) or f"GPQA/{index}"
    question = clean(row.get("Question")) or ""

    choices = []
    for key in ["Choice A", "Choice B", "Choice C", "Choice D"]:
        val = clean(row.get(key))
        if val:
            choices.append(f"{key}: {val}")

    correct = clean(row.get("Correct Answer")) or clean(row.get("answer"))
    if not choices and correct:
        incorrect_1 = clean(row.get("Incorrect Answer 1"))
        incorrect_2 = clean(row.get("Incorrect Answer 2"))
        incorrect_3 = clean(row.get("Incorrect Answer 3"))
        options = [opt for opt in [correct, incorrect_1, incorrect_2, incorrect_3] if opt]
        if options:
            for i, opt in enumerate(options):
                tag = chr(ord('A') + i)
                choices.append(f"({tag}) {opt}")

    if choices:
        question += "\n\nChoices:\n" + "\n".join(choices)

    return Record(
        benchmark="GPQA",
        sub_benchmark=spec.sub_benchmark,
        problem_id=pid,
        question=question,
        answer=correct,
    )


def normalize_autologi(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    pid = clean(row.get("id")) or f"AutoLogi/{index}"
    context = clean(row.get("context"))
    q_text = clean(row.get("question"))

    parts = []
    if context:
        parts.append(f"Context:\n{context}")
    if q_text:
        parts.append(f"Question:\n{q_text}")

    question = "\n\n".join(parts) if parts else ""
    answer = clean(row.get("answer")) or clean(row.get("solution")) or clean(row.get("label"))

    return Record(
        benchmark="AutoLogi",
        sub_benchmark=spec.sub_benchmark,
        problem_id=pid,
        question=question,
        answer=answer,
    )


def normalize_minif2f(row: dict[str, Any], index: int, spec: DatasetSpec) -> Record:
    pid = clean(row.get("name")) or clean(row.get("id")) or f"MiniF2F/{index}"
    informal = clean(row.get("informal_prefix"))
    formal_statement = clean(row.get("formal_statement")) or clean(row.get("statement")) or ""

    question = f"{informal}\n\nFormal statement:\n{formal_statement}" if informal else formal_statement
    answer = clean(row.get("proof")) or clean(row.get("informal_proof")) or formal_statement

    return Record(
        benchmark="MiniF2F",
        sub_benchmark=spec.sub_benchmark,
        problem_id=pid,
        question=question,
        answer=answer,
    )


NORMALIZERS: dict[str, Callable[..., Any]] = {
    "HumanEval": normalize_humaneval,
    "MBPP": normalize_mbpp,
    "CruxEval": normalize_cruxeval,
    "ClassEval": normalize_classeval,
    "LiveCodeBench": normalize_livecodebench,
    "MultiPL-E": normalize_multipl_e,
    "GPQA": normalize_gpqa,
    "AutoLogi": normalize_autologi,
    "MiniF2F": normalize_minif2f,
}


def normalize_row(row: dict[str, Any], index: int, spec: DatasetSpec) -> list[Record]:
    fn = NORMALIZERS.get(spec.benchmark, generic_normalizer)
    result = fn(row, index, spec)
    return result if isinstance(result, list) else [result]


# ============================================================
# Benchmark Registry
# ============================================================

CODING = [
    DatasetSpec("coding", "HumanEval", "openai/openai_humaneval", split="test"),
    DatasetSpec("coding", "MBPP", "Muennighoff/mbpp", config="full", split="test"),
    DatasetSpec("coding", "EvalPlus", "evalplus/humanevalplus", sub_benchmark="HumanEval+", split="test"),
    DatasetSpec("coding", "EvalPlus", "evalplus/mbppplus", sub_benchmark="MBPP+", split="test"),
    DatasetSpec("coding", "EvalPerf", "evalplus/evalperf", split="test"),
    DatasetSpec("coding", "CruxEval", "cruxeval-org/cruxeval", split="test"),
    DatasetSpec("coding", "LiveCodeBench", "livecodebench/code_generation_lite", split="test", sub_benchmark="Code Generation"),
    DatasetSpec("coding", "ClassEval", "FudanSELab/ClassEval", split="test"),
    DatasetSpec("coding", "InfiBench", "llylly001/InfiBench", split=None),
    DatasetSpec("coding", "EvoEval", "evoeval/EvoEval_difficult", split=None, sub_benchmark="Difficult"),
    DatasetSpec("coding", "EvoEval", "evoeval/EvoEval_creative", split=None, sub_benchmark="Creative"),
    DatasetSpec("coding", "EvoEval", "evoeval/EvoEval_subtle", split=None, sub_benchmark="Subtle"),
    DatasetSpec("coding", "EvoEval", "evoeval/EvoEval_tool_use", split=None, sub_benchmark="Tool Use"),
    DatasetSpec("coding", "EvoEval", "evoeval/EvoEval_combine", split=None, sub_benchmark="Combine"),
    DatasetSpec("coding", "SciCode", "budecosystem/scicode", config="scicode_gen", split=None, sub_benchmark="SciCode"),
    DatasetSpec("coding", "MultiPL-E", "nuprl/MultiPL-E", all_configs=True),
    DatasetSpec("coding", "Multi-LCB", "Multi-LCB/Multi-LCB", split=None),
    DatasetSpec("coding", "OJBench", "He-Ren/OJBench_testdata", split=None, sub_benchmark="Full prompts"),
    DatasetSpec("coding", "CodeElo", "Qwen/CodeElo", split="test"),
    DatasetSpec("coding", "DS-1000", "embedding-benchmark/DS1000", config="queries", split=None, sub_benchmark="Queries"),
    DatasetSpec("coding", "LBPP", "CohereLabs/lbpp", split=None, sub_benchmark="LBPP / LBPPv2"),
    DatasetSpec("coding", "DeepSeek LeetCode", "Qwen/CodeElo", split="test", sub_benchmark="PLACEHOLDER_DO_NOT_USE"),
]

REASONING = [
    DatasetSpec("reasoning", "BBEH", "BBEH/bbeh", split="train"),
    DatasetSpec("reasoning", "AGIEval", "junnyu/agieval", config="aqua_rat", split="test", sub_benchmark="aqua_rat"),
    DatasetSpec("reasoning", "AGIEval", "hails/agieval-jec-qa-ca", split="test", sub_benchmark="JEC-QA-CA"),
    DatasetSpec("reasoning", "HLE", "cais/hle", split="test"),
    DatasetSpec("reasoning", "HLE-Verified", "skylenage-ai/HLE-Verified", split=None),
    DatasetSpec("reasoning", "LiveBench", "livebench/reasoning", split="test", sub_benchmark="reasoning"),
    DatasetSpec("reasoning", "LiveBench", "livebench/math", split="test", sub_benchmark="math"),
    DatasetSpec("reasoning", "GPQA", "Idavidrein/gpqa", config="gpqa_diamond", split="train", sub_benchmark="Diamond"),
    DatasetSpec("reasoning", "SuperGPQA", "skylion007/SuperGPQA", split=None),
    DatasetSpec("reasoning", "PopQA", "skylion007/PopQA-exact", split=None),
    DatasetSpec("reasoning", "ZebraLogic", "reasoning-mazes/ZebraLogic", config="grid_mode", split=None, sub_benchmark="Grid"),
    DatasetSpec("reasoning", "ZebraLogic", "reasoning-mazes/ZebraLogic", config="mc_mode", split=None, sub_benchmark="Multiple Choice"),
    DatasetSpec("reasoning", "AutoLogi", "GlobalBetween/autologi-qa", split="train"),
    DatasetSpec("reasoning", "MiniF2F", "m-a-p/miniF2F", config="f2f", split="f2f"),
    DatasetSpec("reasoning", "HMMT", "MathArena/hmmt_feb_2025", split="train", sub_benchmark="February 2025"),
    DatasetSpec("reasoning", "HMMT", "MathArena/hmmt_nov_2025", split="train", sub_benchmark="November 2025"),
    DatasetSpec("reasoning", "IMO-Bench", "Hwilner/imo-answerbench", split="train", sub_benchmark="AnswerBench"),
    DatasetSpec("reasoning", "IMO-Bench", "Hwilner/imo-proofbench", split="train", sub_benchmark="ProofBench"),
    DatasetSpec("reasoning", "IMO-Bench", "Hwilner/imo-gradingbench", split="train", sub_benchmark="GradingBench"),
    DatasetSpec("reasoning", "AIME", "math-ai/aime26", split="test", sub_benchmark="AIME 2026"),
]


# ============================================================
# Processing & Deduplication
# ============================================================

def should_skip(spec: DatasetSpec) -> bool:
    return bool(spec.sub_benchmark and "PLACEHOLDER_DO_NOT_USE" in spec.sub_benchmark)


def iter_spec_datasets(spec: DatasetSpec, token: Optional[str] = None) -> Iterable[tuple[str, Optional[str], Dataset]]:
    if should_skip(spec):
        return

    if spec.all_configs:
        configs = discover_configs(spec.hf_id, token=token)
        if not configs:
            raise RuntimeError(f"Could not discover configs for {spec.hf_id}")

        for cfg in configs:
            try:
                ds = load_one_dataset(spec.hf_id, cfg, spec.split, token=token)
                sub = spec.sub_benchmark or cfg
                yield cfg, sub, ds
            except Exception as e:
                print(f"      WARN config failed: {spec.hf_id}/{cfg}: {e}", file=sys.stderr)
        return

    ds = load_one_dataset(spec.hf_id, spec.config, spec.split, token=token)
    yield spec.config, spec.sub_benchmark, ds


def dedupe_records(records: list[Record]) -> list[Record]:
    seen = set()
    out = []
    for record in records:
        key = (record.benchmark, record.sub_benchmark, record.problem_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def process_spec(spec: DatasetSpec, output_records: list[Record], token: Optional[str] = None) -> dict[str, Any]:
    print(
        f"\n  [{spec.category.upper()}] "
        f"{spec.benchmark}"
        f"{' / ' + str(spec.sub_benchmark) if spec.sub_benchmark else ''}"
        f" <- {spec.hf_id}"
    )

    if should_skip(spec):
        print("      SKIP: unresolved placeholder")
        return {
            "status": "SKIP",
            "benchmark": spec.benchmark,
            "hf_id": spec.hf_id,
            "reason": spec.notes or "Placeholder",
        }

    total = 0
    normalized = 0
    attempted_configs = 0

    try:
        for effective_config, effective_sub_benchmark, ds in iter_spec_datasets(spec, token=token):
            attempted_configs += 1
            print(
                f"      config={effective_config or '-'} "
                f"split={spec.split or 'auto'} "
                f"rows={try_len(ds)}"
            )

            for i, row in enumerate(iterate_dataset(ds)):
                local_spec = DatasetSpec(
                    category=spec.category,
                    benchmark=spec.benchmark,
                    hf_id=spec.hf_id,
                    config=effective_config,
                    split=spec.split,
                    sub_benchmark=effective_sub_benchmark,
                    notes=spec.notes,
                )

                records = normalize_row(row, i, local_spec)
                for record in records:
                    if not record.question.strip():
                        continue
                    output_records.append(record)
                    normalized += 1
                total += 1

        if attempted_configs > 0 and normalized == 0 and total == 0:
            return {
                "status": "ERROR",
                "benchmark": spec.benchmark,
                "hf_id": spec.hf_id,
                "error": "No valid records extracted from any config",
            }

        print(f"      OK: {normalized} normalized records")
        return {
            "status": "OK",
            "benchmark": spec.benchmark,
            "hf_id": spec.hf_id,
            "config": spec.config,
            "split": spec.split,
            "rows": total,
            "normalized": normalized,
        }

    except Exception as e:
        print(f"      ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return {
            "status": "ERROR",
            "benchmark": spec.benchmark,
            "hf_id": spec.hf_id,
            "config": spec.config,
            "split": spec.split,
            "error": str(e),
        }


# ============================================================
# Validation & Persistence
# ============================================================

def validate_records(records: list[Record]) -> tuple[int, list[str]]:
    bad = []
    good = 0
    required = {"benchmark", "sub_benchmark", "problem_id", "question", "answer"}

    for idx, rec in enumerate(records):
        data = rec.to_dict()
        errs = []
        if set(data.keys()) != required:
            errs.append(f"invalid schema: {sorted(data.keys())}")
        if not data["benchmark"]:
            errs.append("missing benchmark")
        if not data["problem_id"]:
            errs.append("missing problem_id")
        if not isinstance(data["question"], str) or not data["question"].strip():
            errs.append("empty or non-string question")
        if data["answer"] is not None and not isinstance(data["answer"], str):
            errs.append("answer must be string or null")

        if errs:
            bad.append(f"record {idx} ({data.get('problem_id', 'unknown')}): " + "; ".join(errs))
        else:
            good += 1

    return good, bad


def save_json(path: Path, records: list[Record]) -> None:
    payload = [r.to_dict() for r in records]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=JSON_INDENT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize benchmarks.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmarks", nargs="*", help="Filter specific benchmark names to run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coding_specs = CODING
    reasoning_specs = REASONING
    if args.benchmarks:
        filter_set = set(b.lower() for b in args.benchmarks)
        coding_specs = [s for s in CODING if s.benchmark.lower() in filter_set]
        reasoning_specs = [s for s in REASONING if s.benchmark.lower() in filter_set]

    print("=" * 90)
    print("BENCHMARK DOWNLOADER + NORMALIZER")
    print(f"Output directory:  {output_dir}")
    print(f"Coding targets:    {len(coding_specs)}")
    print(f"Reasoning targets: {len(reasoning_specs)}")
    print("=" * 90)

    coding_records: list[Record] = []
    reasoning_records: list[Record] = []
    manifest: dict[str, list[Any]] = {"coding": [], "reasoning": []}

    print("\n" + "=" * 90 + "\nCODING\n" + "=" * 90)
    for spec in coding_specs:
        manifest["coding"].append(process_spec(spec, coding_records, token=token))

    print("\n" + "=" * 90 + "\nREASONING\n" + "=" * 90)
    for spec in reasoning_specs:
        manifest["reasoning"].append(process_spec(spec, reasoning_records, token=token))

    coding_records = dedupe_records(coding_records)
    reasoning_records = dedupe_records(reasoning_records)

    c_good, c_bad = validate_records(coding_records)
    r_good, r_bad = validate_records(reasoning_records)

    save_json(output_dir / "coding.json", coding_records)
    save_json(output_dir / "reasoning.json", reasoning_records)

    with (output_dir / "download_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with (output_dir / "normalization_errors.json").open("w", encoding="utf-8") as f:
        json.dump({"coding": c_bad, "reasoning": r_bad}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Coding records:    {len(coding_records):,} (valid: {c_good:,}, errors: {len(c_bad):,})")
    print(f"Reasoning records: {len(reasoning_records):,} (valid: {r_good:,}, errors: {len(r_bad):,})")
    print(f"Output files saved to: {output_dir}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
