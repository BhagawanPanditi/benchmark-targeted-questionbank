# Benchmark Targeted Question Bank: Downloader & Normalizer

A high-performance pipeline to download, parse, validate, and normalize standard **Coding** and **Reasoning** benchmarks from Hugging Face into unified, canonical JSON datasets.

---

## Table of Contents
- [Overview](#overview)
- [Canonical Data Schema](#canonical-data-schema)
- [Supported Benchmarks](#supported-benchmarks)
- [Installation & Setup](#installation--setup)
- [Usage & Execution](#usage--execution)
- [Output Artifacts](#output-artifacts)
- [Comprehensive Caveats & Technical Nuances](#comprehensive-caveats--technical-nuances)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview

Modern evaluation datasets on Hugging Face Hub come in wildly varying schemas: different column naming conventions (`prompt`, `question`, `turns`, `text`), custom class structures (`skeleton`, `methods_info`), varying answer formats (`canonical_solution`, `ground_truth`, `correct_answer`, `options`), and split structures.

`download_and_normalize_benchmarks.py` maps all benchmarks into a single clean format across two primary categories:
1. **Coding Benchmarks** (`coding.json`)
2. **Reasoning Benchmarks** (`reasoning.json`)

---

## Canonical Data Schema

Every record in `coding.json` and `reasoning.json` strictly adheres to the following JSON schema:

```json
{
  "benchmark": "HumanEval",
  "sub_benchmark": null,
  "problem_id": "HumanEval/0",
  "question": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
  "answer": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n"
}
```

### Schema Field Definitions

| Field | Type | Description |
| :--- | :--- | :--- |
| `benchmark` | `string` | Top-level benchmark family (e.g. `HumanEval`, `GPQA`, `LiveCodeBench`). |
| `sub_benchmark` | `string \| null` | Sub-track, variation, split, or language (e.g. `HumanEval+`, `Diamond`, `February 2025`). |
| `problem_id` | `string` | Benchmark-native unique ID, or deterministic synthetic ID fallback (`{benchmark}/{sub}/{index}`). |
| `question` | `string` | Normalized problem prompt, context, or instruction for the model. |
| `answer` | `string \| null` | Reference solution, gold answer, canonical code, or target output. |

---

## Supported Benchmarks

### Coding Benchmarks (`coding`)
- **HumanEval** (`openai/openai_humaneval`)
- **MBPP** (`Muennighoff/mbpp`)
- **EvalPlus** (`evalplus/humanevalplus`, `evalplus/mbppplus`)
- **EvalPerf** (`evalplus/evalperf`)
- **CruxEval** (`cruxeval-org/cruxeval`)
- **LiveCodeBench** (`livecodebench/code_generation_lite`)
- **ClassEval** (`FudanSELab/ClassEval`)
- **InfiBench** (`llylly001/InfiBench`)
- **EvoEval** (Difficult, Creative, Subtle, Tool Use, Combine)
- **SciCode** (`budecosystem/scicode`)
- **MultiPL-E** (`nuprl/MultiPL-E` — multi-language configs)
- **Multi-LCB** (`Multi-LCB/Multi-LCB`)
- **OJBench** (`He-Ren/OJBench_testdata`)
- **CodeElo** (`Qwen/CodeElo`)
- **DS-1000** (`embedding-benchmark/DS1000`)
- **LBPP** (`CohereLabs/lbpp`)

### Reasoning Benchmarks (`reasoning`)
- **BBEH** (`BBEH/bbeh`)
- **AGIEval** (`junnyu/agieval`, `hails/agieval-jec-qa-ca`)
- **HLE** (`cais/hle`) & **HLE-Verified** (`skylenage-ai/HLE-Verified`)
- **LiveBench** (`livebench/reasoning`, `livebench/math`)
- **GPQA & SuperGPQA** (`Idavidrein/gpqa`, `skylion007/SuperGPQA`)
- **PopQA** (`skylion007/PopQA-exact`)
- **ZebraLogic** (`reasoning-mazes/ZebraLogic` — Grid and Multiple Choice)
- **AutoLogi** (`GlobalBetween/autologi-qa`)
- **MiniF2F** (`m-a-p/miniF2F`)
- **HMMT** (`MathArena/hmmt_feb_2025`, `MathArena/hmmt_nov_2025`)
- **IMO-Bench** (`Hwilner/imo-answerbench`, `imo-proofbench`, `imo-gradingbench`)
- **AIME** (`math-ai/aime26`)

---

## Installation & Setup

### 1. Requirements
- Python 3.9+
- `datasets` (>= 2.16.0 recommended)
- `pyarrow`
- `huggingface_hub`

```bash
pip install -U datasets pyarrow huggingface_hub
```

### 2. Hugging Face Authentication
Several reasoning benchmarks (such as GPQA and HLE) require acceptance of gated dataset terms or an authenticated Hugging Face user token.

```bash
# Option A: Export environment variable
export HF_TOKEN="hf_your_access_token_here"

# Option B: Log in via CLI
huggingface-cli login
```

---

## Usage & Execution

### Basic Run (All Benchmarks)
```bash
python3 download_and_normalize_benchmarks.py \
    --output-dir ./benchmarks
```

### Passing Token via CLI Flag
```bash
python3 download_and_normalize_benchmarks.py \
    --hf-token "hf_your_access_token_here" \
    --output-dir ./benchmarks
```

### Targeted Execution (Single or Selected Benchmarks)
You can filter execution to specific benchmarks to speed up iteration:

```bash
python3 download_and_normalize_benchmarks.py \
    --benchmarks HumanEval MBPP GPQA LiveBench \
    --output-dir ./benchmarks
```

---

## Output Artifacts

The output directory contains four primary files:

```
output_dir/
├── coding.json                # Canonical normalized coding records
├── reasoning.json             # Canonical normalized reasoning records
├── download_manifest.json     # Per-dataset download status, row counts & error logs
└── normalization_errors.json  # Validation report for any malformed records
```

### `download_manifest.json` Structure
```json
{
  "coding": [
    {
      "status": "OK",
      "benchmark": "HumanEval",
      "hf_id": "openai/openai_humaneval",
      "config": null,
      "split": "test",
      "rows": 164,
      "normalized": 164
    }
  ],
  "reasoning": [ ... ]
}
```

---

## Comprehensive Caveats & Technical Nuances

### 1. Remote Code Execution (`trust_remote_code=True`)
* **Behavior:** Older and specialized datasets on Hugging Face Hub (e.g. `nuprl/MultiPL-E`, `embedding-benchmark/DS1000`, `openai/openai_humaneval`) bundle custom Python loader scripts.
* **Caveat:** In Hugging Face `datasets` versions `>= 2.16.0` and `3.x`, `load_dataset` will fail with an error unless `trust_remote_code=True` is explicitly passed. This script automatically enables this parameter.

### 2. LiveBench Multi-Turn and Field Formats
* **Behavior:** `LiveBench` stores problems in a `turns` array (e.g. `turns: [{"role": "user", "content": "..."}]` or `turns: ["Question..."]`) and answers under `ground_truth`.
* **Caveat:** Standard question extractors that look for `question` or `prompt` will return `None` on LiveBench, silently dropping all samples. The script's `generic_question` parses `turns` properly.

### 3. GPQA Answer & Choice Extraction
* **Behavior:** `Idavidrein/gpqa` does **not** store choices under standard `Choice A..D` fields; instead, it provides `Question`, `Correct Answer`, and `Incorrect Answer 1..3`.
* **Caveat:** Without synthesizing options, questions lack multiple-choice alternatives. The normalizer combines correct and incorrect answers into labeled choices `(A), (B), (C), (D)` and appends them to the question body.

### 4. AutoLogi Context + Question Merging
* **Behavior:** AutoLogi problems consist of a logic puzzle description under `context` and a specific query under `question`.
* **Caveat:** Using `or` extraction (e.g. `context or question`) drops the actual question query. The normalizer joins both sections clearly labeled.

### 5. MultiPL-E Combinatorial Expansion
* **Behavior:** MultiPL-E consists of 50+ programming language configurations across HumanEval and MBPP translations.
* **Caveat:** With `all_configs=True`, `MultiPL-E` yields tens of thousands of records. Downloading all configs may take several minutes. Use `--benchmarks` to filter when testing.

### 6. MBPP Target Function Signatures
* **Behavior:** MBPP prompts in `text` specify what the function should do, but the expected function name and signature are defined inside `test_list[0]`.
* **Caveat:** The normalizer attaches the first test case signature to ensure models generate functions with matching parameter names.

### 7. Multimodal Benchmarks (e.g., HLE)
* **Behavior:** Humanity's Last Exam (`cais/hle`) includes questions that reference visual diagrams (stored in an `image` column).
* **Caveat:** This pipeline extracts textual questions and answers. For multimodal benchmarks, problems requiring visual interpretation may only retain textual captions/prompts.

### 8. Gated Datasets & Terms Acceptance
* **Behavior:** Certain datasets on Hugging Face require the user to accept a click-through license on the Hugging Face web interface before access is granted.
* **Caveat:** If access has not been granted on Hugging Face, loading will return an HTTP 403 / Gated Repo error. Ensure you visit the dataset page and accept terms when using gated targets.

### 9. Memory & Arrow Table Iteration Performance
* **Caveat:** In PyArrow-backed datasets, using `for i in range(len(ds)): ds[i]` creates significant row slicing overhead. The script utilizes generator iteration (`for row in ds`) for linear memory and maximum throughput.

---

## Troubleshooting & FAQ

### Q1: Why did a dataset return `403 Forbidden` or `GatedRepoError`?
**Answer:** The dataset is gated on Hugging Face. Visit the Hugging Face URL for that dataset, log in, accept the terms, and pass your `HF_TOKEN`.

### Q2: How can I add a new benchmark?
**Answer:**
1. Append a `DatasetSpec` to the `CODING` or `REASONING` list in `download_and_normalize_benchmarks.py`.
2. (Optional) If the dataset has non-standard columns, add a custom `normalize_<benchmark>` function to `NORMALIZERS`.

### Q3: Why is `DeepSeek LeetCode` skipped?
**Answer:** DeepSeek LeetCode currently does not have a single standard first-party Hugging Face release. It is marked as `PLACEHOLDER_DO_NOT_USE` until a specific repository mirror is selected.
