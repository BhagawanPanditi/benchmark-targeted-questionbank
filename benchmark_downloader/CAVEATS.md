# Technical Caveats & Normalization Guide

This document catalogs technical caveats, edge cases, schema quirks, and Hugging Face dataset nuances encountered across the coding and reasoning benchmark suite.

---

## Summary of Critical Fixes & Gotchas

### 1. `trust_remote_code=True` Requirement
* **Affected Datasets:** `nuprl/MultiPL-E`, `embedding-benchmark/DS1000`, `openai/openai_humaneval`, `FudanSELab/ClassEval`.
* **Issue:** Datasets containing legacy dataset scripts fail with `ValueError: The repository ... contains custom code` unless `trust_remote_code=True` is explicitly passed in `load_dataset` and `get_dataset_config_names`.
* **Resolution:** All loader methods now pass `trust_remote_code=True`.

---

### 2. LiveBench Schema (`turns` & `ground_truth`)
* **Affected Datasets:** `livebench/reasoning`, `livebench/math`.
* **Issue:** LiveBench does not have a top-level `question` or `prompt` string. It uses a `turns` list (multi-turn conversation format) and stores answers under `ground_truth`.
* **Resolution:** The question extractor specifically parses `turns` lists, retrieving the first user turn content to prevent dropping LiveBench samples.

---

### 3. GPQA Answer Format & Options Extraction
* **Affected Datasets:** `Idavidrein/gpqa` (including Diamond split).
* **Issue:** GPQA rows on Hugging Face do not provide pre-formatted `Choice A..D` strings. The raw fields are:
  - `Question`
  - `Correct Answer`
  - `Incorrect Answer 1`
  - `Incorrect Answer 2`
  - `Incorrect Answer 3`
  - `Record ID`
* **Resolution:** The GPQA normalizer synthesizes choices `(A), (B), (C), (D)` using both correct and incorrect options, appends the choice list to the question body, and uses `Record ID` as the primary key.

---

### 4. AutoLogi Context & Query Concatenation
* **Affected Datasets:** `GlobalBetween/autologi-qa`.
* **Issue:** AutoLogi separates the logical puzzle definition (`context`) from the specific question asked (`question`). Using a fallback `or` operation causes one of the two parts to be dropped.
* **Resolution:** Both `context` and `question` are combined with explicit markdown headers:
  ```text
  Context:
  ...

  Question:
  ...
  ```

---

### 5. MultiPL-E Combinatorial Volume & Sub-benchmark Naming
* **Affected Datasets:** `nuprl/MultiPL-E`.
* **Issue:** MultiPL-E has over 50 language configurations (e.g. `humaneval-py`, `humaneval-java`, `mbpp-cpp`, etc.). When iterating over all configs, sub-benchmarks can produce duplicate tags (e.g. `humaneval-py/py`).
* **Resolution:** Sub-benchmark naming logic verifies whether the language tag is already present in the configuration slug before appending.

---

### 6. MBPP Target Signature Extraction
* **Affected Datasets:** `Muennighoff/mbpp`.
* **Issue:** The natural language prompt (`text`) describes the logic (e.g., *"Write a function to find the maximum sum..."*), but does not state the expected function identifier. The signature is embedded in `test_list[0]` (e.g., `assert max_sub_array(...) == ...`).
* **Resolution:** The first test case is appended as a reference signature so downstream code generation has unambiguous function entrypoints.

---

### 7. Multimodal Benchmarks & Image Columns
* **Affected Datasets:** `cais/hle` (Humanity's Last Exam).
* **Issue:** Some HLE questions reference visual diagrams or charts contained in an `image` column.
* **Resolution:** In pure text canonicalization, image payloads are omitted while retaining textual descriptions.

---

### 8. MiniF2F Formal vs. Informal Statements
* **Affected Datasets:** `m-a-p/miniF2F`.
* **Issue:** MiniF2F mathematical theorem records provide formal Lean/Isabelle/Metamath statements (`formal_statement`) and optional natural language translations (`informal_prefix`).
* **Resolution:** When `informal_prefix` is present, it is positioned above the formal statement.

---

### 9. PyArrow Table Iteration Performance
* **Issue:** Hugging Face `datasets.Dataset` objects are backed by PyArrow memory tables. Iterating via `for i in range(len(ds)): ds[i]` repeatedly slices the Arrow table row-by-row, introducing substantial CPU overhead on large datasets.
* **Resolution:** Direct iterator consumption `for row in ds:` runs at native C++/PyArrow speed.

---

### 10. Gated Hugging Face Datasets & Authentication
* **Affected Datasets:** `Idavidrein/gpqa`, `cais/hle`, etc.
* **Issue:** Gated repositories require users to authenticate and accept terms of service via the Hugging Face Web UI.
* **Resolution:** Authenticate via `export HF_TOKEN="hf_..."` or `huggingface-cli login` before running the pipeline.
