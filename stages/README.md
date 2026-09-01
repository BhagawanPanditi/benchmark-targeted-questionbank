# `stages/` — Detailed Guide to Every Pipeline Stage

This document explains, in **extreme detail**, what each module in the `stages/`
folder does, why it exists, what data it reads, what data it writes, how it
resumes after interruption, and what the important edge cases are.

If you want the short version, the pipeline turns raw benchmark problems into a
bank of **targeted prerequisite questions**. For each source problem, it tries to
answer this question:

> *What smaller, easier question could I ask that would reveal the exact reason a
> solver would fail the original problem?*

The `stages/` folder is where that logic lives.

---

## 1. What lives in `stages/`

The folder contains these stage modules:

- `stage1_reasoning.py`
- `stage2_raw_concepts.py`
- `stage3_taxonomy.py`
- `stage4_failure_modes.py`
- `stage5_concept_graph.py`
- `stage6_question_gen.py`
- `stage7_validation.py`
- `stage8_output.py`
- `stage9_readme.py`

Each stage is one step in the pipeline. The stages are orchestrated by
`run_pipeline.py`, but the actual per-stage behavior is implemented here.

The prompts used by these stages are **not** in this folder; they live in
`prompts/`. Shared helpers such as JSON I/O, LLM calling, similarity scoring,
and severity ranks live in `utils/`.

---

## 2. Big-picture flow

At a high level, the pipeline does this:

1. Start from source benchmark records in `coding.json` and/or `reasoning.json`.
2. Generate a gold-grounded reasoning trace for each problem.
3. Extract the concepts needed to solve each problem.
4. Normalize those concepts into a domain-wide controlled vocabulary.
5. Identify the ways a solver could fail each problem.
6. Build a prerequisite graph over the concept vocabulary.
7. Generate one easier diagnostic question per failure mode.
8. Validate and deduplicate the generated questions.
9. Assemble the final question-bank JSON files.
10. Regenerate the root `README.md` with live stats.

That flow is implemented as nine numbered stages.

---

## 3. Execution order and orchestration details

`run_pipeline.py` is the entry point. It imports all stage modules and calls
their `run(...)` functions.

### Actual execution pattern

For stages **1 through 8**:

- the pipeline iterates by **stage number first**
- then by **domain** (`coding`, `reasoning`)

So if you run all stages for both domains, the order is:

1. Stage 1 for coding
2. Stage 1 for reasoning
3. Stage 2 for coding
4. Stage 2 for reasoning
5. ...
6. Stage 8 for coding
7. Stage 8 for reasoning
8. Stage 9 once

### Special case: Stage 9

Stage 9 is different:

- it does **not** run once per domain
- it runs **once total**, after all requested domain work for Stages 1–8 is done
- it regenerates the repository root `README.md`

### CLI notes that matter to stage behavior

The pipeline supports:

- `--stages 1,2,3`
- `--domain coding`
- `--concurrency 30`
- `--resume`

Important detail: `--resume` is effectively a compatibility flag. Resume behavior
is **always on**. The stages are written assuming they may be re-run many times.

---

## 4. Cross-cutting design principles used by almost every stage

Before going stage by stage, it helps to understand the design patterns repeated
throughout the codebase.

### 4.1 Resumability is a first-class feature

Nearly every stage loads any existing output file, indexes existing records by a
stable key, and skips work already done.

Common resume keys:

- Stage 1: `problem_id`
- Stage 2: `problem_id`
- Stage 4: `problem_id`
- Stage 6: `(source_problem_id, failure_index)`
- Stage 7: `id`

This means a crash or interruption usually loses **at most one in-flight item**.

### 4.2 Atomic writes

The helper `utils.io.save_json(...)` writes to `file.tmp`, flushes and `fsync`s,
then replaces the destination with `os.replace(...)`.

This matters because:

- partially written JSON files are avoided
- every finished record is checkpointed durably
- re-runs can safely continue from the last successful item

### 4.3 Full-file rewrite after each completed item

Most stages do **not** append a single record to disk. Instead, they:

1. load the full current output
2. update an in-memory map
3. rewrite the entire JSON file atomically

That is slower than append-only logging, but much simpler and safer for resume.

### 4.4 LLM calls all go through one shared helper

All stage LLM traffic goes through `utils.llm.call_llm(...)`.

That helper provides:

- OpenAI-compatible client pointed at the vLLM endpoint from `config.py`
- global concurrency limiting with `asyncio.Semaphore`
- retries on **any** exception
- exponential backoff: `1s`, `2s`, `4s`, ...
- per-call timeout from `config.TIMEOUT_SECONDS`
- JSON extraction/parsing when `expect_json=True`

Important subtlety: a stage-level retry and an LLM-helper retry are different.

Example:

- Stage 1 may retry reasoning generation up to **3 answer attempts** if the model
  keeps giving a reasoning trace whose final stated answer does not match the gold.
- But **each** of those attempts also passes through `call_llm(...)`, which itself
  may retry request/format failures up to `config.RETRY_ATTEMPTS + 1` total tries.

So there is a difference between:

- *infrastructure / response-format failure* → handled inside `call_llm`
- *content-level failure* (wrong answer, bad schema, unusable response) → handled by the stage

### 4.5 Domain split: coding vs reasoning

Most content-producing stages have separate prompts for:

- `coding`
- `reasoning`

The two domains share structure but differ in expectations.

Examples:

- Stage 1 uses different reasoning prompts for coding vs math/reasoning.
- Stage 2 uses different concept taxonomies.
- Stage 6 reasoning questions must include an exact answer and explanation;
  coding questions do not.

### 4.6 Deterministic vs LLM-driven stages

Some stages are LLM-heavy. Others are purely deterministic.

**LLM-driven:**

- Stage 1
- Stage 2
- Stage 3
- Stage 4
- Stage 5
- Stage 6
- Stage 7 (partly deterministic, partly LLM review)

**Deterministic / no LLM calls:**

- Stage 8
- Stage 9

### 4.7 “Skip”, “fail”, and “discard” do not all mean the same thing

The code uses several different failure semantics:

- **Skip**: already processed or intentionally not processed because a cap or
  resume rule applies.
- **Failed but retained**: the record stays in output but is marked failed.
  Example: Stage 1 stores `reasoning_status="failed"`.
- **Discarded / omitted from downstream artifact**: the current stage does not
  emit a usable record for that item.
  Example: Stage 2 may log a discard if raw concept extraction produces invalid
  JSON or no usable concept list.
- **Validated and failed**: Stage 7 keeps the question record but marks
  `validation_passed=false` and records the reason.

That distinction matters a lot when debugging the pipeline.

---

## 5. File evolution: what gets created where

For each domain, the pipeline evolves artifacts through these files:

| Stage | Domain input(s) | Domain output(s) |
| --- | --- | --- |
| 1 | `coding.json` / `reasoning.json` | `coding_with_reasoning.json` / `reasoning_with_reasoning.json` |
| 2 | `*_with_reasoning.json` | `*_raw_concepts.json` |
| 3 | `*_raw_concepts.json` | `*_taxonomy.json` and in-place update of `*_raw_concepts.json` with `normalized_concepts` |
| 4 | `*_with_reasoning.json` + `*_taxonomy.json` | `*_with_failure_modes.json` |
| 5 | `*_taxonomy.json` | `*_concept_graph.json` plus temporary `*.progress.json` during execution |
| 6 | `*_with_failure_modes.json` + `*_concept_graph.json` | `*_questions_raw.json` |
| 7 | `*_questions_raw.json` + source `*.json` | `*_questions_validated.json` |
| 8 | `*_questions_validated.json` + `*_raw_concepts.json` + `*_concept_graph.json` | `*_prerequisite_questions.json` |
| 9 | all generated artifacts | root `README.md` |

One unusual design detail:

- Stage 3 writes a **separate taxonomy file**, but it also mutates the Stage 2
  output file in place by adding `normalized_concepts` to each record.

That is why later stages sometimes read `*_raw_concepts.json` again even though
Stage 3 already produced a taxonomy artifact.

---

## 6. Record evolution: how one source problem changes over time

Start with a source record like:

```json
{
  "benchmark": "HumanEval",
  "sub_benchmark": null,
  "problem_id": "HumanEval/0",
  "question": "...",
  "answer": "..."
}
```

Then fields are added or re-expressed across stages.

### After Stage 1

Adds:

- `reasoning`
- `reasoning_status`

### After Stage 2

Adds:

- `raw_concepts`

### After Stage 3

Adds to the same Stage 2 records:

- `normalized_concepts`

### After Stage 4

Writes a separate record that includes:

- original source fields
- `reasoning`
- `reasoning_status`
- `failure_modes` (a list)

### After Stage 6

The pipeline is no longer storing one record per source problem. It is now
storing one record per **(problem, failure_mode)** pair.

Each generated question gets its own UUID and metadata such as:

- `id`
- `source_problem_id`
- `failure_index`
- `failure_type`
- `concept_involved`
- `question`
- `trap`
- `difficulty`
- and more

### After Stage 7

Adds:

- `validation_passed`
- `validation_reason`

### After Stage 8

Filters to only `validation_passed=true` and outputs final question-bank records
with:

- source metadata
- failure metadata
- question text
- concept metadata
- `prerequisite_depth`
- answer fields for reasoning
- `generated_answer: null`

---

## 7. Stage-by-stage deep dive

# Stage 1 — `stage1_reasoning.py`

## Purpose

Stage 1 generates a **gold-grounded reasoning trace** for each source problem.

It is the pipeline’s first interpretation step. Instead of jumping directly from
problem statement to concepts or failure modes, the pipeline first asks the model
for a detailed expert solution path.

That reasoning trace becomes the backbone for several downstream steps:

- Stage 2 uses it to infer which concepts are truly required.
- Stage 4 Pass A uses it to identify failures that happen **along the correct path**.
- Stage 1 itself also acts as a quality gate: if the model cannot produce a
  reasoning trace whose final answer agrees with the gold answer, the problem is
  marked unusable for downstream reasoning-dependent steps.

## Inputs

For one domain, Stage 1 reads the fixed source file:

- `coding.json` or
- `reasoning.json`

Each record is expected to contain:

- `benchmark`
- `sub_benchmark`
- `problem_id`
- `question`
- `answer`

## Outputs

It writes:

- `coding_with_reasoning.json` or
- `reasoning_with_reasoning.json`

Each output record contains:

- `benchmark`
- `sub_benchmark`
- `problem_id`
- `question`
- `answer`
- `reasoning`
- `reasoning_status`

## Core logic

For each problem, Stage 1:

1. Selects the domain-specific prompt.
2. Inserts the problem statement and gold answer into that prompt.
3. Calls the LLM for a freeform reasoning trace.
4. Extracts the final answer from the last occurrence of the exact phrase:
   `Therefore, the answer is:`
5. Normalizes the extracted answer and the gold answer.
6. Compares them.
7. If they match, accepts the reasoning trace.
8. If they do not match, retries generation up to `MAX_ANSWER_ATTEMPTS = 3`.
9. If all attempts fail, stores the last reasoning text anyway but marks the
   record with `reasoning_status="failed"`.

## Why the answer check exists

Without the answer check, Stage 1 could generate a plausible-looking but
incorrect chain of reasoning, and every later stage would treat that bad trace
as ground truth.

The answer check is therefore a **guardrail** against reasoning drift.

It does **not** prove the reasoning is logically perfect. It only proves that the
trace ends at the correct gold answer after normalization.

## Important helper functions

### `clean_answer(text)`

This strips markdown code fences if the answer is wrapped in triple backticks.
That matters especially for coding tasks, where the answer may be a code snippet.

### `normalize_answer(text)`

This normalizes for comparison by:

- cleaning fences
- lowercasing
- collapsing whitespace
- stripping surrounding punctuation and quotes

This prevents superficial formatting differences from being treated as answer
mismatches.

### `extract_final_answer(reasoning)`

This searches the reasoning trace for the last `Therefore, the answer is:` and
returns whatever comes after it.

Using the **last** occurrence is intentional. If the model says the phrase more
than once during drafting, the stage uses the final conclusion.

### `answers_match(stated, gold)`

This first compares normalized full strings. Then it has one extra coding-aware
fallback:

- if the gold answer is a single line
- and the model’s stated answer is multi-line
- then Stage 1 also checks whether the **first line** of the stated answer
  matches the gold

This helps accept answers where the model gives a valid first line followed by
extra formatting or explanation.

## Domain-specific prompts

### Coding prompt

The coding reasoning prompt explicitly asks for:

1. Restate the task
2. Constraints
3. Core insight
4. Approach
5. Edge cases
6. Final exact line with the gold answer

### Reasoning prompt

The reasoning prompt asks for:

1. Restate
2. Given information and load-bearing constraints
3. Key theorem / trick / insight
4. Full solution steps
5. Sanity check / verification
6. Final exact line with the gold answer

## Resume behavior

Resume key: `problem_id`

If a record with the same `problem_id` already exists in the output file, Stage 1
skips it entirely.

That means Stage 1 is safe to re-run many times, but it also means that if you
want to regenerate reasoning for a single record, you would need to remove that
record from the Stage 1 output file first.

## Failure behavior

There are two major failure classes:

### 1. LLM call failure

If the API call itself fails even after `call_llm(...)` retries, the stage logs
an error and stops retrying that problem at the answer-content level.

### 2. Content failure

If the model returns text but:

- omits the required final line, or
- states an answer that does not match the gold,

then Stage 1 retries up to 3 times.

If still unsuccessful, the record is **retained** but marked failed.

## Downstream consequence of `reasoning_status="failed"`

This is extremely important:

- Stage 2 processes only `reasoning_status == "ok"`
- Stage 4 also processes only `reasoning_status == "ok"`

So a Stage 1 failure effectively removes that source problem from the rest of the
pipeline.

## Why Stage 1 exists at all

You could imagine extracting concepts directly from the problem statement. But
that would often miss implicit prerequisites and solution-specific insights.

Stage 1 creates a more explicit representation of:

- what the problem is really asking
- what constraints matter
- what key idea unlocks the problem
- where solvers are likely to slip

It turns an opaque benchmark item into a more structured teaching object.

---

# Stage 2 — `stage2_raw_concepts.py`

## Purpose

Stage 2 extracts a **freeform concept list** for each successfully reasoned
problem.

This is the pipeline’s first attempt to say:

> *What skills, methods, tools, theorems, data structures, or ideas must a solver
> actually know to solve this problem?*

The concepts are intentionally **not fully normalized yet**. This stage favors
coverage and recall over perfect naming consistency.

## Inputs

Reads:

- `coding_with_reasoning.json` or
- `reasoning_with_reasoning.json`

But only records with:

- `reasoning_status == "ok"`

are processed.

## Outputs

Writes:

- `coding_raw_concepts.json` or
- `reasoning_raw_concepts.json`

Each output record includes:

- source fields
- `reasoning`
- `reasoning_status`
- `raw_concepts`

## Core logic

For each eligible record, Stage 2:

1. Chooses the coding or reasoning concept-extraction prompt.
2. Supplies:
   - question
   - reasoning trace
   - gold answer
3. Expects JSON containing `{"raw_concepts": [...]}`.
4. Validates and sanitizes the concept list.
5. Writes the record if the concept list is usable.

## Sanitization rules

The helper `_coerce_raw_concepts(...)` does several things:

- accepts either a list directly or a dict with `raw_concepts`
- keeps only string items
- strips whitespace
- lowercases every concept
- removes duplicates while preserving first-seen order
- returns `None` if the result is empty or invalid

That means Stage 2 is permissive about minor format variation but strict about
requiring a real concept list.

## Prompt expectations

The prompt asks for **3–8 dot-notation concept tags** such as:

- `algorithms.technique.two-pointer`
- `data-structures.mapping.dictionary`
- `combinatorics.counting.inclusion-exclusion`

The prompt also explicitly tells the model to include:

- primary methods
- supporting tools
- implicit prerequisites
- reusable teachable concepts, not problem-instance descriptions

## Why concepts are “raw” here

At this point, the pipeline is still operating per problem. If you normalize too
early, you risk throwing away useful distinctions before you have seen the full
corpus.

So Stage 2 intentionally allows:

- synonyms
- inconsistent naming
- slight taxonomy drift
- duplicates across problems

Stage 3 cleans that up later at the **domain-wide** level.

## Resume behavior

Resume key: `problem_id`

If the output file already contains that `problem_id`, the record is skipped.

## Failure behavior

If the LLM call fails or the response does not contain a valid usable concept
list, Stage 2 logs a discard and **does not write a record** for that problem.

This is different from Stage 1, which retains failures with a status flag.

So Stage 2 output is effectively “the subset of Stage 1 successes for which
concept extraction succeeded.”

## Why Stage 2 exists separately from Stage 3

The pipeline wants two things that are in tension:

1. **per-problem coverage** of all relevant concepts
2. **global consistency** in concept naming

Stage 2 optimizes for the first.
Stage 3 optimizes for the second.

Keeping them separate is a strong design choice.

---

# Stage 3 — `stage3_taxonomy.py`

## Purpose

Stage 3 converts the domain’s noisy set of raw concept tags into a **controlled,
canonical taxonomy**.

This is one of the most structurally important stages in the system. It takes a
problem-local concept vocabulary and turns it into a domain-level ontology that
later stages can share.

Without this stage, downstream outputs would fragment badly:

- one record might say `dp.dynamic-programming.knapsack`
- another `algorithms.dp.0-1-knapsack`
- another `dp.knapsack.zero-one`

Stage 3 makes those comparable.

## Inputs

Reads:

- `coding_raw_concepts.json` or
- `reasoning_raw_concepts.json`

## Outputs

Writes two things:

1. A taxonomy document:
   - `coding_taxonomy.json`
   - `reasoning_taxonomy.json`
2. An **in-place update** to `*_raw_concepts.json`, adding `normalized_concepts`
   to each record.

## What the taxonomy file contains

The taxonomy document includes:

- `domain`
- `taxonomy` — canonical concept list
- `merge_map` — raw concept → canonical concept
- `removed` — dropped raw concepts with reasons
- `category_summary` — top-level category counts
- `generated_at`

## Core algorithm

### Step 3.1 — collect unique raw concepts

The stage scans all Stage 2 records and builds an **order-preserving** list of
all unique raw concept strings across the whole domain.

### Step 3.2 — normalize with one LLM call

If no taxonomy already exists, the stage sends the full raw concept list to the
Stage 3 prompt, which asks the model to:

- merge synonyms
- remove noise
- enforce exactly 3 dot-separated levels
- standardize naming
- restrict top-level categories to allowed domain-specific sets

### Step 3.3 — write the taxonomy file

The LLM response is validated enough to require at least a `taxonomy` list.
Then the stage writes the taxonomy document.

### Step 3.4 — add `normalized_concepts` to every Stage 2 record

Each record’s `raw_concepts` are mapped through `merge_map` using
`_normalize_list(...)`.

This:

- drops blanks
- preserves order
- de-duplicates

The updated Stage 2 file is saved after each record missing normalization.

## Re-run behavior: Stage 3 is intentionally “one-time”

If the taxonomy file already exists and contains a `merge_map`, Stage 3 does
**not** call the LLM again.

Instead, it:

- reuses the existing `merge_map`
- fills in `normalized_concepts` only for any records missing them

This is important because taxonomy design should be relatively stable. Rebuilding
it every run would make downstream artifacts harder to compare over time.

## Edge case: no raw concepts

If the Stage 2 file contains no raw concepts at all, Stage 3 writes an empty
taxonomy and ensures all records have `normalized_concepts = []`.

This allows downstream stages to keep running instead of crashing.

## Why exactly 3 levels matter

The Stage 3 prompt enforces concept strings like:

- `algorithms.technique.two-pointer`
- `proof-technique.induction.strong-induction`

The explicit three-level structure has several benefits:

- concepts are more regular and comparable
- top-level category summaries become meaningful
- graph nodes become cleaner
- generated tags are easier to inspect manually

## Resume behavior

Stage 3 is not resumed per problem in the same way as Stages 1, 2, 4, or 6.
Instead, it has a **mode switch**:

- if no taxonomy exists → build it once
- if taxonomy exists → reuse it and only apply the merge map to missing records

## Why Stage 3 matters downstream

Stage 4 relies on this normalized vocabulary for `concept_involved`.
Stage 5 builds a concept graph over this vocabulary.
Stage 6 uses graph ancestors derived from it.
Stage 8 uses normalized concepts to define each source problem’s “primary concept.”

So Stage 3 is where the pipeline stops being merely a collection of per-item LLM
outputs and becomes a **shared structured system**.

---

# Stage 4 — `stage4_failure_modes.py`

## Purpose

Stage 4 identifies the **specific ways a solver could fail** each source problem.

This is arguably the conceptual heart of the project.

The project is not merely generating easier related questions. It is trying to
generate questions that diagnose **why** the original problem would be failed.

To do that, Stage 4 extracts failure modes in **two different passes**:

1. **Reasoning-anchored pass (Pass A)**
2. **Anticipatory wrong-solver pass (Pass B)**

That two-pass design is the most distinctive idea in the pipeline.

## Inputs

Reads:

- `*_with_reasoning.json`
- `*_taxonomy.json`

Processes only records with:

- `reasoning_status == "ok"`

## Outputs

Writes:

- `coding_with_failure_modes.json`
- `reasoning_with_failure_modes.json`

Each output record contains:

- source fields
- `reasoning`
- `reasoning_status`
- `failure_modes` — a merged list from both passes

## Why two passes are necessary

### Pass A: reasoning-anchored

This pass asks:

> *Assume we know the correct solution path. Where would a capable but imperfect
> solver likely stumble along or near that path?*

This is good for finding things like:

- missing prerequisite steps
- off-by-one errors
- false assumptions near the correct approach
- incomplete case analysis
- missing a key theorem or identity already visible in the correct solution

### Pass B: anticipatory wrong-solver simulation

This pass asks:

> *If a solver attacked the problem cold, what plausible but wrong approaches would
> they confidently try from the start?*

This is essential for hard problems, because many failures do **not** happen near
the correct path at all. The solver may instead:

- frame the problem incorrectly
- pick the wrong algorithm family
- apply a memorized pattern that superficially matches
- assume a phantom constraint from similar benchmarks

If the pipeline only used Pass A, it would miss a large class of realistic model
failures.

## Failure-type vocabulary

Stage 4 validates against the union of two failure-type sets in
`utils/constants.py`.

### Pass A style types include

- `MISSING_PREREQUISITE`
- `WRONG_MENTAL_MODEL`
- `MISSING_TRICK_OR_INSIGHT`
- `COMMON_MISTAKE`
- `FALSE_ASSUMPTION`
- `MISREAD_CONSTRAINTS`
- `MISSING_DOMAIN_KNOWLEDGE`
- `SHORTCUT_ATTEMPT`
- `OVERCOUNTING_OR_UNDERCOUNTING`
- `INCOMPLETE_CASE_ANALYSIS`
- `UNJUSTIFIED_LOGICAL_STEP`
- `MUTABLE_STATE_OR_ALIASING`
- `TYPE_OR_PRECISION_ERROR`
- `OTHER`

### Pass B style types include

- `WRONG_PROBLEM_FRAME`
- `PLAUSIBLE_WRONG_ALGORITHM`
- `KNOWLEDGE_ILLUSION`
- `PATTERN_OVERFITTING`
- `COMPLEXITY_BLINDNESS`
- `PHANTOM_CONSTRAINT`
- `TERMINATION_ERROR`
- `REPRESENTATION_ERROR`
- `OTHER`

## Core logic

For each eligible record, Stage 4:

1. Builds a string representation of the current normalized taxonomy.
2. Sends **two LLM calls in parallel**:
   - Pass A with question + reasoning + answer + taxonomy
   - Pass B with question + answer + taxonomy
3. Sanitizes each pass’s returned entries.
4. Merges the two pass outputs.
5. Registers any genuinely new concepts into the taxonomy.
6. Saves the problem with its final `failure_modes` list.

## Sanitization behavior

The `_sanitize(...)` helper ensures:

- each entry is a dict
- required fields exist and are non-empty:
  - `failure_type`
  - `description`
  - `concept_involved`
  - `what_correct_understanding_looks_like`
- failure types are uppercased and standardized
- concept names are lowercased
- severity is coerced to `critical`, `major`, or `minor` (default `major`)
- source is recorded as either `reasoning_anchored` or `anticipatory`

## New failure types

If the model emits:

- `OTHER` plus `proposed_new_type`, or
- a non-standard uppercase failure type,

Stage 4 may retain it and mark `is_new_failure_type=true`.

So the failure-type vocabulary is controlled, but not completely closed.

## Merge rule across Pass A and Pass B

The `_merge(...)` function deduplicates on the pair:

- `(failure_type, concept_involved)`

If both passes produce the same pair, Stage 4 keeps the better one using this rule:

1. higher severity wins
2. if severity ties, the **anticipatory** version wins

The reasoning for the tie-break is explicit in the code: anticipatory failures are
harder to generate and often more valuable because they capture truly off-path
mistakes.

## New concept registration

If a failure mode marks `is_new_concept=true`, Stage 4 adds that concept into the
existing taxonomy file by inserting identity mappings into `merge_map` and appending
it to `taxonomy`.

This matters because Stage 5 needs to know about all concepts that appear in failure modes.

## Output size expectations

- Pass A prompt asks for **2–4** failure modes.
- Pass B prompt asks for **3–5** wrong attempts.
- After merging, the code **warns** if the final count is not between **4 and 8**.

This range is advisory, not enforced. The stage accepts any non-empty usable merged set.

## Resume behavior

Resume key: `problem_id`

If a problem already exists in `*_with_failure_modes.json`, it is skipped.

## Failure behavior

If either of the parallel LLM calls fails at the API/JSON-parsing level after
retries, the problem is discarded from Stage 4 output.

If both sanitized lists together produce no usable merged failure modes, the
problem is also discarded.

Again, this is different from Stage 1: Stage 4 does not retain a failed record
with a status flag.

## Why Stage 4 matters so much

Stages 6 and 7 are only as good as the failure modes they are built on.

If Stage 4 is sharp and specific, Stage 6 can generate crisp diagnostic questions.
If Stage 4 is vague, Stage 6 will drift into generic easier problems.

This is the stage where the pipeline decides whether it is building a true
**diagnostic** benchmark or just a loosely related practice set.

---

# Stage 5 — `stage5_concept_graph.py`

## Purpose

Stage 5 builds a **prerequisite graph** over the normalized concept vocabulary.

The graph answers questions like:

- What must someone know before learning concept X?
- Which concepts are atomic leaves?
- Which prerequisites are one, two, or three steps below a concept?

This stage is important because the final dataset is supposed to be about
**prerequisites**, not just topic similarity.

## Inputs

Reads:

- `coding_taxonomy.json` or
- `reasoning_taxonomy.json`

That taxonomy may already include new concepts registered by Stage 4.

## Outputs

Writes:

- `coding_concept_graph.json`
- `reasoning_concept_graph.json`

During execution it also uses a progress file:

- `coding_concept_graph.json.progress.json`
- `reasoning_concept_graph.json.progress.json`

## Graph semantics

An edge is stored as:

```json
{"from": "target-concept", "to": "direct-prerequisite"}
```

So the direction is:

**concept → its prerequisite**

This is the reverse of some curriculum graphs that draw prerequisite → advanced concept.
In this repository, the stored edge means:

> “To understand `from`, you need `to` first.”

That convention is used consistently by Stage 5, Stage 6 ancestor lookup, and
Stage 8 prerequisite-depth calculation.

## Core logic

For each concept in the taxonomy, Stage 5:

1. Asks the LLM for up to 5 **direct prerequisites**.
2. Records whether the concept is a true `is_leaf` concept.
3. Registers any prerequisite concepts that are missing from the vocabulary.
4. Saves the per-concept result to the progress file.
5. After all concepts are processed, assembles all concept→prerequisite edges.
6. Detects and breaks cycles.
7. Computes transitive prerequisite closure up to depth 3.
8. Writes the final graph file.
9. Deletes the progress file.

## Direct vs transitive prerequisites

The prompt is explicit: Stage 5 wants only **direct** prerequisites.

Example:

- If A requires B
- and B requires C

then the LLM should say A requires B, **not** A requires C.

The transitive closure is computed later by deterministic graph traversal.

This separation is very important. If the LLM were allowed to return transitive
ancestors directly, the graph would become noisy and much harder to interpret.

## Per-concept response coercion

The helper `_coerce_result(...)`:

- accepts a dict response
- caps prerequisites at `MAX_PREREQUISITES = 5`
- allows either dict entries or raw strings
- lowercases concept names
- removes self-loops and duplicates
- forces `is_leaf=true` if no prerequisites remain

## New prerequisite concepts

If the model returns a prerequisite concept not already in the taxonomy,
Stage 5 registers it in the taxonomy file.

Subtle but important detail:

- the current run’s `concepts` list is built **once at the beginning**
- newly registered prerequisites are saved into the taxonomy file
- they may become nodes in the final graph immediately if referenced by edges
- but they are **not necessarily queried as target concepts in the same run**

In practice, that means a newly discovered prerequisite may appear in the graph
as a node with incoming references, but its own outgoing prerequisite edges may
only be filled in on a later re-run, when it is part of the taxonomy at startup.

That is one of the more subtle resume behaviors in the whole project.

## Two-layer resume design

Stage 5 has the most sophisticated resumability of any stage.

### Layer 1 — progress file during an active/incomplete run

After every concept, Stage 5 writes the per-concept result into the progress file.
If the run crashes halfway, the stage can continue from the partially filled
progress file.

### Layer 2 — re-derive from an already completed graph

If the final graph file already exists, Stage 5 tries to reconstruct per-concept
results from the graph itself.

It uses:

- `edges`
- `processed`

The `processed` field records which concepts were actually queried as targets.
That matters because some graph nodes may exist only because they appeared as
prerequisites, not because the stage ever asked the LLM about them.

If the graph already covers all current taxonomy concepts and no progress file is
present, the stage simply keeps the graph unchanged and makes **zero LLM calls**.

## Cycle detection and removal

A prerequisite graph should be a DAG, but LLM-generated prerequisite relations
can create cycles.

Example bad cycle:

- A requires B
- B requires C
- C requires A

Stage 5 fixes this deterministically.

### How cycles are found

It uses iterative DFS and looks for a **back edge**.

### How an edge is chosen for removal

Among the edges on the detected cycle, it removes the edge whose **source node**
has the lower centrality, where centrality is approximated as:

- in-degree + out-degree

Tie-break preference goes to the back edge itself.

Removed edges are logged and recorded in `removed_cycles`.

This is a heuristic, not a theorem-proven optimal cycle-breaking strategy, but it
is deterministic and practical.

## Transitive prerequisite closure

After cycle breaking, Stage 5 computes, for every concept, all prerequisites
reachable within depth 3 using BFS.

The result is stored under:

- `transitive_prerequisites`

This is a convenience index so downstream stages do not need to recompute closure.

## Final graph file structure

The graph JSON contains:

- `nodes`
- `edges`
- `leaves`
- `removed_cycles`
- `transitive_prerequisites`
- `processed`

## Failure behavior

If a concept-specific LLM call fails even after retries, the stage does **not**
abort the whole graph build.

Instead, it records that concept as failed and treats it as a node with no
prerequisite edges for this run.

That is an intentionally fault-tolerant design.

## Why Stage 5 exists

Without a concept graph, the pipeline could still generate easier questions, but
it would not know whether a generated question tests:

- the same concept
- a direct prerequisite
- a prerequisite of a prerequisite
- or something unrelated

Stage 5 gives the dataset a notion of **distance from the source concept**.

---

# Stage 6 — `stage6_question_gen.py`

## Purpose

Stage 6 generates the actual **diagnostic prerequisite questions**.

This is where the pipeline transforms abstract failure modes into concrete,
standalone items that can be asked to a learner or model.

Each generated question is supposed to be:

- easier than the source problem
- tightly targeted to one failure mode
- standalone
- paired with a specific trap / predictable wrong answer

## Inputs

Reads:

- `*_with_failure_modes.json`
- `*_concept_graph.json`

## Outputs

Writes:

- `coding_questions_raw.json`
- `reasoning_questions_raw.json`

Each output record corresponds to one **(source problem, failure mode index)** pair.

## Why the generation unit is `(problem, failure mode)`

A single source problem can fail for multiple distinct reasons.

For example, a solver might fail because they:

- lack a prerequisite theorem
- misread a constraint
- use the wrong algorithm family
- make an off-by-one mistake

Each of those deserves a **different** diagnostic question. So Stage 6 generates
one question per failure mode, not one question per source problem.

## Core logic

For each failure mode, Stage 6:

1. Identifies the `(concept_involved, failure_type)` pair.
2. Checks whether that pair already has enough coverage in the raw question bank.
3. If not capped, collects related ancestor concepts from the Stage 5 graph.
4. Sends a domain-specific prompt to the LLM.
5. Validates the returned JSON enough to require non-empty question text.
6. Normalizes difficulty if needed.
7. Writes a new question record.

## Coverage cap

The code enforces:

- `MAX_PER_CONCEPT_FAILURE_TYPE = 3`

meaning that if there are already 3 or more questions for the same:

- `concept_involved`
- `failure_type`

then Stage 6 skips generating another one.

This is a key anti-bloat mechanism.

Without it, high-frequency failure patterns could dominate the dataset and crowd
out diversity.

## Concurrency-safe slot reservation

Because Stage 6 runs many workers in parallel, the stage uses a lock to reserve
coverage slots atomically.

That means two concurrent workers cannot both see “2 existing questions” and both
proceed to generate a 3rd and 4th question for the same pair.

If generation fails after reserving a slot, the slot is released.

This is a careful concurrency detail and prevents coverage-cap races.

## Ancestor concept context

The stage builds a direct adjacency map from the graph file and computes
ancestors up to depth 2 via `_ancestors_up_to(...)`.

These are passed into the prompt as **related prerequisite concepts**.

Important nuance:

- this is prompt context only
- it does not force the generated question to test those ancestors
- the primary target remains the failure mode’s own `concept_involved`

The ancestor context helps the model stay at an appropriate prerequisite level.

## Prompt behavior

The prompts for coding and reasoning are extremely explicit.

They require:

- an easier question than the source problem
- tight targeting to the named failure mode
- a specific trap answer or wrong behavior
- standalone wording
- concrete examples
- a one-sentence explanation of what is being tested

The prompts also contain many **failure-type-specific design rules**.

Examples:

- `FALSE_ASSUMPTION` questions must include an input that violates the assumption.
- `COMMON_MISTAKE` questions must tempt a predictable wrong answer.
- `COMPLEXITY_BLINDNESS` questions must force a more efficient method.
- `WRONG_PROBLEM_FRAME` questions must make the tempting misframing produce a
  different specific output than the correct framing.

This is how the stage tries to keep questions diagnostic rather than generic.

## Output schema highlights

Every raw generated question includes fields such as:

- `id` — UUID4
- `source_problem_id`
- `source_benchmark`
- `source_sub_benchmark`
- `domain`
- `failure_index`
- `failure_type`
- `failure_source`
- `failure_severity`
- `failure_description`
- `what_correct_understanding_looks_like`
- `concept_involved`
- `question`
- `what_it_tests`
- `trap`
- `why_trap_is_tempting`
- `difficulty`
- `tags`

Reasoning questions additionally include:

- `answer`
- `answer_explanation`

Coding questions do not yet include a generated solution; that is deferred.

## Difficulty normalization

Accepted difficulties are:

- `beginner`
- `intermediate`
- `advanced`

Any other returned difficulty is coerced to:

- `intermediate`

## Resume behavior

Resume key: `(source_problem_id, failure_index)`

If that pair already exists in the raw question file, Stage 6 skips it.

This means that if a failure mode changes semantically but keeps the same index,
a re-run will not regenerate that question unless the old record is removed.
That is worth remembering when editing Stage 4 outputs manually.

## Failure behavior

If the LLM call fails or the response is missing non-empty `question` text,
Stage 6 discards that question and writes nothing for that pair.

## Why Stage 6 is not the final output yet

Even with careful prompts, generated questions can still be bad.
They may be:

- too similar to source questions
- near-duplicates of each other
- too generic
- not truly diagnostic of the intended failure mode

So Stage 6 is intentionally just **raw generation**, not publication.

---

# Stage 7 — `stage7_validation.py`

## Purpose

Stage 7 validates and deduplicates generated questions before they become part of
the final bank.

This stage combines:

1. deterministic contamination checking
2. deterministic intra-bank deduplication
3. LLM-based diagnostic quality review

It is the pipeline’s quality-control layer.

## Inputs

Reads:

- `*_questions_raw.json`
- source `coding.json` or `reasoning.json`

## Outputs

Writes:

- `coding_questions_validated.json`
- `reasoning_questions_validated.json`

Crucially, this file contains **all validated records**, both passed and failed.
Nothing is silently dropped at this point.

Each validated record is just the raw question record plus:

- `validation_passed`
- `validation_reason`

## Validation Step 7.1 — contamination check

This step checks whether a generated question is too similar to any original
benchmark question.

### Similarity metric

The similarity function is simple token-level Jaccard overlap using:

- lowercase
- whitespace tokenization
- set semantics

So punctuation, phrasing, and multiplicity are not modeled in a sophisticated way.
This is intentionally lightweight and deterministic.

### Threshold

A question is contaminated if its best similarity satisfies:

- `Jaccard > 0.85`

Note the strict `>`.
Not `>=`.

### Why this exists

The generated prerequisite bank is supposed to contain **new** easier diagnostic
questions, not lightly paraphrased benchmark items.

## Validation Step 7.2 — intra-bank deduplication

This step removes near-duplicate generated questions.

### Grouping key

Questions are grouped by:

- `concept_involved`

This is important: deduplication is **not** limited to identical `failure_type`.
If two questions test the same concept and are near-duplicates in wording, one may
be removed even if their failure types differ.

### Threshold

A duplicate is declared if:

- `Jaccard > 0.92`

again using a strict `>`.

### Tie-break rule

If two questions are near-duplicates, `_pick_loser(...)` chooses which one to drop:

1. lower failure severity loses
2. if severity ties, `reasoning_anchored` loses to `anticipatory`
3. if still tied, later deterministic sort order loses

This means the system prefers to retain:

- more severe failures
- anticipatory/off-path failures
- earlier stable items

## Validation Step 7.3 — LLM review

Questions that survive contamination and deterministic dedup are sent to the
Stage 7 validation prompt.

The validator scores exactly four criteria:

1. `discrimination`
2. `isolation`
3. `drift`
4. `trap_validity`

### Pass rule

A question passes only if:

- `discrimination == pass`
- `isolation == pass`
- `drift in {no_drift, minor_drift}`
- `trap_validity == valid`

The code recomputes pass/fail from those four fields rather than trusting the
validator’s top-level `passes` boolean.

That is a nice defensive design detail.

## Meaning of the four review criteria

### Discrimination

Would someone **with** the failure mode likely get it wrong, while someone
**without** the failure would likely get it right?

If not, it is not diagnostic.

### Isolation

Does the question mainly test the named failure mode, or does it require too many
other unrelated skills?

If it requires too much extra knowledge, a wrong answer is hard to interpret.

### Drift

Did the question stay focused on the specific failure mode, or drift into a
broader generic topic question?

This is critical because generic “easier related problems” are not the goal.

### Trap validity

Is the stated trap actually a specific, checkably wrong outcome rather than a
vague claim?

The project wants trap answers that are inspectable and falsifiable.

## Resume behavior

Resume key: `id`

Questions already present in the validated file are skipped.

However, deterministic contamination and dedup checks are recomputed over the
**full raw question set** on each run before deciding what to do with pending
items. This helps preserve consistency for newly processed records.

## Output ordering

Unlike several earlier stages, Stage 7 explicitly preserves raw-question order in
its validated output file via `_ordered_records(...)`.

That makes downstream review more stable and easier to compare.

## Failure behavior

### Deterministic failure

If a question is contaminated or deduplicated, it is written with:

- `validation_passed = false`
- a human-readable `validation_reason`

### LLM review failure

If the validation LLM call itself fails, the question is also written as failed,
with the reason recording the LLM error.

So Stage 7 never silently loses a question. It always records the verdict.

## Why Stage 7 matters

Stage 6 can generate text that looks polished, but polished is not enough.

Stage 7 tries to ensure final questions are:

- genuinely new
- non-redundant
- truly diagnostic
- anchored to a specific failure mode

This stage is what separates a raw generation pipeline from a usable dataset
construction pipeline.

---

# Stage 8 — `stage8_output.py`

## Purpose

Stage 8 assembles the final per-domain question-bank files.

It is a deterministic packaging stage. It does **not** call the LLM.

Its jobs are to:

1. keep only questions that passed validation
2. compute each question’s `prerequisite_depth`
3. emit the final stable JSON schema used by downstream consumers

## Inputs

Reads:

- `*_questions_validated.json`
- `*_raw_concepts.json`
- `*_concept_graph.json`

## Outputs

Writes:

- `coding_prerequisite_questions.json`
- `reasoning_prerequisite_questions.json`

## Filtering behavior

Only records with:

- `validation_passed is True`

are included.

Everything else remains in the validated file for auditability but is excluded
from the final deliverable.

## Primary concept of a source problem

Stage 8 defines the source problem’s **primary concept** as:

- the first element of `normalized_concepts`

loaded from `*_raw_concepts.json`.

This is a strong simplifying assumption: the first normalized concept is treated
as the main concept the original benchmark problem is about.

That assumption is then used to compute prerequisite depth for generated questions.

## Computing `prerequisite_depth`

The stage builds an adjacency map from the concept graph where:

- `concept -> direct prerequisites`

Then for each passed question:

1. let `primary` = source problem’s first normalized concept
2. let `concept` = question’s `concept_involved`
3. BFS outward from `primary` along prerequisite edges up to depth 3
4. if `concept` is found, its distance is the `prerequisite_depth`
5. otherwise depth is 0

### Meaning of values

- `1` = direct prerequisite of the source problem’s primary concept
- `2` = prerequisite of a prerequisite
- `3` = three steps below
- `0` = same concept, unknown relation, or no path within tracked depth

### Important nuance

If the generated question tests the **same concept** as the source problem’s
primary concept, depth is `0`, not `1`.

So `0` does **not** always mean “bad” or “unrelated.” It can also mean “same level.”

## Output schema

Each final record includes:

- source benchmark metadata
- source problem id
- domain
- failure metadata
- `concept_involved`
- `prerequisite_depth`
- generated question content
- tags
- `generated_answer`

For `reasoning`, it also includes:

- `answer`
- `answer_explanation`

For `coding`, it sets:

- `generated_answer = null`

The code currently also sets `generated_answer = null` for reasoning, while still
including the explicit answer fields.

## Determinism and idempotence

Stage 8 simply recomputes the final file from upstream artifacts each run.

That means:

- no resume logic is necessary in the usual sense
- no partial state is preserved
- re-running Stage 8 is safe and cheap

## Why Stage 8 exists separately

Keeping final assembly separate from validation has a few advantages:

- the validated file can retain failures for debugging and auditing
- the final file stays clean and consumer-facing
- graph-derived metadata like prerequisite depth can be recomputed deterministically

---

# Stage 9 — `stage9_readme.py`

## Purpose

Stage 9 regenerates the repository root `README.md` from live pipeline artifacts.

This stage is purely documentation/reporting. It does not affect the actual
question-bank JSON outputs.

## Inputs

It reads, as available:

- input source files
- taxonomy files
- concept graph files
- raw question files
- final output files

## Output

Writes:

- root `README.md`

using an atomic text write.

## What it includes

The generated README contains sections for:

- overview
- input file stats
- input format
- how to run
- pipeline stages table
- concept taxonomy summaries
- concept graph stats
- output statistics
- output schema
- estimated LLM call budget

## Why this is useful

The root README is not hand-maintained static prose. Instead, it is a snapshot of
current pipeline state plus general usage documentation.

That means as artifacts evolve, the README can reflect:

- record counts
- taxonomy sizes
- graph sizes
- validation pass counts
- benchmark distribution
- failure-type distribution
- difficulty distribution

## Deterministic behavior

Stage 9 performs no LLM calls. It just formats whatever artifacts exist.

If an artifact is missing, the README section usually falls back gracefully, for
example by saying the taxonomy or graph has not been generated yet.

## Why Stage 9 is in `stages/`

Even though it is documentation-oriented, it still behaves like a pipeline stage:

- it has a numbered place in the run order
- it consumes prior artifacts
- it produces a derived artifact (`README.md`)

So conceptually it belongs with the other stages.

---

## 8. Stage dependency graph in words

Here is the dependency structure in plain English:

- Stage 1 depends only on the source input files.
- Stage 2 depends on Stage 1 success for each problem.
- Stage 3 depends on Stage 2 concept output across the whole domain.
- Stage 4 depends on Stage 1 reasoning and Stage 3 taxonomy.
- Stage 5 depends on the Stage 3/4 taxonomy state.
- Stage 6 depends on Stage 4 failure modes and Stage 5 graph context.
- Stage 7 depends on Stage 6 raw questions and the original source problems.
- Stage 8 depends on Stage 7 validated questions plus Stage 3/5 metadata.
- Stage 9 depends on whatever artifacts exist and summarizes them.

One compact way to think about it is:

```text
source problems
  -> reasoning
  -> concepts
  -> taxonomy
  -> failure modes
  -> concept graph
  -> raw diagnostic questions
  -> validated diagnostic questions
  -> final prerequisite question bank
  -> documentation snapshot
```

---

## 9. The most important conceptual ideas in this pipeline

If you only remember a few design ideas from this repository, these are the big ones.

### 9.1 Gold-grounded reasoning first

The pipeline does not immediately ask “what concepts are here?”
It first asks “how does an expert actually solve this?”

### 9.2 Failure modes, not generic topic decomposition

The generated questions are supposed to diagnose **specific failure mechanisms**,
not merely produce easier topic-adjacent exercises.

### 9.3 Two-pass failure extraction

Correct-path failures and wrong-first-attempt failures are both necessary to model
how systems actually fail on hard benchmarks.

### 9.4 Controlled vocabulary + prerequisite graph

Concept normalization and graphing turn freeform LLM outputs into a more coherent,
reusable curriculum structure.

### 9.5 Validation is multi-layered

The pipeline does not trust raw generation. It uses:

- contamination checking
- deduplication
- LLM review

before final publication.

### 9.6 Resumability is built into the architecture

The stages are designed to survive long runs, crashes, and partial progress.

---

## 10. Practical debugging notes

When something goes wrong, these are the first places to look.

### If Stage 1 has many failures

Likely causes:

- prompt/model mismatch
- gold answers that are hard to normalize
- coding answers with formatting the matcher does not tolerate
- reasoning traces not ending with the required final line

### If Stage 2 output is much smaller than Stage 1 success count

Likely causes:

- invalid JSON from the model
- model returning concepts outside the expected list structure
- empty or malformed `raw_concepts`

### If Stage 4 yields weak questions later

Often the real issue is earlier:

- vague or low-quality failure modes
- concept tags too broad or too noisy
- taxonomy drift causing poor `concept_involved` alignment

### If Stage 5 graph looks sparse

Possible reasons:

- the model is overusing `is_leaf=true`
- many concept calls failed and were retained as edge-less nodes
- new prerequisite concepts were discovered late and need a re-run to be queried as targets

### If Stage 6 generates too many similar questions

Check:

- whether failure modes are overly repetitive
- whether concept/failure-type coverage is being exhausted by similar records
- whether ancestor context is too generic

### If Stage 7 rejects a lot of questions

Find out whether rejections are mostly:

- contamination
- near-duplicate removal
- drift / isolation failure
- invalid trap claims

Those correspond to different root causes.

---

## 11. Summary in one sentence per stage

- **Stage 1:** Generate a gold-grounded reasoning trace and verify it ends at the correct answer.
- **Stage 2:** Extract freeform concepts required to solve each successfully reasoned problem.
- **Stage 3:** Normalize those concepts into a stable domain-wide taxonomy and annotate records with canonical forms.
- **Stage 4:** Identify specific failure modes using both correct-path analysis and wrong-solver simulation.
- **Stage 5:** Build a prerequisite DAG over the concept vocabulary and compute short-depth closure.
- **Stage 6:** Generate one easier, standalone diagnostic question per failure mode.
- **Stage 7:** Remove contaminated or redundant questions and validate whether each item truly diagnoses the intended failure.
- **Stage 8:** Keep only validated questions and package them into the final output schema with prerequisite depth metadata.
- **Stage 9:** Regenerate the root README from current artifacts and statistics.

---

## 12. If you are reading the code

A good order for reading the implementation is:

1. `run_pipeline.py`
2. `utils/llm.py`
3. `utils/io.py`
4. `stage1_reasoning.py`
5. `stage2_raw_concepts.py`
6. `stage3_taxonomy.py`
7. `stage4_failure_modes.py`
8. `stage5_concept_graph.py`
9. `stage6_question_gen.py`
10. `stage7_validation.py`
11. `stage8_output.py`
12. `stage9_readme.py`

That order mirrors how information gains structure as it flows through the system.
