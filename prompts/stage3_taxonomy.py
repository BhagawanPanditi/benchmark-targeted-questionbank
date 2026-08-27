"""Stage 3 prompt — taxonomy normalization (one-time, per domain).

PROMPT_NORMALIZE turns the full list of raw concept tags into a canonical
controlled vocabulary: merged synonyms, removed noise, exactly 3
dot-notation levels per concept, and a raw->canonical merge map.
"""
from string import Template

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
