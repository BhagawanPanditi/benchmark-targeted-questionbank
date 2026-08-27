"""Stage 2 prompts — freeform concept extraction.

PROMPT_CODING / PROMPT_REASONING extract 3-8 dot-notation concept tags a solver
must know to solve the problem. Tags are freeform here; Stage 3 normalizes
them into the controlled vocabulary.
"""
from string import Template

PROMPT_CODING = Template(r"""You are a programming education expert.

Analyze this problem, its reasoning trace, and its answer. Extract every distinct
concept, technique, data structure, algorithm, or pattern that a solver MUST know
or apply to solve this problem correctly.

Include concepts at ALL levels:
  - The primary algorithm or technique (e.g., algorithms.technique.two-pointer)
  - Data structures used (e.g., data-structures.mapping.dictionary)
  - Mathematical tools required (e.g., math.modular-arithmetic.modular-inverse)
  - Implicit prerequisites the solution relies on without stating them explicitly
    (e.g., if the solution uses binary search, include it even if not named)

Return concepts in taxonomical dot-notation: category.subcategory.specific-concept

Rules:
  - Each concept must be a reusable, teachable skill — not specific to this one problem
  - Use lowercase-with-hyphens within levels, dots between levels
  - Aim for 3-8 concepts per problem
  - Top-level categories should be chosen from:
    algorithms, data-structures, functionality, analytics, graph, dp, string, math

Examples:
  algorithms.technique.two-pointer
  data-structures.mapping.dictionary
  algorithms.sorting.merge-sort
  dp.technique.memoization
  math.modular-arithmetic.modular-inverse
  graph.traversal.bfs
  string.matching.kmp

Problem:
${question}

Reasoning Trace:
${reasoning}

Answer:
${answer}

Return ONLY valid JSON:
{"raw_concepts": ["category.subcategory.concept", ...]}""")

PROMPT_REASONING = Template(r"""You are a mathematics and logic education expert.

Analyze this problem, its reasoning trace, and its answer. Extract every distinct
concept, theorem, technique, or mathematical tool that a solver MUST know or apply
to solve this problem correctly.

Include concepts at ALL levels:
  - The primary method or theorem (e.g., combinatorics.counting.inclusion-exclusion)
  - Supporting mathematical tools (e.g., number-theory.divisibility.prime-factorization)
  - Logical structures used (e.g., logic.proof-technique.contradiction)
  - Implicit prerequisites the solution relies on without naming them

Return concepts in taxonomical dot-notation: category.subcategory.specific-concept

Rules:
  - Each concept must be a reusable, teachable skill — not specific to this one problem
  - Use lowercase-with-hyphens within levels, dots between levels
  - Aim for 3-8 concepts per problem
  - Top-level categories should be chosen from:
    algebra, combinatorics, number-theory, geometry, logic, probability, calculus,
    proof-technique

Examples:
  combinatorics.counting.inclusion-exclusion
  number-theory.divisibility.prime-factorization
  logic.constraint-satisfaction.backtracking
  algebra.inequalities.am-gm
  probability.expectation.linearity-of-expectation
  proof-technique.induction.strong-induction

Problem:
${question}

Reasoning Trace:
${reasoning}

Answer:
${answer}

Return ONLY valid JSON:
{"raw_concepts": ["category.subcategory.concept", ...]}""")
