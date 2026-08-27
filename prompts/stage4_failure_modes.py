"""Stage 4 prompts — two-pass failure-mode extraction.

PROMPT_PASS_A (reasoning-anchored): given the correct reasoning trace, find the
failures that occur ALONG the correct solution path.

PROMPT_PASS_B (anticipatory, wrong-solver simulation): WITHOUT the trace,
simulate plausible-but-wrong first attempts — catches failures that never
appear on the correct path.
"""
from string import Template

PROMPT_PASS_A = Template(r"""You are an expert AI evaluator and curriculum designer.

You have a problem, its correct reasoning trace, and its answer. Your task is to
identify the specific reasons why a capable but imperfect model or learner would
FAIL this problem — specifically, failures that occur ALONG or NEAR the correct
solution path.

Think carefully about each of these failure types:

MISSING_PREREQUISITE:
  A foundational concept required by the solution is not known at all.
  The solver cannot even begin the correct approach because they lack a building block.
  Example: Does not know modular inverse exists → cannot complete a number theory solution.

WRONG_MENTAL_MODEL:
  The concept is known but the learner has an incorrect or incomplete understanding
  of how it works in practice.
  Example: Knows binary search exists but believes it only works on strictly increasing
  arrays, so rejects it when duplicates are present.

MISSING_TRICK_OR_INSIGHT:
  A non-obvious insight is required that you either know or you do not — it cannot
  be derived by brute force reasoning alone in reasonable time.
  Example: "The answer is always the XOR of all elements" — you need to have seen
  this trick; a solver without it will spin indefinitely.

COMMON_MISTAKE:
  A mistake that is easy to make and looks almost right, that passes most test cases
  but fails on specific ones.
  Example: Using < instead of <= in a boundary check, causing an off-by-one that
  only manifests on inputs where the boundary is exactly hit.

FALSE_ASSUMPTION:
  The solver assumes something about the input or problem structure that is not
  guaranteed, producing a solution that passes most cases but fails when the
  assumption is violated.
  Example: Assuming the input array is always non-empty; assuming values are always
  positive; assuming the graph is always connected.

MISREAD_CONSTRAINTS:
  The problem statement contains a constraint that is easy to overlook or misread,
  and missing it leads to a fundamentally different (and wrong) solution.
  Example: "Return the indices, not the values" — solver returns values.
  Example: "All elements are distinct" is NOT stated, but solver assumes it.

MISSING_DOMAIN_KNOWLEDGE:
  Specialized knowledge outside of core algorithms is required.
  Example: Knowing Python float has 53-bit mantissa precision.
  Example: Knowing a specific mathematical identity or theorem by name.
  Example: Knowing that a particular graph structure guarantees a property.

SHORTCUT_ATTEMPT:
  The solver tries a simpler approach (greedy, brute force, heuristic) that seems
  to work on the examples provided but fails on edge cases or at scale.
  Example: Greedy interval selection when problem weights require DP.

OVERCOUNTING_OR_UNDERCOUNTING:
  In combinatorics, counting, or probability: double-counting overlapping outcomes
  (e.g. failing inclusion-exclusion) or undercounting by missing symmetries/partitions.
  Example: Counting configurations where order does not matter as if order did matter.

INCOMPLETE_CASE_ANALYSIS:
  The solver sets up valid casework but omits one or more critical subcases, boundary
  scenarios, or degenerate configurations.
  Example: Handling positive roots but omitting zero or negative roots.

UNJUSTIFIED_LOGICAL_STEP:
  In mathematical deduction or proof: asserting a property or step without sufficient
  justification, or confusing necessary vs sufficient conditions (affirming the consequent).
  Example: Assuming f(x) is monotonic without verifying its derivative sign.

MUTABLE_STATE_OR_ALIASING:
  In coding: unintended in-place mutation of a shared data structure, shallow copy bugs,
  closure variable capture in loops, or modifying a container during iteration.
  Example: Appending a mutable list to results without creating a copy.

TYPE_OR_PRECISION_ERROR:
  In coding/math: floating-point precision loss, integer division truncation, numeric
  overflow, or type mismatch.
  Example: Direct float equality `a == b` instead of `abs(a - b) < eps`.

OTHER:
  Use ONLY if the failure mechanism genuinely does not fit any category above.
  When using OTHER, you MUST supply "proposed_new_type" in UPPER_SNAKE_CASE.

Problem:
${question}

Correct Reasoning Trace:
${reasoning}

Correct Answer:
${answer}

Normalized concept vocabulary (use ONLY these for concept_involved, pick closest match;
if genuinely new, use same dot-notation format and set is_new_concept=true):
${normalized_taxonomy}

Return a JSON array of failure modes. Each entry:
{
  "failure_type": "MISSING_PREREQUISITE | WRONG_MENTAL_MODEL | MISSING_TRICK_OR_INSIGHT | COMMON_MISTAKE | FALSE_ASSUMPTION | MISREAD_CONSTRAINTS | MISSING_DOMAIN_KNOWLEDGE | SHORTCUT_ATTEMPT | OVERCOUNTING_OR_UNDERCOUNTING | INCOMPLETE_CASE_ANALYSIS | UNJUSTIFIED_LOGICAL_STEP | MUTABLE_STATE_OR_ALIASING | TYPE_OR_PRECISION_ERROR | OTHER",
  "proposed_new_type": "string (UPPER_SNAKE_CASE, only when failure_type is OTHER, else null)",
  "description": "1-2 sentences: exactly what the failure is for THIS specific problem, not generic",
  "concept_involved": "canonical.concept.from.taxonomy",
  "is_new_concept": true | false,
  "severity": "critical | major | minor",
  "what_correct_understanding_looks_like": "1 sentence: what the solver needs to know or do instead",
  "source": "reasoning_anchored"
}

Return 2-4 failure modes. Prioritize severity (critical first).
Return ONLY a valid JSON array. No explanation outside the array.""")

PROMPT_PASS_B = Template(r"""You are simulating a capable but imperfect AI model or learner attempting a ${domain}
problem cold — without any hints about the correct approach.

Your goal is to generate PLAUSIBLE WRONG SOLUTION ATTEMPTS: the kinds of approaches
that a model or learner would confidently try, often without realizing they are wrong,
before seeing the correct solution.

IMPORTANT: This is NOT about finding edge cases in the correct solution.
This IS about: what completely different, wrong approaches would seem reasonable to
try from the beginning?

You will be shown the correct answer ONLY for reference to verify your wrong attempts
are actually wrong. Do NOT let the correct answer anchor your thinking — generate
the wrong attempts first, then check they are wrong.

Problem:
${question}

Correct Answer (for reference only — do not anchor to this):
${answer}

Generate 3-5 distinct wrong solution attempts. Each attempt must represent a solver
who starts from a plausible but wrong frame or strategy. At least ONE attempt must
represent a solver who never gets close to the correct approach — not someone who
got 90% of the way there.

For each wrong attempt, identify its root cause from these categories:

WRONG_PROBLEM_FRAME:
  Solver misunderstood what the problem is asking at a fundamental level.
  They are solving a related but different problem than the one stated.
  Example: Problem asks for count of valid pairs; solver finds the pairs themselves.
  Example: Problem asks for minimum cost; solver finds any valid solution ignoring cost.

PLAUSIBLE_WRONG_ALGORITHM:
  Solver correctly recognized the problem domain but chose an algorithm that works
  on most cases but fails on this problem's specific constraints or structure.
  Example: Uses greedy for a problem that requires DP because of overlapping subproblems.
  Example: Uses BFS for a weighted shortest path problem (should use Dijkstra).

KNOWLEDGE_ILLUSION:
  Solver believes they know a theorem, formula, or rule and applies it, but their
  understanding is subtly wrong — the rule has a condition they are not checking.
  Example: Applying AM-GM without verifying non-negativity of terms.
  Example: Using the formula for combinations but forgetting the ordering constraint.
  Example: Thinking binary search works on any sequence, not just sorted ones.

PATTERN_OVERFITTING:
  Solver has seen similar-looking problems and applies the pattern from those problems
  without verifying it applies here. Surface features of the problem trigger a
  memorized (wrong) template.
  Example: Seeing "subarray" and immediately applying sliding window template, which
  requires non-negative values that this problem does not guarantee.
  Example: Seeing a tree and applying in-order traversal template when problem needs
  post-order.

COMPLEXITY_BLINDNESS:
  Solver produces a logically correct approach that is computationally infeasible
  for the given constraints, not recognizing that their solution will TLE.
  Or: solver "optimizes" their approach in a way that inadvertently changes its semantics.
  Example: Correct O(n^3) solution for n=10^5 input.
  Example: Caching results incorrectly, making a correct algorithm return stale values.

PHANTOM_CONSTRAINT:
  Solver adds a constraint that is NOT stated in the problem (because it appears in
  similar problems they have seen), artificially restricting their solution space.
  Example: Assuming graph is undirected when it is directed.
  Example: Assuming values are distinct when duplicates are allowed.
  Example: Assuming input is 1-indexed when it is 0-indexed.

TERMINATION_ERROR:
  Solver's approach and algorithm are directionally correct but the solution terminates
  too early, too late, or on the wrong condition — missing or overcounting results.
  Example: Returning on first match instead of continuing to find the best match.
  Example: Loop runs to n instead of n-1, processing a phantom element.

REPRESENTATION_ERROR:
  Solver chooses the wrong data structure or representation for the problem, causing
  a correct conceptual algorithm to produce wrong results or be infeasible.
  Example: Using adjacency matrix for a sparse graph with 10^5 nodes (memory blows up).
  Example: Storing cumulative counts when the problem requires point values.

OTHER:
  Use ONLY if the wrong attempt represents a fundamentally different failure mechanism.
  When using OTHER, you MUST supply "proposed_new_type" in UPPER_SNAKE_CASE.

For each wrong attempt, return:
{
  "attempt_description": "2-3 sentences: what the wrong solver does, specifically",
  "why_it_seems_reasonable": "1-2 sentences: why a capable solver would confidently try this",
  "wrong_answer_or_behavior": "what this approach produces or how it fails",
  "failure_type": "WRONG_PROBLEM_FRAME | PLAUSIBLE_WRONG_ALGORITHM | KNOWLEDGE_ILLUSION | PATTERN_OVERFITTING | COMPLEXITY_BLINDNESS | PHANTOM_CONSTRAINT | TERMINATION_ERROR | REPRESENTATION_ERROR | OTHER",
  "proposed_new_type": "string (UPPER_SNAKE_CASE, only when failure_type is OTHER, else null)",
  "description": "1-2 sentences: the precise failure for THIS specific problem",
  "concept_involved": "canonical.concept.from.taxonomy or new concept in dot-notation",
  "is_new_concept": true | false,
  "severity": "critical | major | minor",
  "what_correct_understanding_looks_like": "1 sentence: what the solver needs to know or do instead",
  "source": "anticipatory"
}

Normalized concept vocabulary (prefer these for concept_involved):
${normalized_taxonomy}

Return ONLY a valid JSON array of wrong attempts. No explanation outside the array.""")
