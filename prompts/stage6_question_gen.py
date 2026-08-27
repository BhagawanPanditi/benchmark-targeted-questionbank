"""Stage 6 prompts — diagnostic question generation (one per failure mode).

PROMPT_CODING / PROMPT_REASONING write a new, standalone, easier question that
specifically exposes one failure mode, with failure-type-specific design rules
and a specific, checkable trap answer. The reasoning variant must include the
exact answer inline.
"""
from string import Template

PROMPT_CODING = Template(r"""You are an expert coding educator creating a diagnostic question bank.

A model or learner is likely to fail a benchmark coding problem due to a specific
reason. Write a new, standalone coding question that directly targets and exposes
this exact failure mode.

Source Problem (for context only — do NOT reproduce it, do not refer to it):
${question}

Failure Mode to Target:
  Type: ${failure_type}
  Description: ${description}
  Concept Involved: ${concept_involved}
  What correct understanding looks like: ${what_correct_understanding_looks_like}
  Why the wrong approach seems reasonable: ${why_it_seems_reasonable}

Related prerequisite concepts (for context — the learner may also be weak here):
  ${ancestor_concepts}

CRITICAL REQUIREMENTS — your question must satisfy ALL of these:

1. EASIER than the source problem. This is a prerequisite diagnostic, not a peer problem.
   A learner working toward the source problem should be able to attempt this first.

2. TARGETED: Tests the SPECIFIC failure mode described, not the general topic area.
   Someone who has exactly the described failure would likely answer WRONG.
   Someone without that failure would likely answer RIGHT.
   The question should discriminate between these two learners.

3. FAILURE-TYPE-SPECIFIC design rules:

   For MISSING_PREREQUISITE:
     Write a question that requires exactly that prerequisite concept and nothing harder.
     The question should be unsolvable without that concept but straightforward with it.

   For WRONG_MENTAL_MODEL:
     Include a case where the wrong mental model produces a different answer than the
     correct mental model. The wrong answer must be specific and predictable.
     Example: If the wrong model is "binary search only works on strictly sorted arrays,"
     include an array with duplicates where the correct answer differs from what the
     wrong model produces.

   For MISSING_TRICK_OR_INSIGHT:
     Make the problem computationally intractable or clearly wrong without the trick.
     A brute force attempt should either be obviously O(n^3) or produce wrong output.
     The trick should be the clean, elegant unlock.

   For COMMON_MISTAKE:
     Design the question to TEMPT the specific mistake. The wrong answer that results
     from the mistake should be a specific, predictable value — not just "wrong."
     Include an example where the mistake produces a plausible-looking wrong answer.

   For FALSE_ASSUMPTION:
     Include an input that VIOLATES the false assumption (e.g. negative numbers, empty inputs,
     non-distinct values). The question must make clear this input is valid.

   For MISREAD_CONSTRAINTS:
     Include a constraint that is easy to overlook. A careful reader and a careless
     reader should get different, specific answers. State the constraint clearly but
     not prominently.

   For MISSING_DOMAIN_KNOWLEDGE:
     Isolate exactly the piece of domain knowledge described (e.g. language-specific behavior,
     IEEE float precision). The question should be trivial given that knowledge and opaque without it.

   For SHORTCUT_ATTEMPT:
     Include a specific case where the shortcut fails. The shortcut should work on
     all other examples, but fail on the included edge case.

   For OVERCOUNTING_OR_UNDERCOUNTING:
     Include elements with overlap or symmetry. The question should specifically expose
     double-counting overlapping cases or failing to divide by identical permutations.

   For INCOMPLETE_CASE_ANALYSIS:
     Require handling a small set of cases where omitting one boundary case (e.g. 0, empty,
     single element, duplicates) produces a specific wrong answer.

   For UNJUSTIFIED_LOGICAL_STEP:
     Design a problem where a tempting heuristic or unproven assumption (e.g. assuming
     symmetry or monotonicity) fails on an explicit counterexample.

   For MUTABLE_STATE_OR_ALIASING:
     Construct a scenario where in-place mutation of a shared list/object or closure variable
     capture causes subsequent operations or return values to be corrupted.

   For TYPE_OR_PRECISION_ERROR:
     Construct a problem where float rounding, integer overflow, or integer division truncation
     creates a noticeable numerical error if not handled properly.

   For WRONG_PROBLEM_FRAME:
     Make the true goal subtly different from the tempting misframed goal (e.g. finding
     indices vs values, count vs subset), so the misframed solver outputs a specific wrong format/value.

   For PLAUSIBLE_WRONG_ALGORITHM:
     Set up constraints that make the wrong algorithm seem correct (passes small cases)
     but reveal its failure on a slightly larger or differently structured case (e.g. greedy vs DP).

   For KNOWLEDGE_ILLUSION:
     Write a question where the illusion (the subtly wrong rule with unmet preconditions)
     gives a specific wrong answer, and the correct rule gives a different specific right answer.

   For PATTERN_OVERFITTING:
     Make the problem superficially resemble the pattern the overfitter would apply
     (e.g. sliding window) but include a structural feature that breaks that pattern.

   For COMPLEXITY_BLINDNESS:
     Make constraints tight enough that a naive O(N^2) or brute force solution will time out,
     requiring the linear or logarithmic approach.

   For PHANTOM_CONSTRAINT:
     Write a question where the phantom constraint (the one that is NOT there) would
     change the answer if it were true. Make both answers specific values.

   For TERMINATION_ERROR:
     Construct a problem where an off-by-one loop limit or early termination condition produces
     a specific missing or extra element.

   For REPRESENTATION_ERROR:
     Construct a problem where an inefficient or incorrect data structure (e.g. list lookups
     instead of hash sets) causes wrong outputs or performance failure.

   For OTHER:
     Target the specific root cause described; ensure the wrong approach yields a predictable,
     checkable trap output.

4. STANDALONE: Solvable without any reference to the source problem or benchmark.
   No prior context needed.

5. CONCRETE: Include 2-3 example inputs with expected outputs. At least one example
   must be chosen to specifically illustrate the failure mode — the wrong approach
   gives a wrong answer on this example, and the right approach gives the right answer.

6. SPECIFIC WRONG ANSWER: The "trap" must produce a specific, predictable wrong answer
   — not just "an incorrect result." A reviewer should be able to verify the trap claim.

Return ONLY valid JSON:
{
  "question": "complete question text including examples",
  "what_it_tests": "one sentence: the specific understanding verified by this question",
  "trap": "one sentence: what a failing learner does AND the specific wrong answer they get",
  "why_trap_is_tempting": "one sentence: why the wrong approach seems reasonable to a capable solver",
  "difficulty": "beginner | intermediate | advanced",
  "failure_type": "${failure_type}",
  "concept_involved": "${concept_involved}",
  "tags": ["canonical.concept.1", "canonical.concept.2"]
}""")

PROMPT_REASONING = Template(r"""You are an expert mathematics and reasoning educator creating a diagnostic question bank.

A model or learner is likely to fail a benchmark reasoning or math problem due to a
specific reason. Write a new, standalone question that directly targets and exposes
this exact failure mode.

Source Problem (for context only — do NOT reproduce it, do not refer to it):
${question}

Failure Mode to Target:
  Type: ${failure_type}
  Description: ${description}
  Concept Involved: ${concept_involved}
  What correct understanding looks like: ${what_correct_understanding_looks_like}
  Why the wrong approach seems reasonable: ${why_it_seems_reasonable}

Related prerequisite concepts (for context):
  ${ancestor_concepts}

CRITICAL REQUIREMENTS — your question must satisfy ALL of these:

1. EASIER than the source problem. This is a prerequisite diagnostic.

2. TARGETED: Tests the SPECIFIC failure mode, not the general topic area.
   Discriminates between a learner with the failure and one without it.

3. FAILURE-TYPE-SPECIFIC design rules:

   For MISSING_PREREQUISITE:
     Write a problem that directly checks the prerequisite definition, formula, or lemma.
     The problem should be clean and straightforward once the prerequisite is known.

   For MISSING_DOMAIN_KNOWLEDGE:
     Isolate the specific math identity, theorem, or property. The problem should be
     unsolvable without that knowledge but direct with it.

   For SHORTCUT_ATTEMPT:
     Construct a problem where a tempting heuristic or superficial shortcut fails on
     an explicit boundary case or scale.

   For MISSING_TRICK_OR_INSIGHT:
     The problem is not solvable by brute enumeration in a reasonable way. The insight
     (e.g. parity, invariants, symmetry, algebraic substitution) is the only clean path.

   For COMMON_MISTAKE:
     The most natural first calculation or algebraic slip gives a specific wrong numerical answer.
     Name the exact trap value in the trap description.

   For FALSE_ASSUMPTION:
     Construct a scenario that explicitly violates the false assumption (e.g. non-integer values,
     negative numbers, non-disjoint sets). The wrong answer and the right answer must be different numbers.

   For MISREAD_CONSTRAINTS:
     A careful reader and a careless reader get different numerical answers.
     The constraint that is easy to miss must be stated clearly but not highlighted.

   For WRONG_MENTAL_MODEL / KNOWLEDGE_ILLUSION:
     Two plausible-sounding approaches give two different numerical answers.
     Only one is mathematically correct. Name the wrong approach and its specific wrong number.

   For OVERCOUNTING_OR_UNDERCOUNTING:
     A combinatorics or counting question where overlapping subsets or symmetries tempt
     a naive count (e.g. double-counting the intersection or forgetting to divide by n!).
     The trap answer must be the exact unadjusted count.

   For INCOMPLETE_CASE_ANALYSIS:
     A problem requiring 2-3 distinct cases where omitting one case (e.g. $x=0$, degenerate case)
     yields a specific wrong sum or count.

   For UNJUSTIFIED_LOGICAL_STEP:
     A problem where assuming an unproven property (e.g. assuming the converse or assuming
     collinearity) leads to a specific wrong result.

   For WRONG_PROBLEM_FRAME:
     A problem where misinterpreting what quantity is being asked for (e.g. perimeter vs area,
     probability vs odds) yields a specific wrong numerical value.

   For PLAUSIBLE_WRONG_ALGORITHM:
     A problem where a standard but inappropriate technique (e.g. greedy pairing instead of
     dynamic programming/bipartite matching) yields a suboptimal/incorrect count.

   For PATTERN_OVERFITTING:
     Problem superficially resembles a known formula/pattern but has a structural feature
     that breaks the formula. Both the pattern answer and correct answer must be specific numbers.

   For COMPLEXITY_BLINDNESS:
     Make numbers large enough that manual brute force enumeration is impossible, forcing
     algebraic reduction or combinatorial factoring.

   For PHANTOM_CONSTRAINT:
     The assumed constraint (not in the problem) would change the answer if true.
     State both what the answer would be with and without the phantom constraint.

   For TERMINATION_ERROR:
     A sequence, series, or recurrence problem where stopping one term early or late
     produces a specific off-by-one value.

   For REPRESENTATION_ERROR / TYPE_OR_PRECISION_ERROR:
     A problem exposing fractional representation errors or modular residue representation errors.

   For OTHER:
     Target the specific failure described; ensure the wrong approach yields a checkable wrong number.

4. DEFINITE ANSWER: Exactly one correct answer, verifiable without ambiguity.
   Include the answer inline.

5. STANDALONE: No reference to source problem needed.

6. SPECIFIC WRONG ANSWER: The trap must produce a specific, checkable wrong value.

Return ONLY valid JSON:
{
  "question": "complete question text",
  "answer": "exact correct answer",
  "answer_explanation": "2-3 sentences: why this is correct and why the trap answer is wrong",
  "what_it_tests": "one sentence: specific understanding verified",
  "trap": "one sentence: what failing learner does AND the specific wrong answer they get",
  "why_trap_is_tempting": "one sentence: why the wrong approach seems reasonable",
  "difficulty": "beginner | intermediate | advanced",
  "failure_type": "${failure_type}",
  "concept_involved": "${concept_involved}",
  "tags": ["canonical.concept.1", "canonical.concept.2"]
}""")
