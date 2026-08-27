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
     Include an input that VIOLATES the false assumption. The question must make
     clear this input is valid. The wrong approach fails specifically on this input.

   For MISREAD_CONSTRAINTS:
     Include a constraint that is easy to overlook. A careful reader and a careless
     reader should get different, specific answers. State the constraint clearly but
     not prominently.

   For MISSING_DOMAIN_KNOWLEDGE:
     Isolate exactly the piece of domain knowledge described. The question should be
     trivial given that knowledge and opaque without it.

   For SHORTCUT_ATTEMPT:
     Include a specific case where the shortcut fails. The shortcut should work on
     all other examples. The question should include at least one example where the
     shortcut gives a wrong answer.

   For PLAUSIBLE_WRONG_ALGORITHM:
     Set up constraints that make the wrong algorithm seem correct (passes small cases)
     but reveal its failure on a slightly larger or differently structured case.

   For KNOWLEDGE_ILLUSION:
     Write a question where the illusion (the subtly wrong rule) gives a specific
     wrong answer, and the correct rule gives a different specific right answer.

   For PATTERN_OVERFITTING:
     Make the problem superficially resemble the pattern the overfitter would apply
     but include a structural difference that breaks that pattern.

   For PHANTOM_CONSTRAINT:
     Write a question where the phantom constraint (the one that is NOT there) would
     change the answer if it were true. Make both answers specific values.

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

3. FAILURE-TYPE-SPECIFIC design rules (same logic as coding, adapted for math/reasoning):

   For MISSING_TRICK_OR_INSIGHT:
     The problem is not solvable by brute enumeration in a reasonable way. The insight
     is the only clean path. Include a scale hint that makes brute force clearly infeasible.

   For COMMON_MISTAKE:
     The most natural first calculation or approach gives a specific wrong numerical answer.
     Include this wrong answer as a plausible-looking option (even in free-response format,
     name the trap answer so the validator can check it).

   For FALSE_ASSUMPTION:
     Construct a scenario that explicitly violates the false assumption.
     The wrong answer (from the assumption) and the right answer must be different numbers.

   For MISREAD_CONSTRAINTS:
     A careful reader and a careless reader get different numerical answers.
     The constraint that is easy to miss must be stated but not highlighted.

   For WRONG_MENTAL_MODEL / KNOWLEDGE_ILLUSION:
     Two plausible-sounding approaches give two different numerical answers.
     Only one is mathematically correct. Name the wrong approach so the trap is verifiable.

   For PATTERN_OVERFITTING:
     Problem superficially resembles a known pattern but has a structural feature
     that breaks the pattern. Both the pattern answer and the correct answer must
     be specific numbers.

   For PHANTOM_CONSTRAINT:
     The assumed constraint (not in the problem) would change the answer if true.
     State both what the answer would be with and without the phantom constraint.

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
