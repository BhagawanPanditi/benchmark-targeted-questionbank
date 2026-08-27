"""Stage 7 prompt — 4-criteria quality review of one generated question.

PROMPT_VALIDATE checks discrimination, isolation, drift, and trap validity;
the question passes only if all four criteria are met.
"""
from string import Template

PROMPT_VALIDATE = Template(r"""You are a quality reviewer for a diagnostic question bank.

This question was generated to target a specific failure mode. Determine whether
it actually does so, or whether it has drifted into a generic topic question.

Failure Mode It Should Target:
  Type: ${failure_type}
  Description: ${description}
  What correct understanding looks like: ${what_correct_understanding_looks_like}

Generated Question:
${question}

Stated trap (what a failing learner does and their specific wrong answer):
${trap}

Why the trap is tempting:
${why_trap_is_tempting}

Evaluate on exactly these four criteria:

DISCRIMINATION:
  Would a learner WITH the failure mode described likely answer this WRONG?
  Would a learner WITHOUT it likely answer it RIGHT?
  Pass requires YES to both. Fail if the question is too easy (everyone gets it right)
  or too hard (everyone gets it wrong regardless of the failure mode).

ISOLATION:
  Does the question test primarily this failure mode, or does it require so many
  other unrelated concepts that a learner could fail for completely unrelated reasons?
  Pass if the failure mode is the primary discriminator.
  Fail if the question is a multi-concept problem where this failure is one of many.

DRIFT:
  Has the question stayed focused on the failure mode, or drifted into being a
  generic question on the topic area?
  no_drift: clearly targets the failure mode
  minor_drift: mostly on target with slight generalization
  major_drift: has become a generic topic question

TRAP_VALIDITY:
  Is the stated trap answer actually wrong (not a trick question where both
  answers are defensible)? Is the wrong answer specific and checkable?
  valid: trap produces a clearly wrong, specific answer
  invalid: trap answer is ambiguous, or actually could be correct

A question PASSES only if:
  discrimination = pass
  AND isolation = pass
  AND drift = no_drift OR minor_drift
  AND trap_validity = valid

Return ONLY valid JSON:
{
  "passes": true | false,
  "discrimination": "pass | fail",
  "isolation": "pass | fail",
  "drift": "no_drift | minor_drift | major_drift",
  "trap_validity": "valid | invalid",
  "reason": "one sentence: why it passes, or the primary reason it fails"
}""")
