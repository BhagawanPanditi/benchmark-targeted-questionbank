"""Shared constants for the pipeline stages.

SEVERITY_RANK is used by Stage 4 (merge/dedup of failure modes) and Stage 7
(dedup tie-breaking between near-duplicate questions).
"""

SEVERITY_RANK: dict[str, int] = {"critical": 3, "major": 2, "minor": 1}

VALID_FAILURE_TYPES_PASS_A: frozenset[str] = frozenset({
    "MISSING_PREREQUISITE",
    "WRONG_MENTAL_MODEL",
    "MISSING_TRICK_OR_INSIGHT",
    "COMMON_MISTAKE",
    "FALSE_ASSUMPTION",
    "MISREAD_CONSTRAINTS",
    "MISSING_DOMAIN_KNOWLEDGE",
    "SHORTCUT_ATTEMPT",
    "OVERCOUNTING_OR_UNDERCOUNTING",
    "INCOMPLETE_CASE_ANALYSIS",
    "UNJUSTIFIED_LOGICAL_STEP",
    "MUTABLE_STATE_OR_ALIASING",
    "TYPE_OR_PRECISION_ERROR",
    "OTHER",
})

VALID_FAILURE_TYPES_PASS_B: frozenset[str] = frozenset({
    "WRONG_PROBLEM_FRAME",
    "PLAUSIBLE_WRONG_ALGORITHM",
    "KNOWLEDGE_ILLUSION",
    "PATTERN_OVERFITTING",
    "COMPLEXITY_BLINDNESS",
    "PHANTOM_CONSTRAINT",
    "TERMINATION_ERROR",
    "REPRESENTATION_ERROR",
    "OTHER",
})

VALID_FAILURE_TYPES: frozenset[str] = VALID_FAILURE_TYPES_PASS_A | VALID_FAILURE_TYPES_PASS_B
