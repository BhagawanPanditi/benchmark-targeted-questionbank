"""Shared constants for the pipeline stages.

SEVERITY_RANK is used by Stage 4 (merge/dedup of failure modes) and Stage 7
(dedup tie-breaking between near-duplicate questions).
"""

SEVERITY_RANK: dict[str, int] = {"critical": 3, "major": 2, "minor": 1}
