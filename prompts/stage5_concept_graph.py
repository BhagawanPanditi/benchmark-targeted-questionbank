"""Stage 5 prompt — direct-prerequisite elicitation (once per concept).

PROMPT_PREREQ returns the direct prerequisites (max 5) of one concept, plus an
is_leaf flag and is_new flags for prerequisites missing from the vocabulary.
"""
from string import Template

PROMPT_PREREQ = Template(r"""You are a ${domain} curriculum expert building a prerequisite dependency graph.

Target concept: "${concept}"

What concepts must a learner ALREADY understand before they can properly learn
or apply "${concept}"?

Rules:
  - List only DIRECT prerequisites — concepts one step back in the dependency chain.
    Do NOT list transitive prerequisites (those will be found by traversing the graph).
  - Prefer concepts from this known vocabulary (use exact strings where possible):
    ${all_concepts}
  - If a genuine direct prerequisite is missing from the vocabulary entirely, add it
    in the same dot-notation format and mark it new.
  - If this concept is ATOMIC — meaning it has no meaningful prerequisites in a
    ${domain} learning context, it is a true starting point — return an empty list
    and set is_leaf=true.
  - Maximum 5 prerequisites.
  - Be conservative: only list concepts that are genuinely REQUIRED to understand
    "${concept}", not merely helpful or related.

${examples}

Return ONLY valid JSON:
{
  "concept": "${concept}",
  "is_leaf": true | false,
  "prerequisites": [
    {"name": "exact.concept.string", "is_new": false},
    ...
  ]
}""")
