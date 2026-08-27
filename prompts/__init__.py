"""All LLM prompt templates, in one place.

One module per pipeline stage; the stage logic (``stages/``) imports its
prompts from here so no prompt text lives inside the stage code. Every
template is a ``string.Template`` — fill it with ``safe_substitute`` and send
the result through ``utils.llm.call_llm``.
"""
from prompts import (
    stage1_reasoning,
    stage2_raw_concepts,
    stage3_taxonomy,
    stage4_failure_modes,
    stage5_concept_graph,
    stage6_question_gen,
    stage7_validation,
)

__all__ = [
    "stage1_reasoning",
    "stage2_raw_concepts",
    "stage3_taxonomy",
    "stage4_failure_modes",
    "stage5_concept_graph",
    "stage6_question_gen",
    "stage7_validation",
]
