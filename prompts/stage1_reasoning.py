"""Stage 1 prompts — gold-grounded reasoning trace generation.

PROMPT_CODING / PROMPT_REASONING produce a step-by-step expert trace that must
end with the exact line "Therefore, the answer is: <gold answer>"; the stage
verifies the stated answer against the gold answer before accepting the trace
(up to 3 attempts, then the record is kept with reasoning_status="failed").
"""
from string import Template

PROMPT_CODING = Template(r"""You are an expert software engineer and computer science educator.

You are given a coding problem and its reference solution. Write a step-by-step reasoning
trace that explains exactly how an expert would arrive at this solution from scratch.

Your trace must cover:
1. RESTATE: What is the problem actually asking? Restate the core task in your own words,
   stripping away flavor text and surface details.
2. CONSTRAINTS: What are the key constraints, input ranges, and edge cases that matter?
   Which constraints are load-bearing for the algorithm choice?
3. INSIGHT: What is the core algorithmic insight, trick, or pattern that unlocks this
   problem? This is the thing a failing solver would miss. Be specific — do not say
   "use dynamic programming," say what the recurrence is and why it holds.
4. APPROACH: Walk through the solution step by step. For each step, explain WHY this
   step, not just what it does. Connect every decision back to a constraint or insight.
5. EDGE CASES: Which edge cases does the solution handle, and how? Which edge cases
   would break a naive attempt?
6. FINAL: Conclude your derivation and state the final solution code on the last line in the format:
Therefore, the answer is: <solution>

Problem:
${question}

Reference Solution (for grounding):
${answer}

Write the reasoning trace now. Be specific to this problem, not generic.""")

PROMPT_REASONING = Template(r"""You are an expert mathematician and logician.

You are given a reasoning or math problem and its gold answer. Write a step-by-step
reasoning trace that proves and derives this answer from scratch.

Your trace must cover:
1. RESTATE: What is the problem asking? What quantity or object are we solving for?
   Restate precisely, stripping away narrative framing.
2. GIVEN: What information is provided? What are the constraints? Which ones are
   load-bearing for the solution?
3. INSIGHT: What is the key theorem, lemma, trick, or observation that makes this
   problem tractable? This is the thing a failing solver would miss or get wrong.
   Be specific — name the theorem, state the observation, explain why it applies here.
4. SOLUTION: Work through the solution completely. Show every logical step and explain
   WHY each step follows from the previous. Do not skip steps that seem obvious —
   obvious steps are often where wrong solvers make errors.
5. VERIFY: Does the answer make sense? Perform a sanity check: dimensional analysis,
   boundary case check, or substitution back into the original problem.
6. FINAL: Conclude your derivation and state the final result on the last line in the format:
Therefore, the answer is: <answer>

Problem:
${question}

Gold Answer (for grounding):
${answer}

Write the reasoning trace now. Be specific to this problem, not generic.""")
