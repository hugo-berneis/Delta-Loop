EVAL_PROMPT = """You are a rigorous AI evaluator. You will be given a task, a ground truth answer, and an agent's reasoning trace and final answer.

Your job is to:
1. Determine if the agent's final answer is correct given the ground truth
2. If incorrect, identify the failure mode from this taxonomy:
   - WRONG_RETRIEVAL: Agent retrieved irrelevant or no context
   - WRONG_REASONING: Agent retrieved correct context but reasoned incorrectly
   - INCOMPLETE_ANSWER: Agent stopped reasoning too early
   - HALLUCINATION: Agent stated facts not present in context
   - TOOL_MISUSE: Agent called the wrong tool or with wrong arguments

Respond ONLY with valid JSON matching this schema:
{{
  "is_correct": boolean,
  "score": float between 0.0 and 1.0,
  "failure_mode": string from taxonomy above or null if correct,
  "failure_explanation": string explaining the failure in one sentence or null if correct
}}

TASK: {task_question}
GROUND TRUTH: {ground_truth}
AGENT REASONING STEPS: {reasoning_steps}
AGENT FINAL ANSWER: {final_answer}"""

EVAL_PROMPT_STRICT = """IMPORTANT: Your previous response was not valid JSON. You MUST respond with ONLY a JSON object and nothing else — no explanation, no markdown, no code fences.

The JSON must match exactly:
{{"is_correct": bool, "score": float, "failure_mode": str_or_null, "failure_explanation": str_or_null}}

TASK: {task_question}
GROUND TRUTH: {ground_truth}
AGENT REASONING STEPS: {reasoning_steps}
AGENT FINAL ANSWER: {final_answer}"""

CORRECTION_PROMPT = """You are an expert AI reasoning corrector. An agent attempted a task and failed.

Your job is to rewrite the agent's reasoning steps from scratch, correcting the specific failure.

FAILURE MODE: {failure_mode}
FAILURE EXPLANATION: {failure_explanation}

The corrected trace must:
1. Use the same tools and context that were available to the original agent
2. Fix the specific failure mode identified above
3. Lead to the correct answer: {ground_truth}
4. Be valid JSON: a list of strings, each string being one reasoning step

Respond ONLY with valid JSON: a list of strings.
Example: ["Step 1: ...", "Step 2: ...", "Step 3: ..."]

TASK: {task_question}
ORIGINAL REASONING STEPS: {reasoning_steps}
ORIGINAL FINAL ANSWER: {final_answer}"""

CORRECTION_PROMPT_STRICT = """IMPORTANT: Your previous response was not valid JSON. Respond ONLY with a JSON array of strings — no explanation, no markdown.

Example: ["Step 1: ...", "Step 2: ..."]

TASK: {task_question}
FAILURE MODE: {failure_mode}
CORRECT ANSWER: {ground_truth}"""
