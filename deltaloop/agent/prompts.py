PLAN_PROMPT = """You are a methodical AI agent. Given a task, create a concise step-by-step plan.

TASK: {question}
CONTEXT TYPE: {context_type}
AVAILABLE TOOLS: search_document, describe_image, navigate_page

Respond ONLY with valid JSON matching this schema:
{{"steps": [
  {{"step": 1, "action": "what you will do", "rationale": "why"}},
  {{"step": 2, "action": "what you will do", "rationale": "why"}}
]}}"""

PLAN_PROMPT_STRICT = """IMPORTANT: Your previous response was not valid JSON. Respond ONLY with a JSON object, no other text.

Schema: {{"steps": [{{"step": 1, "action": "...", "rationale": "..."}}]}}

TASK: {question}
CONTEXT TYPE: {context_type}"""

REASON_PROMPT = """You are a reasoning AI agent working through a task step by step.

TASK: {question}
CONTEXT TYPE: {context_type}

REASONING SO FAR:
{reasoning_steps}

RETRIEVED CONTEXT:
{retrieved_context}

MULTIMODAL OUTPUT:
{multimodal_output}

TOOLS ALREADY CALLED (do NOT repeat a tool that already returned an error):
{tool_history}

AVAILABLE TOOLS: search_document, describe_image, navigate_page

Decide your next action. Respond ONLY with valid JSON:
{{
  "next_step": "description of this reasoning step",
  "needs_tool": true or false,
  "tool_name": "tool name or null",
  "tool_args": {{}},
  "ready_to_answer": true or false
}}

Set ready_to_answer to true only when you have enough information to give a final answer.
If a tool already failed or returned an error, do not call it again — set ready_to_answer to true instead."""

REASON_PROMPT_STRICT = """IMPORTANT: Respond ONLY with a JSON object, no other text.

Schema: {{"next_step": "...", "needs_tool": bool, "tool_name": str_or_null, "tool_args": {{}}, "ready_to_answer": bool}}

TASK: {question}
REASONING SO FAR: {reasoning_steps}"""

SYNTHESIZE_PROMPT = """You are an AI agent. Provide a final answer based on your reasoning.

TASK: {question}

REASONING STEPS:
{reasoning_steps}

RETRIEVED CONTEXT:
{retrieved_context}

MULTIMODAL OUTPUT:
{multimodal_output}

Give a clear, direct answer to the task. Answer only — no preamble."""

MULTIMODAL_PROMPT = """Carefully examine this image and answer the following question.

QUESTION: {question}

Describe what you observe and provide a specific, factual answer."""

DESCRIBE_IMAGE_PROMPT = """Re-examine this image with focus on the following aspect.

FOCUS: {focus}
ORIGINAL QUESTION: {question}

Provide a detailed description relevant to the focus area."""
