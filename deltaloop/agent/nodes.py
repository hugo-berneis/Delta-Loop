"""LangGraph node functions. Each takes AgentState, returns partial AgentState update."""
import base64
import json
from pathlib import Path

from loguru import logger

from deltaloop.agent.ollama_client import get_ollama_client
from deltaloop.agent.prompts import (
    MULTIMODAL_PROMPT,
    PLAN_PROMPT,
    PLAN_PROMPT_STRICT,
    REASON_PROMPT,
    REASON_PROMPT_STRICT,
    SYNTHESIZE_PROMPT,
)
from deltaloop.agent.state import AgentState
from deltaloop.agent.tools import AVAILABLE_TOOLS
from deltaloop.config import settings

_REFUSAL_PHRASES = (
    "i cannot", "i can't", "i don't know", "i am unable",
    "i'm unable", "no information", "not able to",
)


def _parse_json_response(raw: str) -> dict | None:
    """Try to extract a JSON object from an LLM response."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

async def plan(state: AgentState) -> dict:
    task = state["task"]
    client = get_ollama_client()

    prompt = PLAN_PROMPT.format(
        question=task.question, context_type=task.context_type
    )
    with logger.contextualize(node="plan", task_id=task.id):
        raw = await client.complete(
            settings.agent_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        parsed = _parse_json_response(raw)
        if parsed is None:
            logger.warning("plan: malformed JSON, retrying with strict prompt")
            raw = await client.complete(
                settings.agent_model,
                [{"role": "user", "content": PLAN_PROMPT_STRICT.format(
                    question=task.question, context_type=task.context_type
                )}],
                temperature=0.0,
            )
            parsed = _parse_json_response(raw)

        if parsed and "steps" in parsed:
            steps = parsed["steps"]
            plan_text = " | ".join(
                f"Step {s['step']}: {s['action']}" for s in steps
            )
        else:
            plan_text = f"Plan: analyze {task.context_type} content and answer the question"

        logger.info(f"plan complete: {len(plan_text)} chars")
        return {"reasoning_steps": [plan_text]}


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------

async def retrieve_context(state: AgentState) -> dict:
    task = state["task"]
    context_path = task.context_path

    with logger.contextualize(node="retrieve_context", task_id=task.id):
        if not context_path or not Path(context_path).exists():
            logger.warning(f"retrieve_context: file not found at {context_path}")
            return {"retrieved_context": ""}

        path = Path(context_path)
        if path.suffix.lower() == ".pdf":
            try:
                import pdfplumber  # type: ignore[import-untyped]
                with pdfplumber.open(str(path)) as pdf:
                    text = "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
            except Exception as exc:
                logger.error(f"retrieve_context: pdfplumber failed: {exc}")
                text = ""
        else:
            text = path.read_text(encoding="utf-8")

        logger.info(f"retrieve_context: extracted {len(text)} chars from {path.name}")
        return {"retrieved_context": text[:8000]}  # cap to avoid token overflow


# ---------------------------------------------------------------------------
# analyze_multimodal
# ---------------------------------------------------------------------------

async def analyze_multimodal(state: AgentState) -> dict:
    task = state["task"]
    context_path = task.context_path
    client = get_ollama_client()

    with logger.contextualize(node="analyze_multimodal", task_id=task.id):
        if not context_path or not Path(context_path).exists():
            logger.warning(f"analyze_multimodal: image not found at {context_path}")
            return {"multimodal_output": "Image not available."}

        image_data = Path(context_path).read_bytes()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        prompt = MULTIMODAL_PROMPT.format(question=task.question)
        result = await client.complete_multimodal(
            model=settings.multimodal_model,
            messages=[{"role": "user", "content": prompt}],
            images=[image_b64],
        )
        logger.info(f"analyze_multimodal: got {len(result)} chars")
        return {"multimodal_output": result}


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------

async def reason(state: AgentState) -> dict:
    task = state["task"]
    client = get_ollama_client()
    steps = state.get("reasoning_steps", [])

    with logger.contextualize(node="reason", task_id=task.id):
        prompt = REASON_PROMPT.format(
            question=task.question,
            context_type=task.context_type,
            reasoning_steps="\n".join(f"- {s}" for s in steps) or "(none yet)",
            retrieved_context=state.get("retrieved_context", "") or "(none)",
            multimodal_output=state.get("multimodal_output", "") or "(none)",
        )

        raw = await client.complete(
            settings.agent_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        parsed = _parse_json_response(raw)
        if parsed is None:
            logger.warning("reason: malformed JSON, retrying with strict prompt")
            raw = await client.complete(
                settings.agent_model,
                [{"role": "user", "content": REASON_PROMPT_STRICT.format(
                    question=task.question,
                    reasoning_steps="\n".join(steps) or "(none)",
                )}],
                temperature=0.0,
            )
            parsed = _parse_json_response(raw)

        if parsed is None:
            logger.error("reason: JSON parsing failed twice, forcing synthesize")
            return {
                "reasoning_steps": [*steps, "Reasoning step failed — proceeding to synthesize."],
                "tool_calls": [*state.get("tool_calls", []), {"ready": True}],
            }

        next_step = parsed.get("next_step", "Continuing analysis.")
        needs_tool = parsed.get("needs_tool", False)
        tool_name = parsed.get("tool_name")
        tool_args = parsed.get("tool_args", {})
        ready = parsed.get("ready_to_answer", False)

        updated_steps = [*steps, next_step]
        updated_calls = list(state.get("tool_calls", []))

        if needs_tool and tool_name in AVAILABLE_TOOLS:
            updated_calls.append({"name": tool_name, "args": tool_args, "pending": True})
            logger.info(f"reason: requesting tool '{tool_name}'")
        else:
            updated_calls.append({"ready": True})
            logger.info(f"reason: ready_to_answer={ready}")

        return {"reasoning_steps": updated_steps, "tool_calls": updated_calls}


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------

async def call_tool(state: AgentState) -> dict:
    task = state["task"]
    tool_calls = state.get("tool_calls", [])

    with logger.contextualize(node="call_tool", task_id=task.id):
        pending = next(
            (tc for tc in reversed(tool_calls) if tc.get("pending")), None
        )
        if pending is None:
            logger.warning("call_tool: no pending tool call found")
            return {}

        name = pending["name"]
        args = pending.get("args", {})
        tool_fn = AVAILABLE_TOOLS.get(name)

        if tool_fn is None:
            logger.error(f"call_tool: unknown tool '{name}'")
            result = f"Tool '{name}' is not available."
        else:
            try:
                result = await tool_fn(state, args)
                logger.info(f"call_tool: '{name}' returned {len(result)} chars")
            except Exception as exc:
                logger.error(f"call_tool: '{name}' raised {exc}")
                result = f"Tool '{name}' failed: {exc}"

        # Mark the call as completed and store result
        updated_calls = [
            {**tc, "pending": False, "result": result} if tc is pending else tc
            for tc in tool_calls
        ]

        # Route result to the appropriate state field
        if name == "describe_image":
            return {"tool_calls": updated_calls, "multimodal_output": result}
        return {"tool_calls": updated_calls, "retrieved_context": result}


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

async def synthesize(state: AgentState) -> dict:
    task = state["task"]
    client = get_ollama_client()
    steps = state.get("reasoning_steps", [])

    with logger.contextualize(node="synthesize", task_id=task.id):
        prompt = SYNTHESIZE_PROMPT.format(
            question=task.question,
            reasoning_steps="\n".join(f"- {s}" for s in steps) or "(none)",
            retrieved_context=state.get("retrieved_context", "") or "(none)",
            multimodal_output=state.get("multimodal_output", "") or "(none)",
        )

        answer = await client.complete(
            settings.agent_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        logger.info(f"synthesize: answer length={len(answer)}")
        return {"final_answer": answer.strip()}


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

async def validate(state: AgentState) -> dict:
    answer = state.get("final_answer", "").strip()
    task = state["task"]

    with logger.contextualize(node="validate", task_id=task.id):
        if not answer:
            logger.warning("validate: empty final_answer")
            return {"is_complete": False, "error": "Empty final answer"}

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in _REFUSAL_PHRASES):
            logger.warning(f"validate: refusal detected in answer: {answer[:80]}")
            return {"is_complete": False, "error": f"Refusal detected: {answer[:80]}"}

        logger.info("validate: answer accepted")
        return {"is_complete": True, "error": None}
