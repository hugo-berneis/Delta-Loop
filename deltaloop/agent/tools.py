"""Tool definitions — each is an async function (state, args) -> str."""
import base64
import re
from html.parser import HTMLParser
from pathlib import Path

from loguru import logger

from deltaloop.agent.state import AgentState


# ---------------------------------------------------------------------------
# HTML text extractor (stdlib only — no extra deps)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_tags = {"script", "style", "head"}
        self._current_skip: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._skip_tags:
            self._current_skip = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._current_skip:
            self._current_skip = None

    def handle_data(self, data: str) -> None:
        if self._current_skip is None:
            stripped = data.strip()
            if stripped:
                self._chunks.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _extract_html_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# search_document
# ---------------------------------------------------------------------------

async def search_document(state: AgentState, args: dict) -> str:
    """Keyword search over retrieved_context."""
    query: str = args.get("query", "")
    context: str = state.get("retrieved_context", "")

    if not context:
        return "No document context available to search."
    if not query:
        return context[:2000]

    query_lower = query.lower()
    sentences = re.split(r"(?<=[.!?])\s+", context)
    matches = [s for s in sentences if query_lower in s.lower()]

    if matches:
        result = " ".join(matches[:10])
        logger.debug(f"search_document query='{query}' found {len(matches)} sentences")
        return result

    logger.debug(f"search_document query='{query}' no exact matches, returning excerpt")
    return context[:2000]


# ---------------------------------------------------------------------------
# describe_image
# ---------------------------------------------------------------------------

async def describe_image(state: AgentState, args: dict) -> str:
    """Re-call llava with a focused prompt on the image."""
    from deltaloop.agent.ollama_client import get_ollama_client
    from deltaloop.agent.prompts import DESCRIBE_IMAGE_PROMPT
    from deltaloop.config import settings

    focus: str = args.get("focus", "the main content")
    task = state["task"]
    context_path = task.context_path

    if not context_path or not Path(context_path).exists():
        logger.warning(f"describe_image: image not found at {context_path}")
        return f"Image not available at {context_path}"

    image_data = Path(context_path).read_bytes()
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    prompt = DESCRIBE_IMAGE_PROMPT.format(focus=focus, question=task.question)
    client = get_ollama_client()
    result = await client.complete_multimodal(
        model=settings.multimodal_model,
        messages=[{"role": "user", "content": prompt}],
        images=[image_b64],
    )
    logger.debug(f"describe_image focus='{focus}' response_len={len(result)}")
    return result


# ---------------------------------------------------------------------------
# navigate_page
# ---------------------------------------------------------------------------

async def navigate_page(state: AgentState, args: dict) -> str:
    """Extract text content from a static HTML snapshot."""
    task = state["task"]
    context_path = task.context_path

    selector: str = args.get("selector", "")  # optional CSS-like hint (unused, for future use)

    if not context_path or not Path(context_path).exists():
        logger.warning(f"navigate_page: HTML file not found at {context_path}")
        return f"Page not available at {context_path}"

    html = Path(context_path).read_text(encoding="utf-8")
    text = _extract_html_text(html)
    logger.debug(f"navigate_page extracted {len(text)} chars from {context_path}")
    return text[:4000]  # cap to avoid overwhelming context


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: dict = {
    "search_document": search_document,
    "describe_image": describe_image,
    "navigate_page": navigate_page,
}
