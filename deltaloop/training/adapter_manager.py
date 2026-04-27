from loguru import logger


class AdapterManager:
    """Hot-swap LoRA adapters onto a base model without process restart.

    Loading the 4-bit quantized base model takes ~30-60 seconds.
    Swapping an adapter (small delta weights) takes <1 second.
    This class keeps the base model in memory and replaces only the adapter.
    """

    def __init__(self, base_model, tokenizer) -> None:
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._current_adapter: str | None = None
        self._model = None

    def swap(self, adapter_path: str) -> None:
        """Load a new LoRA adapter from disk, replacing the current one."""
        from peft import PeftModel  # type: ignore[import-untyped]

        logger.info(f"AdapterManager.swap: {self._current_adapter!r} → {adapter_path!r}")
        self._model = PeftModel.from_pretrained(self._base_model, adapter_path)
        self._current_adapter = adapter_path
        logger.info("AdapterManager.swap: complete")

    def get_model(self):
        """Return the current PEFT model. Raises if no adapter is loaded."""
        if self._model is None:
            raise RuntimeError("No adapter loaded. Call swap() first.")
        return self._model

    @property
    def current_adapter(self) -> str | None:
        return self._current_adapter
