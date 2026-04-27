"""DPO fine-tuning with LoRA via Unsloth (falls back to standard transformers on CPU).

Requires a CUDA GPU for practical training. On CPU this will run but very slowly.
Unsloth is Linux/CUDA only; on macOS the fallback path is used automatically.
"""
from pathlib import Path

from datasets import Dataset
from loguru import logger

from deltaloop.config import settings


def _load_model_and_tokenizer():
    """Load base model with Unsloth if available, otherwise standard transformers."""
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-untyped]

        logger.info("dpo_trainer: loading model via Unsloth (4-bit quantized)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.1-8b-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=True,
        )
        return model, tokenizer

    except ImportError:
        logger.warning(
            "dpo_trainer: Unsloth not available (expected on macOS/CPU). "
            "Falling back to standard transformers + PEFT."
        )
        from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        model_name = "unsloth/llama-3.1-8b-bnb-4bit"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        lora_config = LoraConfig(
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer


async def run_dpo_training(
    dataset: Dataset,
    iteration: int,
    mlflow_tracker=None,
) -> str:
    """Run a DPO fine-tuning pass and save the resulting LoRA adapter.

    Args:
        dataset: HF Dataset with columns prompt/chosen/rejected.
        iteration: Current loop iteration number (used to name the adapter dir).
        mlflow_tracker: Optional MLflowTracker for logging training metrics.

    Returns:
        Path to the saved adapter directory.
    """
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-untyped]

    adapter_path = str(Path("adapters") / f"iteration_{iteration}")
    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"run_dpo_training: starting iteration={iteration} "
        f"dataset_size={len(dataset)} adapter_path={adapter_path}"
    )

    model, tokenizer = _load_model_and_tokenizer()

    config = DPOConfig(
        output_dir=adapter_path,
        num_train_epochs=settings.training_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",  # MLflow logging is handled manually below
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(adapter_path)

    logger.info(f"run_dpo_training: adapter saved to {adapter_path}")

    if mlflow_tracker is not None:
        mlflow_tracker.log_training_run(iteration, adapter_path, trainer.state.log_history)

    return adapter_path
