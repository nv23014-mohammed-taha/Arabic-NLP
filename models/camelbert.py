# models/camelbert.py
# ──────────────────────────────────────────────────────────────
# CAMeLBERT wrapper (CAMeL Lab, NYU Abu Dhabi)
#
# CAMeLBERT is a family of BERT models trained on different
# Arabic varieties: MSA, CA, DA, and mixed. This is particularly
# relevant to MBZUAI given the NYU Abu Dhabi connection.
#
# Paper: https://arxiv.org/abs/2103.06678
# Models: https://huggingface.co/CAMeL-Lab
# ──────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Any, Literal

import torch
from transformers import AutoTokenizer, pipeline

from models.base import ArabicNLPModel


# CAMeLBERT has multiple variants — we expose the most useful ones
CAMELBERT_VARIANTS = {
    "msa":   "CAMeL-Lab/bert-base-arabic-camelbert-msa",          # Modern Standard Arabic
    "ca":    "CAMeL-Lab/bert-base-arabic-camelbert-ca",            # Classical Arabic
    "da":    "CAMeL-Lab/bert-base-arabic-camelbert-da",            # Dialectal Arabic
    "mixed": "CAMeL-Lab/bert-base-arabic-camelbert-mix",           # Mixed
    "msa-sentiment": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment",
    "da-sentiment":  "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
}


class CAMeLBERTModel(ArabicNLPModel):
    """
    Wrapper around CAMeLBERT (CAMeL Lab, NYU Abu Dhabi).

    Supports MSA, CA, DA and mixed variants. For sentiment tasks,
    dedicated fine-tuned checkpoints are available and preferred.

    Args:
        variant: One of {"msa", "ca", "da", "mixed",
                         "msa-sentiment", "da-sentiment"}.
        task: "sentiment" | "ner" | "qa"
    """

    def __init__(
        self,
        variant: str = "msa-sentiment",
        task: str = "sentiment",
        device: str = "auto",
    ):
        if variant not in CAMELBERT_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(CAMELBERT_VARIANTS)}")
        super().__init__(
            model_name=CAMELBERT_VARIANTS[variant],
            task=task,
            device=device,
        )
        self.variant = variant
        self.pipeline = None

    def load(self) -> None:
        print(f"  [CAMeLBERT-{self.variant}] Loading {self.model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        task_map = {
            "sentiment": "text-classification",
            "ner":       "ner",
            "qa":        "question-answering",
        }
        hf_task = task_map[self.task]

        kwargs = dict(
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
            truncation=True,
            max_length=512,
        )
        if self.task == "ner":
            kwargs["aggregation_strategy"] = "simple"

        self.pipeline = pipeline(hf_task, **kwargs)
        print(f"  [CAMeLBERT-{self.variant}] ✓ Loaded ({self.device})")

    def predict(self, texts: List[str], **kwargs) -> List[Any]:
        if self.pipeline is None:
            raise RuntimeError("Call .load() before .predict()")

        if self.task == "qa":
            questions = kwargs["questions"]
            return [
                self.pipeline({"question": q, "context": c})
                for q, c in zip(questions, texts)
            ]

        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.pipeline(batch))
        return results
