# models/multilingual_bert.py
# ──────────────────────────────────────────────────────────────
# Multilingual baselines: mBERT and XLM-RoBERTa
#
# These serve as cross-lingual baselines to contextualise how
# much Arabic-specific pre-training actually helps.
#
# Models:
#   - bert-base-multilingual-cased  (mBERT, Google, 104 langs)
#   - xlm-roberta-base              (XLM-R, Meta, 100 langs)
#   - xlm-roberta-large             (XLM-R Large — strong baseline)
# ──────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Any

import torch
from transformers import AutoTokenizer, pipeline

from models.base import ArabicNLPModel


MULTILINGUAL_CHECKPOINTS = {
    "mbert":      "bert-base-multilingual-cased",
    "xlmr-base":  "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
}


class MultilingualBERTModel(ArabicNLPModel):
    """
    Wrapper around multilingual baselines (mBERT, XLM-R).

    Use these as upper/lower bounds to compare against
    Arabic-specific models.

    Args:
        variant: "mbert" | "xlmr-base" | "xlmr-large"
        task: "sentiment" | "ner" | "qa"
    """

    def __init__(
        self,
        variant: str = "xlmr-base",
        task: str = "sentiment",
        device: str = "auto",
    ):
        if variant not in MULTILINGUAL_CHECKPOINTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(MULTILINGUAL_CHECKPOINTS)}")
        super().__init__(
            model_name=MULTILINGUAL_CHECKPOINTS[variant],
            task=task,
            device=device,
        )
        self.variant = variant
        self.pipeline = None

    def load(self) -> None:
        print(f"  [{self.variant.upper()}] Loading {self.model_name} …")
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
        print(f"  [{self.variant.upper()}] ✓ Loaded ({self.device})")

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
