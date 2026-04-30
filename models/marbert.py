# models/marbert.py
# ──────────────────────────────────────────────────────────────
# MARBERT wrapper (UBC NLP)
#
# MARBERT is trained exclusively on dialectal Arabic tweets,
# making it the go-to model for social-media NLP tasks.
# MARBERTv2 extends coverage with more data.
#
# Paper: https://arxiv.org/abs/2101.01785
# Models: https://huggingface.co/UBC-NLP
# ──────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Any

import torch
from transformers import AutoTokenizer, pipeline

from models.base import ArabicNLPModel


MARBERT_CHECKPOINTS = {
    "base":    "UBC-NLP/MARBERT",
    "v2":      "UBC-NLP/MARBERTv2",
    # Fine-tuned sentiment heads (community)
    "sentiment-v2": "UBC-NLP/MARBERTv2",   # we use the base; swap for fine-tuned ckpt if available
}


class MARBERTModel(ArabicNLPModel):
    """
    Wrapper around MARBERT / MARBERTv2 (UBC NLP).

    MARBERT is pre-trained on 1B tweets in dialectal Arabic,
    so it excels on social-media sentiment and dialect tasks.

    Args:
        version: "base" | "v2"  (v2 recommended)
        task: "sentiment" | "ner" | "qa"
    """

    def __init__(
        self,
        version: str = "v2",
        task: str = "sentiment",
        device: str = "auto",
    ):
        ckpt_key = version
        if ckpt_key not in MARBERT_CHECKPOINTS:
            raise ValueError(f"Unknown version '{version}'. Choose from: {list(MARBERT_CHECKPOINTS)}")
        super().__init__(
            model_name=MARBERT_CHECKPOINTS[ckpt_key],
            task=task,
            device=device,
        )
        self.version = version
        self.pipeline = None

    def load(self) -> None:
        print(f"  [MARBERTv{self.version}] Loading {self.model_name} …")
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
        print(f"  [MARBERTv{self.version}] ✓ Loaded ({self.device})")

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
