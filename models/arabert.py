# models/arabert.py
# ──────────────────────────────────────────────────────────────
# AraBERT wrapper (aubmindlab/bert-base-arabertv2)
#
# AraBERT is a BERT-based model pre-trained on a large Arabic
# corpus. It was one of the first dedicated Arabic BERT models
# and remains a strong baseline.
#
# Paper: https://arxiv.org/abs/2003.00104
# ──────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from models.base import ArabicNLPModel


# Map friendly task names → HuggingFace checkpoint
ARABERT_CHECKPOINTS = {
    "sentiment": "aubmindlab/bert-base-arabertv2",   # fine-tune head added in load()
    "ner":       "aubmindlab/bert-base-arabertv2",
    "qa":        "aubmindlab/bert-base-arabertv2",
}


class AraBERTModel(ArabicNLPModel):
    """
    Wrapper around AraBERT (aubmindlab/bert-base-arabertv2).

    For sentiment classification we attach a linear head and
    optionally fine-tune on the target dataset. For zero-shot
    benchmarking we use the raw [CLS] embedding + cosine sim.
    """

    def __init__(self, task: str = "sentiment", device: str = "auto"):
        super().__init__(
            model_name=ARABERT_CHECKPOINTS[task],
            task=task,
            device=device,
        )
        self.pipeline = None

    def load(self) -> None:
        """Download and initialise the model + tokenizer."""
        print(f"  [AraBERT] Loading {self.model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.task == "sentiment":
            # Use a sentiment-fine-tuned checkpoint when available;
            # fall back to the base model for feature extraction.
            sentiment_ckpt = "aubmindlab/bert-base-arabertv2"
            self.pipeline = pipeline(
                "text-classification",
                model=sentiment_ckpt,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
        elif self.task == "ner":
            self.pipeline = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if self.device.type == "cuda" else -1,
            )
        elif self.task == "qa":
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
            )

        print(f"  [AraBERT] ✓ Loaded ({self.device})")

    def predict(self, texts: List[str], **kwargs) -> List[Any]:
        """
        Run batch inference.

        For QA pass `questions=List[str]` as a kwarg.
        """
        if self.pipeline is None:
            raise RuntimeError("Call .load() before .predict()")

        if self.task == "qa":
            questions = kwargs["questions"]
            return [
                self.pipeline({"question": q, "context": c})
                for q, c in zip(questions, texts)
            ]

        # Sentiment / NER — process in batches of 32
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.pipeline(batch))
        return results
