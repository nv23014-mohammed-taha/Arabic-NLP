# models/base.py
# ──────────────────────────────────────────────────────────────
# Abstract base class for all Arabic NLP model wrappers.
# Every model implements encode() and predict() so evaluate.py
# can call them through a uniform interface.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
import numpy as np


class ArabicNLPModel(ABC):
    """
    Base wrapper for Arabic NLP models.

    Subclasses must implement:
        - load()     : load tokenizer + model weights
        - predict()  : run inference on a list of texts
    """

    def __init__(self, model_name: str, task: str, device: str = "auto"):
        self.model_name = model_name
        self.task = task  # "sentiment" | "ner" | "qa"
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.tokenizer = None
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load tokenizer and model from HuggingFace Hub."""
        ...

    @abstractmethod
    def predict(self, texts: List[str], **kwargs) -> List[Any]:
        """
        Run inference.

        Args:
            texts: List of Arabic input strings.
            **kwargs: Task-specific arguments (e.g., questions for QA).

        Returns:
            List of predictions (labels, spans, etc.)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, task={self.task}, device={self.device})"
