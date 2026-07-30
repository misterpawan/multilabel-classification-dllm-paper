"""Centralised configuration: dataset defaults, prompts, backbone registry, paths.

The notebook used module-level globals; we keep the same names for compatibility but
let CLI flags override them at startup via :func:`set_paths` / env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GLOBAL_SEED = int(os.environ.get("DLLM_SEED", 13))
DRY_RUN = os.environ.get("DLLM_DRY_RUN", "0") == "1"

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

REUTERS_TOPIC_TEXT = {
    "earn": "earnings",
    "acq": "acquisitions",
    "money-fx": "foreign exchange",
    "grain": "grain",
    "crude": "crude oil",
    "trade": "trade",
    "interest": "interest rates",
    "ship": "shipping",
    "wheat": "wheat",
    "corn": "corn",
    "dlr": "US dollar",
    "money-supply": "money supply",
    "coffee": "coffee",
    "sugar": "sugar",
    "oilseed": "oilseeds",
    "gold": "gold",
    "nat-gas": "natural gas",
    "cpi": "consumer price index",
    "veg-oil": "vegetable oil",
    "cocoa": "cocoa",
    "reserves": "foreign reserves",
    "gnp": "gross national product",
    "livestock": "livestock",
    "strategic-metal": "strategic metals",
    "iron-steel": "iron and steel",
    "meal-feed": "meal and feed",
}

VERBALIZER_CANDIDATES = [
    (" yes", " no"),
    (" true", " false"),
    (" relevant", " irrelevant"),
    (" present", " absent"),
]

PROMPT_INSTRUCTIONS = {
    "default": "Decide whether each candidate label applies to the document. Use exactly one token for each label, in the same order as listed.",
    "concise": "For each candidate label, answer whether it applies to the document. Keep the label order unchanged.",
    "analysis": "Read the document, inspect every candidate label, and answer with one token per label in order.",
}

DATASET_DEFAULTS = {
    "goemotions": {
        "hypothesis_template": "This text expresses {}.",
        "prompt_max_context_tokens": 512,
        "encoder_max_length": 128,
        "bart_max_length": 256,
        "t5_max_source_length": 128,
        "t5_max_target_length": 48,
        "fewshot_shots_per_label": 8,
        "shortlist_k": None,
    },
    "reuters21578_top20": {
        "hypothesis_template": "This news article is about {}.",
        "prompt_max_context_tokens": 768,
        "encoder_max_length": 256,
        "bart_max_length": 256,
        "t5_max_source_length": 256,
        "t5_max_target_length": 48,
        "fewshot_shots_per_label": 8,
        "shortlist_k": None,
    },
    "eurlex57k": {
        "hypothesis_template": "This legal document is about {}.",
        "prompt_max_context_tokens": 1024,
        "encoder_max_length": 512,
        "bart_max_length": 512,
        "t5_max_source_length": 512,
        "t5_max_target_length": 96,
        "fewshot_shots_per_label": 4,
        "shortlist_k": 32,
    },
    "multieurlex_en": {
        "hypothesis_template": "This legal document is about {}.",
        "prompt_max_context_tokens": 1024,
        "encoder_max_length": 512,
        "bart_max_length": 512,
        "t5_max_source_length": 512,
        "t5_max_target_length": 96,
        "fewshot_shots_per_label": 4,
        "shortlist_k": 32,
    },
}

BACKBONE_REGISTRY = {
    "mdlm": {
        "model_id": "kuleshov-group/mdlm-owt",
        "tokenizer_id": "gpt2",
        "loader": "maskedlm",
        "trust_remote_code": True,
        "default_dtype": "float32",
        "max_context_tokens": 1024,
    },
    "llada": {
        "model_id": "GSAI-ML/LLaDA-8B-Base",
        "tokenizer_id": "GSAI-ML/LLaDA-8B-Base",
        "loader": "auto",
        "trust_remote_code": True,
        "default_dtype": "bfloat16",
        "max_context_tokens": 1024,
    },
}

SUPERVISED_MODEL_REGISTRY = {
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "t5_base": "google/flan-t5-base",
    "bart_mnli": "facebook/bart-large-mnli",
    "setfit_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "retriever": "sentence-transformers/all-MiniLM-L6-v2",
}


@dataclass
class PathConfig:
    root: Path
    cache: Path
    results: Path
    models: Path
    plots: Path
    tables: Path
    exports: Path

    @classmethod
    def from_root(cls, root: Path) -> "PathConfig":
        root = Path(root)
        return cls(
            root=root,
            cache=root / "cache",
            results=root / "results",
            models=root / "saved_models",
            plots=root / "plots",
            tables=root / "tables",
            exports=root / "exports",
        )

    def ensure(self) -> None:
        for path in [self.root, self.cache, self.results, self.models, self.plots, self.tables, self.exports]:
            path.mkdir(parents=True, exist_ok=True)


_DEFAULT_ROOT = Path(os.environ.get("DLLM_RUN_ROOT", Path.cwd() / "runs"))
PATHS = PathConfig.from_root(_DEFAULT_ROOT)
PATHS.ensure()


def set_paths(root: Optional[Path] = None) -> PathConfig:
    """Re-point all artifact directories. Used by the CLI."""
    global PATHS
    if root is not None:
        PATHS = PathConfig.from_root(Path(root))
        PATHS.ensure()
    return PATHS
