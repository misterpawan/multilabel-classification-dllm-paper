"""dLLM-SetScore: Multi-label classification with discrete diffusion language models.

This package wraps the experimental code for the paper as a clean modular library.
The heavy implementation lives in `dllm_setscore.core`, which is imported lazily so that
unit tests and CLI dry runs do not pay the model-loading cost up front.
"""

from dllm_setscore.config import (
    GLOBAL_SEED,
    DRY_RUN,
    BACKBONE_REGISTRY,
    DATASET_DEFAULTS,
    SUPERVISED_MODEL_REGISTRY,
    VERBALIZER_CANDIDATES,
    PROMPT_INSTRUCTIONS,
    PathConfig,
    set_paths,
)

__all__ = [
    "GLOBAL_SEED",
    "DRY_RUN",
    "BACKBONE_REGISTRY",
    "DATASET_DEFAULTS",
    "SUPERVISED_MODEL_REGISTRY",
    "VERBALIZER_CANDIDATES",
    "PROMPT_INSTRUCTIONS",
    "PathConfig",
    "set_paths",
]
