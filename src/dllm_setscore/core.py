"""dLLM-SetScore core implementation.

This module is the modular form of `codes/dllm_setscore_notebook_script.py`.
The package-level CLI (``python -m dllm_setscore``) configures globals via
:func:`configure_runtime` and then calls the orchestration helpers exposed below
(``run_main_comparison_for_bundle``, ``run_supervised_comparison_for_bundle``,
``run_ablation_suite_for_bundle``, ``run_all_experiments``).
"""

import gc
import html
import io
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download, list_repo_files
try:
    from IPython.display import display  # noqa: F401  (notebook usage only)
except Exception:  # pragma: no cover
    def display(*args, **kwargs):
        pass
from matplotlib import pyplot as plt
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except Exception:
    MultilabelStratifiedShuffleSplit = None

try:
    from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
except Exception:
    Dataset = None
    DatasetDict = None
    get_dataset_config_names = None
    load_dataset = None

try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
except Exception:
    InputExample = None
    SentenceTransformer = None
    losses = None

try:
    from torch.utils.data import DataLoader
except Exception:
    DataLoader = None

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

# Artifact paths can be redirected by configure_runtime() at process start.
ROOT = Path(os.environ.get("DLLM_RUN_ROOT", str(Path.cwd() / "runs")))
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "saved_models"
PLOTS_DIR = ROOT / "plots"
TABLES_DIR = ROOT / "tables"
EXPORT_DIR = ROOT / "exports"

for path in [ROOT, CACHE_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, TABLES_DIR, EXPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = int(os.environ.get("DLLM_SEED", 13))
DRY_RUN = os.environ.get("DLLM_DRY_RUN", "0") == "1"
OVERWRITE_CACHE = os.environ.get("DLLM_OVERWRITE", "0") == "1"
RUN_SMOKE_TESTS = False
RUN_MAIN_EXPERIMENTS = True
RUN_SUPERVISED_EXPERIMENTS = True
RUN_ABLATIONS = True
RUN_OPTIONAL_EXTENSIONS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.set_float32_matmul_precision("high")


def configure_runtime(
    root: Optional[Path] = None,
    *,
    dry_run: Optional[bool] = None,
    overwrite_cache: Optional[bool] = None,
    seed: Optional[int] = None,
    run_main: Optional[bool] = None,
    run_supervised: Optional[bool] = None,
    run_ablations: Optional[bool] = None,
    run_optional: Optional[bool] = None,
) -> None:
    """Re-point artifact directories and toggle pipeline switches at process start.

    Must be called before invoking any of the orchestration functions because
    several module-level paths are baked in for cache lookup.
    """
    global ROOT, CACHE_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, TABLES_DIR, EXPORT_DIR
    global DRY_RUN, OVERWRITE_CACHE, GLOBAL_SEED
    global RUN_MAIN_EXPERIMENTS, RUN_SUPERVISED_EXPERIMENTS, RUN_ABLATIONS, RUN_OPTIONAL_EXTENSIONS

    if root is not None:
        ROOT = Path(root)
        CACHE_DIR = ROOT / "cache"
        RESULTS_DIR = ROOT / "results"
        MODELS_DIR = ROOT / "saved_models"
        PLOTS_DIR = ROOT / "plots"
        TABLES_DIR = ROOT / "tables"
        EXPORT_DIR = ROOT / "exports"
        for path in [ROOT, CACHE_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, TABLES_DIR, EXPORT_DIR]:
            path.mkdir(parents=True, exist_ok=True)
    if dry_run is not None:
        DRY_RUN = bool(dry_run)
    if overwrite_cache is not None:
        OVERWRITE_CACHE = bool(overwrite_cache)
    if seed is not None:
        GLOBAL_SEED = int(seed)
        set_seed(GLOBAL_SEED)
    if run_main is not None:
        RUN_MAIN_EXPERIMENTS = bool(run_main)
    if run_supervised is not None:
        RUN_SUPERVISED_EXPERIMENTS = bool(run_supervised)
    if run_ablations is not None:
        RUN_ABLATIONS = bool(run_ablations)
    if run_optional is not None:
        RUN_OPTIONAL_EXTENSIONS = bool(run_optional)

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
    # ECtHR (European Court of Human Rights) Task A from LexGLUE.
    # 10 high-level articles of the Convention, ~9k/1k/1k split, long legal
    # case facts. Complements EURLEX57K in the long-doc / low-cardinality
    # corner of the design space.
    "ecthr_a": {
        "hypothesis_template": "This case concerns {}.",
        "prompt_max_context_tokens": 1024,
        "encoder_max_length": 512,
        "bart_max_length": 512,
        "t5_max_source_length": 512,
        "t5_max_target_length": 64,
        "fewshot_shots_per_label": 8,
        "shortlist_k": None,
    },
    # Jigsaw Toxic Comment Classification — 6 labels, very short social text,
    # very different label correlation structure from the rest of our suite.
    "jigsaw_toxic": {
        "hypothesis_template": "This comment is {}.",
        "prompt_max_context_tokens": 384,
        "encoder_max_length": 192,
        "bart_max_length": 192,
        "t5_max_source_length": 192,
        "t5_max_target_length": 32,
        "fewshot_shots_per_label": 8,
        "shortlist_k": None,
    },
    "aapd": {
        "hypothesis_template": "This paper is about {}.",
        "prompt_max_context_tokens": 768,
        "encoder_max_length": 512,
        "bart_max_length": 512,
        "t5_max_source_length": 512,
        "t5_max_target_length": 96,
        "fewshot_shots_per_label": 4,
        "shortlist_k": 54,
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
        "default_dtype": "bfloat16" if BF16_OK else "float32",
        # LLaDA-8B's max_sequence_length is 4096; we keep 2048 as a safe default
        # so EURLEX57K (long documents + 100-label list) fits without truncation
        # while staying within RTX 5090's memory budget at query batch 64.
        "max_context_tokens": 2048,
    },
    # LLaDA-8B-Instruct: same architecture, same tokenizer as LLaDA-8B-Base.
    # Drop-in swap at the adapter level, so the same MaskedDiffusionBackbone
    # code path works with only the model_id changed.
    "llada_instruct": {
        "model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "tokenizer_id": "GSAI-ML/LLaDA-8B-Instruct",
        "loader": "auto",
        "trust_remote_code": True,
        "default_dtype": "bfloat16" if BF16_OK else "float32",
        "max_context_tokens": 2048,
    },
    # Dream-7B is the second masked-diffusion family we test, so the
    # Base-vs-Instruct finding has more than one model to rest on. Dream is
    # initialised from Qwen2.5-7B and uses the Qwen tokenizer; the
    # `trust_remote_code` path picks up its custom AutoModel class. Mask
    # token convention is "<|mask|>" in the Qwen vocab, so the
    # MaskedDiffusionBackbone._infer_mask_token_id fallback (which scans
    # for "<|mask|>") will resolve it without backbone-specific code.
    "dream": {
        "model_id": "Dream-org/Dream-v0-Base-7B",
        "tokenizer_id": "Dream-org/Dream-v0-Base-7B",
        "loader": "auto",
        "trust_remote_code": True,
        "default_dtype": "bfloat16" if BF16_OK else "float32",
        "max_context_tokens": 2048,
    },
    "dream_instruct": {
        "model_id": "Dream-org/Dream-v0-Instruct-7B",
        "tokenizer_id": "Dream-org/Dream-v0-Instruct-7B",
        "loader": "auto",
        "trust_remote_code": True,
        "default_dtype": "bfloat16" if BF16_OK else "float32",
        "max_context_tokens": 2048,
    },
    # BD3-LMs is the MDLM successor that ships with both flex- and
    # SDPA-attention backends, so it loads on RTX 5090 sm_120 without
    # patching the flash-attn import. We force attn_backend="sdpa" via
    # the from_pretrained kwargs because flex_attention requires a
    # PyTorch >= 2.5 build that does not always cooperate with sm_120.
    # This unblocks the still-blocked MDLM cells in tab:mainresults.
    "bd3lm": {
        "model_id": "kuleshov-group/bd3lm-owt-block_size16",
        "tokenizer_id": "gpt2",
        "loader": "maskedlm",
        "trust_remote_code": True,
        # BD3-LM has a hardcoded float32 cast inside its timestep_embedding
        # path that conflicts with bf16 model weights, so we load it in
        # float32. The model is only ~170M parameters; float32 fits easily.
        "default_dtype": "float32",
        "max_context_tokens": 1024,
        "from_pretrained_kwargs": {"attn_backend": "sdpa"},
    },
}

SUPERIVSED_MODEL_REGISTRY = {
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "t5_base": "google/flan-t5-base",
    "bart_mnli": "facebook/bart-large-mnli",
    "setfit_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "retriever": "sentence-transformers/all-MiniLM-L6-v2",
}


@dataclass
class DatasetBundle:
    name: str
    train_texts: List[str]
    val_texts: List[str]
    test_texts: List[str]
    train_labels: List[List[int]]
    val_labels: List[List[int]]
    test_labels: List[List[int]]
    label_names: List[str]
    label_texts: List[str]
    hypothesis_template: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    instruction_key: str = "default"
    answer_separator: str = ";"
    include_label_descriptions: bool = True
    randomize_label_order: bool = False
    truncation_strategy: str = "head"
    max_context_tokens: int = 1024


@dataclass
class ShortlistConfig:
    method: str = "sbert"          # sbert | tfidf | random
    k: Optional[int] = None
    retriever_model: str = SUPERIVSED_MODEL_REGISTRY["retriever"]
    fit_on_train_texts: bool = True
    seed: int = GLOBAL_SEED


@dataclass
class InferenceConfig:
    unary_mode: str = "all_masked"  # all_masked | negative_fill_mc
    mc_samples_unary: int = 1
    mc_samples_refine: int = 4
    refine_sweeps: int = 2
    remask_strategy: str = "uniform_count"  # query_only | uniform_count | bernoulli | all_other_slots
    remask_probability: float = 0.5
    unary_batch_size: int = 4
    query_batch_size: int = 32
    seed: int = GLOBAL_SEED


@dataclass
class CalibrationConfig:
    strategy: str = "global"      # none | global | labelwise | expected_cardinality
    temperature_grid: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    threshold_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.1, 1.0, 0.1), 2))
    ece_bins: int = 15


@dataclass
class EncoderTrainConfig:
    max_length: int
    truncation_strategy: str = "head_tail"
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    seed: int = GLOBAL_SEED
    save_model: bool = False
    gradient_checkpointing: bool = False


@dataclass
class T5TrainConfig:
    max_source_length: int
    max_target_length: int
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 3
    seed: int = GLOBAL_SEED
    save_model: bool = False


def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(GLOBAL_SEED)
print(f"Device: {DEVICE} | BF16: {BF16_OK} | Root: {ROOT}")




def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def config_hash(obj: Any) -> str:
    payload = stable_json_dumps(obj).encode("utf-8")
    import hashlib
    return hashlib.sha1(payload).hexdigest()[:12]


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_trim_text(text: str, max_chars: Optional[int] = None) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = text.replace("\r", "\n").replace("\t", " ")
    text = text.strip().strip('"')
    text = re.sub(r"[ \xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text.strip()


def natural_key(x: Any) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(x))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def choose_existing(name_candidates: Sequence[str], columns: Sequence[str]) -> Optional[str]:
    lower_to_real = {c.lower(): c for c in columns}
    for name in name_candidates:
        if name.lower() in lower_to_real:
            return lower_to_real[name.lower()]
    return None


def labels_to_multihot(label_lists: Sequence[Sequence[int]], num_labels: int) -> np.ndarray:
    y = np.zeros((len(label_lists), num_labels), dtype=np.int64)
    for i, labels in enumerate(label_lists):
        if labels is None:
            continue
        for label in labels:
            if 0 <= int(label) < num_labels:
                y[i, int(label)] = 1
    return y


def multihot_to_label_lists(y: np.ndarray) -> List[List[int]]:
    return [np.flatnonzero(row > 0).astype(int).tolist() for row in y]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(np.float32).reshape(-1)
    y_prob = np.asarray(y_prob).astype(np.float32).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    ece_bins: int = 15,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(np.float32)
        metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=ece_bins)
        metrics["brier"] = float(np.mean((y_prob - y_true) ** 2))
    else:
        metrics["ece"] = np.nan
        metrics["brier"] = np.nan
    return metrics


def apply_thresholds(y_prob: np.ndarray, thresholds: Any) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if np.isscalar(thresholds):
        return (y_prob >= float(thresholds)).astype(np.int64)
    thresholds = np.asarray(thresholds).reshape(1, -1)
    return (y_prob >= thresholds).astype(np.int64)


def search_global_threshold(y_prob: np.ndarray, y_true: np.ndarray, grid: Sequence[float]) -> Tuple[float, float]:
    best_tau, best_score = 0.5, -1.0
    for tau in grid:
        pred = (y_prob >= float(tau)).astype(np.int64)
        score = f1_score(y_true, pred, average="micro", zero_division=0)
        if score > best_score:
            best_tau, best_score = float(tau), float(score)
    return best_tau, float(best_score)


def search_labelwise_thresholds(y_prob: np.ndarray, y_true: np.ndarray, grid: Sequence[float]) -> Tuple[np.ndarray, float]:
    m = y_true.shape[1]
    thresholds = np.full(m, 0.5, dtype=np.float32)
    global_tau, _ = search_global_threshold(y_prob, y_true, grid)
    for j in range(m):
        positives = y_true[:, j].sum()
        if positives < 3:
            thresholds[j] = global_tau
            continue
        best_tau, best_score = global_tau, -1.0
        for tau in grid:
            score = f1_score(y_true[:, j], (y_prob[:, j] >= tau).astype(np.int64), zero_division=0)
            if score > best_score:
                best_tau, best_score = float(tau), float(score)
        thresholds[j] = best_tau
    preds = apply_thresholds(y_prob, thresholds)
    micro = f1_score(y_true, preds, average="micro", zero_division=0)
    return thresholds, float(micro)


def threshold_from_expected_cardinality(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    target_cardinality = float(y_true.sum(axis=1).mean())
    candidate_thresholds = np.unique(np.round(y_prob.reshape(-1), 4))
    if len(candidate_thresholds) > 400:
        candidate_thresholds = np.linspace(0.01, 0.99, 200)
    best_tau, best_gap = 0.5, float("inf")
    for tau in candidate_thresholds:
        avg_card = float((y_prob >= tau).sum(axis=1).mean())
        gap = abs(avg_card - target_cardinality)
        if gap < best_gap:
            best_tau, best_gap = float(tau), float(gap)
    return best_tau


def tune_temperature_and_threshold(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    calibration_cfg: CalibrationConfig,
    scores_are_logits: bool = True,
) -> Dict[str, Any]:
    val_scores = np.asarray(val_scores, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    if calibration_cfg.strategy == "none":
        if scores_are_logits:
            probs = expit(val_scores)
        else:
            probs = np.clip(val_scores, 0.0, 1.0)
        return {"temperature": 1.0, "threshold": 0.5, "val_micro_f1": f1_score(y_val, probs >= 0.5, average="micro", zero_division=0)}

    # When the user picks `auto`, sweep over every concrete strategy and pick the
    # best one on the validation set.  This avoids the BART-MNLI failure mode where
    # the default `global` threshold of 0.5 leaves most labels unpredicted, while
    # `expected_cardinality` recovers a much better operating point.
    if calibration_cfg.strategy == "auto":
        candidates = []
        for sub in ("global", "labelwise", "expected_cardinality"):
            sub_cfg = CalibrationConfig(
                strategy=sub,
                temperature_grid=calibration_cfg.temperature_grid,
                threshold_grid=calibration_cfg.threshold_grid,
                ece_bins=calibration_cfg.ece_bins,
            )
            picked = tune_temperature_and_threshold(val_scores, y_val, sub_cfg, scores_are_logits=scores_are_logits)
            picked["chosen_strategy"] = sub
            candidates.append(picked)
        return max(candidates, key=lambda c: c["val_micro_f1"])

    best = None
    for temperature in calibration_cfg.temperature_grid if scores_are_logits else (1.0,):
        probs = expit(val_scores / temperature) if scores_are_logits else np.clip(val_scores, 0.0, 1.0)

        if calibration_cfg.strategy == "global":
            tau, score = search_global_threshold(probs, y_val, calibration_cfg.threshold_grid)
            candidate = {"temperature": float(temperature), "threshold": float(tau), "val_micro_f1": float(score)}

        elif calibration_cfg.strategy == "labelwise":
            taus, score = search_labelwise_thresholds(probs, y_val, calibration_cfg.threshold_grid)
            candidate = {"temperature": float(temperature), "threshold": taus.tolist(), "val_micro_f1": float(score)}

        elif calibration_cfg.strategy == "expected_cardinality":
            tau = threshold_from_expected_cardinality(probs, y_val)
            pred = (probs >= tau).astype(np.int64)
            score = f1_score(y_val, pred, average="micro", zero_division=0)
            candidate = {"temperature": float(temperature), "threshold": float(tau), "val_micro_f1": float(score)}

        else:
            raise ValueError(f"Unknown calibration strategy: {calibration_cfg.strategy}")

        if best is None or candidate["val_micro_f1"] > best["val_micro_f1"]:
            best = candidate

    return best


def probs_from_scores(scores: np.ndarray, calibration_result: Dict[str, Any], scores_are_logits: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores_are_logits:
        temp = float(calibration_result.get("temperature", 1.0))
        return expit(scores / temp)
    return np.clip(scores, 0.0, 1.0)


def predict_from_scores(scores: np.ndarray, calibration_result: Dict[str, Any], scores_are_logits: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    probs = probs_from_scores(scores, calibration_result, scores_are_logits=scores_are_logits)
    pred = apply_thresholds(probs, calibration_result.get("threshold", 0.5))
    return pred, probs


def iterative_train_val_split(
    texts: List[str],
    label_lists: List[List[int]],
    num_labels: int,
    val_size: float = 0.1,
    seed: int = GLOBAL_SEED,
) -> Tuple[List[str], List[str], List[List[int]], List[List[int]]]:
    y = labels_to_multihot(label_lists, num_labels)

    if MultilabelStratifiedShuffleSplit is not None:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(texts)), y))
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(texts))
        rng.shuffle(idx)
        cut = int(len(idx) * (1.0 - val_size))
        train_idx, val_idx = idx[:cut], idx[cut:]

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [label_lists[i] for i in train_idx]
    val_labels = [label_lists[i] for i in val_idx]
    return train_texts, val_texts, train_labels, val_labels


def maybe_subsample_split(texts: List[str], label_lists: List[List[int]], fraction: float, seed: int = GLOBAL_SEED) -> Tuple[List[str], List[List[int]]]:
    if fraction >= 1.0:
        return texts, label_lists
    rng = np.random.default_rng(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    keep = idx[: max(1, int(len(idx) * fraction))]
    keep = sorted(keep.tolist())
    return [texts[i] for i in keep], [label_lists[i] for i in keep]


def write_result_row(row: Dict[str, Any], table_name: str) -> Path:
    path = RESULTS_DIR / f"{table_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def timer() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def current_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))





def _assert_datasets_available() -> None:
    if load_dataset is None:
        raise ImportError("The `datasets` package is required. Set INSTALL_REQUIREMENTS=True and rerun the setup cell.")


def _parse_generic_label_field(value: Any, name_to_idx: Dict[str, int], num_labels: Optional[int] = None) -> List[int]:
    if value is None:
        return []

    if isinstance(value, (np.ndarray,)):
        value = value.tolist()

    if isinstance(value, (int, np.integer)):
        return [int(value)]

    if isinstance(value, str):
        if value.strip() == "":
            return []
        if value.strip().startswith("[") and value.strip().endswith("]"):
            try:
                parsed = json.loads(value)
                return _parse_generic_label_field(parsed, name_to_idx, num_labels=num_labels)
            except Exception:
                pass
        if "," in value:
            return sorted({name_to_idx[x.strip()] for x in value.split(",") if x.strip() in name_to_idx})
        if value in name_to_idx:
            return [name_to_idx[value]]
        if value.isdigit():
            return [int(value)]

    if isinstance(value, dict):
        labels = []
        for k, v in value.items():
            if not v:
                continue
            if k in name_to_idx:
                labels.append(name_to_idx[k])
            elif str(k).isdigit():
                labels.append(int(k))
        return sorted(set(labels))

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return []
        # list of ids
        if all(isinstance(x, (int, np.integer)) for x in value):
            if num_labels is not None and len(value) == num_labels and set(np.unique(value)).issubset({0, 1}):
                return np.flatnonzero(np.asarray(value) > 0).astype(int).tolist()
            return sorted(set(int(x) for x in value))
        # list of names
        if all(isinstance(x, str) for x in value):
            out = []
            for x in value:
                if x in name_to_idx:
                    out.append(name_to_idx[x])
                elif x.isdigit():
                    out.append(int(x))
            return sorted(set(out))
        # list of bools / 0-1
        try:
            arr = np.asarray(value)
            if num_labels is not None and arr.ndim == 1 and len(arr) == num_labels and set(np.unique(arr)).issubset({0, 1, False, True}):
                return np.flatnonzero(arr.astype(int) > 0).astype(int).tolist()
        except Exception:
            pass

    return []


def _try_load_dataset(candidates: Sequence[Tuple[str, Optional[str], Optional[Dict[str, Any]]]]) -> Tuple[Any, str, Optional[str]]:
    _assert_datasets_available()
    last_error = None
    for repo_id, config_name, kwargs in candidates:
        kwargs = kwargs or {}
        try:
            if config_name is None:
                ds = load_dataset(repo_id, **kwargs)
            else:
                ds = load_dataset(repo_id, config_name, **kwargs)
            return ds, repo_id, config_name
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load dataset candidates. Last error: {last_error}")


def _discover_label_map_from_hf_repo(repo_id: str, repo_type: str = "dataset") -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        repo_files = list_repo_files(repo_id, repo_type=repo_type)
    except Exception:
        return mapping

    candidate_files = [
        f for f in repo_files
        if re.search(r"(label|labels|descriptor|descriptors|eurovoc|concept)", f, flags=re.I)
        and re.search(r"\.(json|jsonl|csv|tsv|txt)$", f, flags=re.I)
    ]

    def _parse_candidate_file(local_path: str) -> Dict[str, str]:
        local_mapping: Dict[str, str] = {}
        suffix = Path(local_path).suffix.lower()

        try:
            if suffix in {".json", ".jsonl"}:
                with open(local_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    # direct id -> text
                    for k, v in raw.items():
                        if isinstance(v, str):
                            local_mapping[str(k)] = v
                        elif isinstance(v, dict):
                            desc = v.get("label") or v.get("descriptor") or v.get("text") or v.get("name")
                            if desc is not None:
                                local_mapping[str(k)] = str(desc)
                elif isinstance(raw, list):
                    for row in raw:
                        if not isinstance(row, dict):
                            continue
                        id_key = next((k for k in row if k.lower() in {"id", "label_id", "concept_id", "eurovoc_id"}), None)
                        desc_key = next((k for k in row if k.lower() in {"label", "descriptor", "text", "name"}), None)
                        if id_key and desc_key:
                            local_mapping[str(row[id_key])] = str(row[desc_key])

            elif suffix in {".csv", ".tsv", ".txt"}:
                sep = "\t" if suffix in {".tsv", ".txt"} else ","
                df = pd.read_csv(local_path, sep=sep)
                id_col = next((c for c in df.columns if c.lower() in {"id", "label_id", "concept_id", "eurovoc_id"}), None)
                text_col = next((c for c in df.columns if c.lower() in {"label", "descriptor", "text", "name"}), None)
                if id_col and text_col:
                    local_mapping = {str(k): str(v) for k, v in zip(df[id_col], df[text_col]) if pd.notna(k) and pd.notna(v)}
        except Exception:
            return {}
        return local_mapping

    for filename in candidate_files:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            parsed = _parse_candidate_file(local)
            if len(parsed) > len(mapping):
                mapping = parsed
        except Exception:
            continue

    return mapping


def _combine_text_fields(example: Dict[str, Any], preferred_order: Sequence[str]) -> str:
    parts = []
    for key in preferred_order:
        if key in example and example[key]:
            value = example[key]
            if isinstance(value, str):
                parts.append(maybe_trim_text(value))
            elif isinstance(value, list):
                parts.extend(maybe_trim_text(v) for v in value if v)
    if not parts and "text" in example and isinstance(example["text"], str):
        parts = [maybe_trim_text(example["text"])]
    return "\n\n".join([p for p in parts if p]).strip()


def load_goemotions_bundle() -> DatasetBundle:
    candidates = [
        ("google-research-datasets/go_emotions", None, {}),
        ("go_emotions", "raw", {}),
        ("mrm8488/goemotions", None, {}),
    ]
    ds, source_repo, source_config = _try_load_dataset(candidates)

    train_split = ds["train"]
    val_split = ds["validation"] if "validation" in ds else (ds["val"] if "val" in ds else None)
    test_split = ds["test"]

    text_col = choose_existing(["text", "comment_text", "sentence"], train_split.column_names)
    label_col = choose_existing(["labels", "label", "emotions"], train_split.column_names)
    if text_col is None or label_col is None:
        raise RuntimeError(f"Could not infer text/label columns for GoEmotions: {train_split.column_names}")

    label_names = GOEMOTIONS_LABELS
    try:
        feature = train_split.features[label_col]
        if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
            label_names = list(feature.feature.names)
        elif hasattr(feature, "names"):
            label_names = list(feature.names)
    except Exception:
        pass
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    def _convert(split):
        texts, labels = [], []
        for ex in split:
            text = maybe_trim_text(ex[text_col])
            labs = _parse_generic_label_field(ex[label_col], name_to_idx, num_labels=len(label_names))
            if text:
                texts.append(text)
                labels.append(labs)
        return texts, labels

    train_texts, train_labels = _convert(train_split)
    if val_split is not None:
        val_texts, val_labels = _convert(val_split)
    else:
        train_texts, val_texts, train_labels, val_labels = iterative_train_val_split(
            train_texts, train_labels, len(label_names), val_size=0.1, seed=GLOBAL_SEED
        )
    test_texts, test_labels = _convert(test_split)

    if DRY_RUN:
        train_texts, train_labels = maybe_subsample_split(train_texts, train_labels, 0.05)
        val_texts, val_labels = maybe_subsample_split(val_texts, val_labels, 0.10)
        test_texts, test_labels = maybe_subsample_split(test_texts, test_labels, 0.10)

    cfg = DATASET_DEFAULTS["goemotions"]
    return DatasetBundle(
        name="goemotions",
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=label_names,
        label_texts=label_names.copy(),
        hypothesis_template=cfg["hypothesis_template"],
        metadata={"source_repo": source_repo, "source_config": source_config},
    )


def load_reuters21578_top20_bundle(top_k: int = 20) -> DatasetBundle:
    # `Tellurio/reuters-21578` ships the standard ModApte (lewis_split) Reuters with
    # raw `topics`, `body`, `title` columns and is parquet-only, so it works with
    # `datasets>=3` (which dropped script-based loaders).
    candidates = [
        ("Tellurio/reuters-21578", None, {}),
        ("ucirvine/reuters21578", "ModApte", {}),
        ("ucirvine/reuters21578", None, {}),
    ]
    ds, source_repo, source_config = _try_load_dataset(candidates)

    train_split = ds["train"]
    test_split = ds["test"]

    title_col = choose_existing(["title"], train_split.column_names)
    text_col = choose_existing(["text", "body"], train_split.column_names)
    topic_col = choose_existing(["topics", "labels", "label"], train_split.column_names)
    split_col = choose_existing(["lewis_split", "lewissplit", "attr__lewissplit", "split"], train_split.column_names)
    if text_col is None or topic_col is None:
        raise RuntimeError(f"Could not infer text/label columns for Reuters: {train_split.column_names}")

    def _is_modapte_train(ex) -> bool:
        if split_col is None:
            return True
        val = str(ex[split_col]).upper()
        return val in {"TRAIN", "TRAINING", "TRAINING-SET"}

    def _is_modapte_test(ex) -> bool:
        if split_col is None:
            return True
        val = str(ex[split_col]).upper()
        return val in {"TEST", "TESTING", "TEST-SET"}

    # Some sources (Tellurio/reuters-21578) store topics as ClassLabel ints; resolve
    # them to the canonical string names via the dataset features.
    topic_int_to_str = None
    try:
        feat = train_split.features[topic_col]
        inner = getattr(feat, "feature", None) or feat
        names = getattr(inner, "names", None)
        if names is not None:
            topic_int_to_str = {i: name for i, name in enumerate(names)}
    except Exception:
        topic_int_to_str = None

    def _topics_to_str(values):
        if values is None:
            return []
        out = []
        for v in values:
            if isinstance(v, str):
                out.append(v)
            elif topic_int_to_str is not None:
                name = topic_int_to_str.get(int(v))
                if name is not None:
                    out.append(name)
        return out

    def _raw_topics(split, modapte_filter=None) -> List[List[str]]:
        out = []
        for ex in split:
            if modapte_filter is not None and not modapte_filter(ex):
                continue
            labels = ex[topic_col]
            if isinstance(labels, str):
                labels = [labels]
            out.append(_topics_to_str(labels))
        return out

    train_topics = _raw_topics(train_split, _is_modapte_train if split_col else None)
    freq = Counter(t for row in train_topics for t in row)
    top_labels = [lab for lab, _ in freq.most_common(top_k)]
    name_to_idx = {name: i for i, name in enumerate(top_labels)}

    def _convert(split, modapte_filter=None):
        texts, labels = [], []
        for ex in split:
            if modapte_filter is not None and not modapte_filter(ex):
                continue
            title = maybe_trim_text(ex[title_col]) if title_col else ""
            body = maybe_trim_text(ex[text_col])
            text = "\n\n".join([x for x in [title, body] if x]).strip()
            topics_str = _topics_to_str(ex[topic_col])
            labs = [name_to_idx[t] for t in topics_str if t in name_to_idx]
            if text and labs:
                texts.append(text)
                labels.append(sorted(set(labs)))
        return texts, labels

    # When the source dataset uses lewis_split rows mixed across the parquet file,
    # the explicit ModApte filter ensures we use the canonical 9603 / 3299 split.
    train_texts, train_labels = _convert(train_split, _is_modapte_train if split_col else None)
    test_texts, test_labels = _convert(test_split, _is_modapte_test if split_col else None)

    train_texts, val_texts, train_labels, val_labels = iterative_train_val_split(
        train_texts, train_labels, len(top_labels), val_size=0.1, seed=GLOBAL_SEED
    )

    if DRY_RUN:
        train_texts, train_labels = maybe_subsample_split(train_texts, train_labels, 0.10)
        val_texts, val_labels = maybe_subsample_split(val_texts, val_labels, 0.20)
        test_texts, test_labels = maybe_subsample_split(test_texts, test_labels, 0.20)

    cfg = DATASET_DEFAULTS["reuters21578_top20"]
    label_texts = [REUTERS_TOPIC_TEXT.get(label, label.replace("-", " ")) for label in top_labels]

    return DatasetBundle(
        name="reuters21578_top20",
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=top_labels,
        label_texts=label_texts,
        hypothesis_template=cfg["hypothesis_template"],
        metadata={"source_repo": source_repo, "source_config": source_config},
    )


def load_eurlex57k_bundle() -> DatasetBundle:
    candidates = [
        ("jonathanli/eurlex", "eurlex57k", {}),
        ("jonathanli/eurlex", None, {}),
        ("coastalcph/lex_glue", "eurlex", {}),
        ("lex_glue", "eurlex", {}),
    ]
    ds, source_repo, source_config = _try_load_dataset(candidates)

    train_split = ds["train"]
    if "validation" in ds:
        val_split = ds["validation"]
    elif "dev" in ds:
        val_split = ds["dev"]
    else:
        val_split = None
    test_split = ds["test"]

    label_col = choose_existing(["labels", "label", "eurovoc_concepts", "concepts"], train_split.column_names)
    if label_col is None:
        raise RuntimeError(f"Could not find labels column for EURLEX: {train_split.column_names}")

    def _convert(split):
        texts, labels = [], []
        for ex in split:
            text = _combine_text_fields(ex, ["title", "header", "recitals", "main_body", "text"])
            labels_raw = ex[label_col]
            if isinstance(labels_raw, str):
                labels_raw = [labels_raw]
            labels.append([str(x) for x in labels_raw if str(x) != ""])
            texts.append(text)
        return texts, labels

    train_texts_raw, train_label_ids_raw = _convert(train_split)
    if val_split is not None:
        val_texts_raw, val_label_ids_raw = _convert(val_split)
    else:
        val_texts_raw, val_label_ids_raw = [], []
    test_texts_raw, test_label_ids_raw = _convert(test_split)

    all_label_ids = sorted(
        set(x for row in train_label_ids_raw + val_label_ids_raw + test_label_ids_raw for x in row),
        key=natural_key,
    )
    id_to_idx = {lab: i for i, lab in enumerate(all_label_ids)}

    label_map = _discover_label_map_from_hf_repo("jonathanli/eurlex", repo_type="dataset")
    if len(label_map) < len(all_label_ids) // 2:
        label_map = _discover_label_map_from_hf_repo("coastalcph/multi_eurlex", repo_type="dataset") or label_map

    def _map_labels(rows):
        return [[id_to_idx[x] for x in labels if x in id_to_idx] for labels in rows]

    train_labels = _map_labels(train_label_ids_raw)
    if val_split is not None:
        val_labels = _map_labels(val_label_ids_raw)
        val_texts = train_texts_raw if False else val_texts_raw
    else:
        train_texts_raw, val_texts_raw, train_labels, val_labels = iterative_train_val_split(
            train_texts_raw, train_labels, len(all_label_ids), val_size=0.1, seed=GLOBAL_SEED
        )
        val_texts = val_texts_raw

    test_labels = _map_labels(test_label_ids_raw)

    label_names = all_label_ids
    label_texts = [label_map.get(label_id, str(label_id)) for label_id in label_names]

    train_texts = train_texts_raw
    if val_split is not None:
        val_texts = val_texts_raw
    test_texts = test_texts_raw

    if DRY_RUN:
        train_texts, train_labels = maybe_subsample_split(train_texts, train_labels, 0.03)
        val_texts, val_labels = maybe_subsample_split(val_texts, val_labels, 0.10)
        test_texts, test_labels = maybe_subsample_split(test_texts, test_labels, 0.10)

    cfg = DATASET_DEFAULTS["eurlex57k"]
    return DatasetBundle(
        name="eurlex57k",
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=label_names,
        label_texts=label_texts,
        hypothesis_template=cfg["hypothesis_template"],
        metadata={
            "source_repo": source_repo,
            "source_config": source_config,
            "label_map_coverage": float(sum(label_map.get(x) is not None for x in label_names)) / max(1, len(label_names)),
        },
    )


def load_multieurlex_en_bundle(label_granularity: str = "original") -> DatasetBundle:
    _assert_datasets_available()
    config_names = []
    if get_dataset_config_names is not None:
        try:
            config_names = get_dataset_config_names("coastalcph/multi_eurlex")
        except Exception:
            config_names = []

    preferred_configs = []
    for cfg_name in config_names:
        if cfg_name.lower() in {"en", "english"} or "en" == cfg_name.lower():
            preferred_configs.append(cfg_name)
    if not preferred_configs:
        preferred_configs = [None]

    candidates = [("coastalcph/multi_eurlex", cfg, {}) for cfg in preferred_configs] + [("coastalcph/multi_eurlex", None, {})]
    ds, source_repo, source_config = _try_load_dataset(candidates)

    train_split = ds["train"]
    val_split = ds["validation"] if "validation" in ds else (ds["dev"] if "dev" in ds else None)
    test_split = ds["test"]

    possible_label_cols = {
        "original": ["labels", "label", "eurovoc_concepts", "concepts"],
        "level_1": ["labels_level_1", "level_1_labels"],
        "level_2": ["labels_level_2", "level_2_labels"],
        "level_3": ["labels_level_3", "level_3_labels"],
    }[label_granularity]
    label_col = choose_existing(possible_label_cols, train_split.column_names) or choose_existing(["labels", "eurovoc_concepts"], train_split.column_names)
    if label_col is None:
        raise RuntimeError(f"Could not find labels column for MultiEURLEX: {train_split.column_names}")

    def _extract_text(ex):
        if "text" in ex:
            if isinstance(ex["text"], dict):
                for key in ["en", "english"]:
                    if key in ex["text"]:
                        return maybe_trim_text(ex["text"][key])
            if isinstance(ex["text"], str):
                return maybe_trim_text(ex["text"])
        return _combine_text_fields(ex, ["title", "header", "recitals", "main_body"])

    def _convert(split):
        texts, label_rows = [], []
        for ex in split:
            text = _extract_text(ex)
            labels_raw = ex[label_col]
            if isinstance(labels_raw, str):
                labels_raw = [labels_raw]
            label_rows.append([str(x) for x in labels_raw if str(x) != ""])
            texts.append(text)
        return texts, label_rows

    train_texts_raw, train_label_ids_raw = _convert(train_split)
    val_texts_raw, val_label_ids_raw = _convert(val_split)
    test_texts_raw, test_label_ids_raw = _convert(test_split)

    all_label_ids = sorted(
        set(x for row in train_label_ids_raw + val_label_ids_raw + test_label_ids_raw for x in row),
        key=natural_key,
    )
    id_to_idx = {lab: i for i, lab in enumerate(all_label_ids)}
    label_map = _discover_label_map_from_hf_repo("coastalcph/multi_eurlex", repo_type="dataset")

    def _map(rows):
        return [[id_to_idx[x] for x in row if x in id_to_idx] for row in rows]

    train_labels = _map(train_label_ids_raw)
    val_labels = _map(val_label_ids_raw)
    test_labels = _map(test_label_ids_raw)

    if DRY_RUN:
        train_texts_raw, train_labels = maybe_subsample_split(train_texts_raw, train_labels, 0.02)
        val_texts_raw, val_labels = maybe_subsample_split(val_texts_raw, val_labels, 0.05)
        test_texts_raw, test_labels = maybe_subsample_split(test_texts_raw, test_labels, 0.05)

    cfg = DATASET_DEFAULTS["multieurlex_en"]
    return DatasetBundle(
        name="multieurlex_en",
        train_texts=train_texts_raw,
        val_texts=val_texts_raw,
        test_texts=test_texts_raw,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=all_label_ids,
        label_texts=[label_map.get(x, x) for x in all_label_ids],
        hypothesis_template=cfg["hypothesis_template"],
        metadata={"source_repo": source_repo, "source_config": source_config, "label_granularity": label_granularity},
    )


_ECTHR_A_LABEL_NAMES = [
    "Article 2 (right to life)",
    "Article 3 (prohibition of torture)",
    "Article 5 (right to liberty and security)",
    "Article 6 (right to a fair trial)",
    "Article 8 (right to respect for private life)",
    "Article 9 (freedom of thought, conscience and religion)",
    "Article 10 (freedom of expression)",
    "Article 11 (freedom of assembly and association)",
    "Article 14 (prohibition of discrimination)",
    "Protocol 1 Article 1 (protection of property)",
]


def load_ecthr_a_bundle(max_paragraph_chars: int = 4000) -> DatasetBundle:
    """ECtHR Task A from LexGLUE. 10 articles of the European Convention on
    Human Rights, ~9k train / 1k val / 1k test, long case facts. Long-doc /
    low-cardinality corner of the dataset design space.

    Text format: the LexGLUE release stores each example as a list of
    paragraphs. We concatenate them with newline separators and truncate by
    character count so the bundle loader stays deterministic; the backbone
    tokenizers then cut at their own max_length.
    """
    candidates = [
        ("coastalcph/lex_glue", "ecthr_a", {}),
        ("lex_glue", "ecthr_a", {}),
    ]
    ds, source_repo, source_config = _try_load_dataset(candidates)
    train_split = ds["train"]
    val_split = ds["validation"] if "validation" in ds else ds.get("dev")
    test_split = ds["test"]

    def _convert(split):
        texts: List[str] = []
        labels: List[List[int]] = []
        for ex in split:
            paragraphs = ex.get("text") or []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            joined = "\n".join(paragraphs).strip()
            if max_paragraph_chars and len(joined) > max_paragraph_chars:
                joined = joined[:max_paragraph_chars]
            texts.append(joined)
            # labels field is a List[int] of label indices in the 10-class space
            lab = ex.get("labels") or []
            if isinstance(lab, int):
                lab = [lab]
            labels.append([int(x) for x in lab])
        return texts, labels

    train_texts, train_labels = _convert(train_split)
    val_texts, val_labels = _convert(val_split) if val_split is not None else ([], [])
    test_texts, test_labels = _convert(test_split)

    if not val_texts:
        train_texts, val_texts, train_labels, val_labels = iterative_train_val_split(
            train_texts, train_labels, len(_ECTHR_A_LABEL_NAMES), val_size=0.1, seed=GLOBAL_SEED
        )

    label_names = _ECTHR_A_LABEL_NAMES
    label_texts = _ECTHR_A_LABEL_NAMES[:]

    if DRY_RUN:
        train_texts, train_labels = maybe_subsample_split(train_texts, train_labels, 0.05)
        val_texts, val_labels = maybe_subsample_split(val_texts, val_labels, 0.20)
        test_texts, test_labels = maybe_subsample_split(test_texts, test_labels, 0.20)

    cfg = DATASET_DEFAULTS["ecthr_a"]
    return DatasetBundle(
        name="ecthr_a",
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=label_names,
        label_texts=label_texts,
        hypothesis_template=cfg["hypothesis_template"],
        metadata={
            "source_repo": source_repo,
            "source_config": source_config,
            "truncation": f"head-{max_paragraph_chars} chars",
        },
    )


_JIGSAW_LABEL_NAMES = [
    "toxic",
    "severely toxic",
    "obscene",
    "a threat",
    "an insult",
    "identity hate",
]
# Mapping from the 6 CSV columns in the Kaggle release to the list index used above.
_JIGSAW_CSV_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_jigsaw_toxic_bundle(
    max_train: int = 20000,
    max_val: int = 1500,
    max_test: int = 1500,
) -> DatasetBundle:
    """Jigsaw Toxic Comment Classification Challenge. Six binary labels
    (toxic, severe_toxic, obscene, threat, insult, identity_hate) on short
    Wikipedia talk-page comments. Biggest domain shift in our suite.

    We pull the HF mirror ``thesofakillers/jigsaw-toxic-comment-classification-challenge``.
    The mirror ships the raw Kaggle ``train.csv`` (159,571 labeled rows) and
    the original ``test.csv`` --- note that Kaggle never released the gold
    test labels publicly, so the ``test`` split on this mirror has all
    label columns = None. We therefore build train / val / test from a
    deterministic 80/10/10 stratified split of the labeled ``train`` split,
    which is what most published Jigsaw-based experiments do anyway.
    """
    candidates = [
        ("thesofakillers/jigsaw-toxic-comment-classification-challenge", None, {}),
    ]
    ds, source_repo, source_config = _try_load_dataset(candidates)

    def _rows_to_arrays(split) -> Tuple[List[str], List[List[int]]]:
        texts: List[str] = []
        labels: List[List[int]] = []
        for ex in split:
            text = ex.get("comment_text") or ex.get("text") or ""
            label_vec: List[int] = []
            any_missing = False
            for col in _JIGSAW_CSV_COLS:
                raw = ex.get(col, 0)
                if raw is None:
                    any_missing = True
                    break
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    any_missing = True
                    break
                if val == -1:
                    any_missing = True
                    break
                label_vec.append(val)
            if any_missing:
                continue
            pos = [i for i, v in enumerate(label_vec) if v]
            texts.append(str(text))
            labels.append(pos)
        return texts, labels

    all_texts, all_labels = _rows_to_arrays(ds["train"])
    # Deterministic 80/10/10 split (RNG seeded from GLOBAL_SEED). We do it
    # manually to keep the split reproducible across datasets versions.
    n = len(all_texts)
    rng = np.random.default_rng(GLOBAL_SEED + 7)
    perm = rng.permutation(n)
    n_test_pool = max(1, int(n * 0.10))
    n_val_pool = max(1, int(n * 0.10))
    test_ids = perm[:n_test_pool]
    val_ids = perm[n_test_pool : n_test_pool + n_val_pool]
    train_ids = perm[n_test_pool + n_val_pool :]

    def _slice(ids):
        return [all_texts[i] for i in ids], [all_labels[i] for i in ids]

    train_texts, train_labels = _slice(train_ids)
    val_texts, val_labels = _slice(val_ids)
    test_texts, test_labels = _slice(test_ids)

    # Stratified subsample from each pool so the positive rate matches the
    # natural distribution (preserves the sharp toxic vs non-toxic imbalance).
    rng_sample = np.random.default_rng(GLOBAL_SEED + 13)

    def _stratified_slice(texts, labels, n):
        if len(texts) <= n:
            return texts, labels
        has_pos = np.array([1 if lbl else 0 for lbl in labels])
        pos_idx = np.where(has_pos == 1)[0]
        neg_idx = np.where(has_pos == 0)[0]
        p_rate = len(pos_idx) / max(1, len(has_pos))
        n_pos = int(round(n * p_rate))
        n_neg = n - n_pos
        n_pos = min(n_pos, len(pos_idx))
        n_neg = min(n_neg, len(neg_idx))
        keep_pos = rng_sample.choice(pos_idx, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
        keep_neg = rng_sample.choice(neg_idx, size=n_neg, replace=False) if n_neg else np.array([], dtype=int)
        keep = np.sort(np.concatenate([keep_pos, keep_neg]))
        return [texts[i] for i in keep], [labels[i] for i in keep]

    test_texts, test_labels = _stratified_slice(test_texts, test_labels, max_test)
    val_texts, val_labels = _stratified_slice(val_texts, val_labels, max_val)
    train_texts, train_labels = _stratified_slice(train_texts, train_labels, max_train)

    if DRY_RUN:
        train_texts, train_labels = maybe_subsample_split(train_texts, train_labels, 0.05)
        val_texts, val_labels = maybe_subsample_split(val_texts, val_labels, 0.20)
        test_texts, test_labels = maybe_subsample_split(test_texts, test_labels, 0.20)

    cfg = DATASET_DEFAULTS["jigsaw_toxic"]
    return DatasetBundle(
        name="jigsaw_toxic",
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=_JIGSAW_LABEL_NAMES,
        label_texts=_JIGSAW_LABEL_NAMES[:],
        hypothesis_template=cfg["hypothesis_template"],
        metadata={
            "source_repo": source_repo,
            "source_config": source_config,
            "subsample": {"train": max_train, "val": max_val, "test": max_test},
        },
    )


def load_aapd_bundle() -> DatasetBundle:
    """Load the AAPD (ArXiv Academic Paper Dataset) for multi-label classification.

    54 CS/math/physics subject labels, ~54K train, ~1K dev, ~1K test.
    Source: Yang et al. 2018 via Zenodo record 6344750.
    The CSV files must be present at ``data/aapd/{train,dev,test}.csv``.
    """
    import csv

    base = Path(__file__).resolve().parent.parent.parent / "data" / "aapd"
    if not base.exists():
        raise FileNotFoundError(
            f"AAPD data not found at {base}. Download from "
            "https://zenodo.org/records/6344750 and unzip to data/aapd/."
        )

    def _read_csv(path: Path):
        texts, labels_list = [], []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            label_cols = header[1:]  # first column is 'abstract'
            for row in reader:
                if len(row) < len(header):
                    continue
                texts.append(row[0])
                idxs = [i for i, v in enumerate(row[1:]) if v.strip() == "1"]
                labels_list.append(idxs)
        return texts, labels_list, label_cols

    train_texts, train_labels, label_names = _read_csv(base / "train.csv")
    val_texts, val_labels, _ = _read_csv(base / "dev.csv")
    test_texts, test_labels, _ = _read_csv(base / "test.csv")

    # Human-readable label texts: replace dots with spaces, expand abbreviations
    label_texts = [
        ln.replace(".", " ").replace("cs ", "computer science ")
          .replace("cond-mat ", "condensed matter ")
          .replace("math ", "mathematics ")
          .replace("stat ", "statistics ")
          .replace("nlin ", "nonlinear ")
          .replace("q-bio ", "quantitative biology ")
        for ln in label_names
    ]

    return DatasetBundle(
        name="aapd",
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_names=label_names,
        label_texts=label_texts,
        hypothesis_template="This paper is about {}.",
    )


def load_dataset_bundle(name: str) -> DatasetBundle:
    if name == "goemotions":
        return load_goemotions_bundle()
    if name == "reuters21578_top20":
        return load_reuters21578_top20_bundle()
    if name == "eurlex57k":
        return load_eurlex57k_bundle()
    if name == "multieurlex_en":
        return load_multieurlex_en_bundle()
    if name == "ecthr_a":
        return load_ecthr_a_bundle()
    if name == "jigsaw_toxic":
        return load_jigsaw_toxic_bundle()
    if name == "aapd":
        return load_aapd_bundle()
    raise ValueError(f"Unknown dataset name: {name}")


def describe_bundle(bundle: DatasetBundle) -> pd.DataFrame:
    rows = []
    for split_name, texts, labels in [
        ("train", bundle.train_texts, bundle.train_labels),
        ("val", bundle.val_texts, bundle.val_labels),
        ("test", bundle.test_texts, bundle.test_labels),
    ]:
        card = [len(x) for x in labels]
        rows.append({
            "dataset": bundle.name,
            "split": split_name,
            "num_examples": len(texts),
            "num_labels": len(bundle.label_names),
            "avg_cardinality": float(np.mean(card)) if card else 0.0,
            "median_cardinality": float(np.median(card)) if card else 0.0,
            "avg_chars": float(np.mean([len(t) for t in texts])) if texts else 0.0,
        })
    return pd.DataFrame(rows)





class ShortlistRetriever:
    def __init__(
        self,
        label_texts: List[str],
        train_texts: Optional[List[str]],
        config: ShortlistConfig,
        cache_namespace: str,
    ) -> None:
        self.label_texts = label_texts
        self.train_texts = train_texts or []
        self.config = config
        self.cache_namespace = cache_namespace
        self.method = config.method
        self.k = config.k
        self._built = False
        self._vectorizer = None
        self._label_matrix = None
        self._st_model = None
        self._label_embeddings = None

    @property
    def cache_path(self) -> Path:
        cfg = {
            "label_count": len(self.label_texts),
            "train_count": len(self.train_texts),
            "config": asdict(self.config),
            "namespace": self.cache_namespace,
        }
        return CACHE_DIR / "retrievers" / f"{slugify(self.cache_namespace)}_{config_hash(cfg)}.pkl"

    def _ensure_query_encoder(self) -> None:
        if self.method == "sbert" and self._st_model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for SBERT retrieval.")
            self._st_model = SentenceTransformer(self.config.retriever_model, device=DEVICE)

    def fit(self) -> "ShortlistRetriever":
        if self._built:
            self._ensure_query_encoder()
            return self
        if self.cache_path.exists() and not OVERWRITE_CACHE:
            payload = load_pickle(self.cache_path)
            self.__dict__.update(payload)
            self._built = True
            self._ensure_query_encoder()
            return self

        if self.method == "random":
            self._built = True
            save_pickle({
                "_built": True,
                "method": self.method,
                "label_texts": self.label_texts,
                "train_texts": self.train_texts,
                "config": self.config,
                "cache_namespace": self.cache_namespace,
            }, self.cache_path)
            return self

        if self.method == "tfidf":
            corpus = list(self.label_texts)
            if self.config.fit_on_train_texts and self.train_texts:
                sample_texts = self.train_texts
                if len(sample_texts) > 20000:
                    rng = np.random.default_rng(self.config.seed)
                    idx = rng.choice(len(sample_texts), size=20000, replace=False)
                    sample_texts = [sample_texts[i] for i in idx]
                corpus = corpus + sample_texts
            vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
            vectorizer.fit(corpus)
            label_matrix = vectorizer.transform(self.label_texts)
            self._vectorizer = vectorizer
            self._label_matrix = label_matrix

        elif self.method == "sbert":
            self._ensure_query_encoder()
            embeds = self._st_model.encode(
                self.label_texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self._label_embeddings = embeds.astype(np.float32)

        else:
            raise ValueError(f"Unknown shortlist method: {self.method}")

        self._built = True
        save_pickle({
            "label_texts": self.label_texts,
            "train_texts": self.train_texts,
            "config": self.config,
            "cache_namespace": self.cache_namespace,
            "method": self.method,
            "_built": self._built,
            "_vectorizer": self._vectorizer,
            "_label_matrix": self._label_matrix,
            "_label_embeddings": self._label_embeddings,
        }, self.cache_path)
        return self

    def topk(self, text: str, k: Optional[int] = None) -> List[int]:
        self.fit()
        k = k or self.k
        if k is None or k >= len(self.label_texts):
            return list(range(len(self.label_texts)))

        if self.method == "random":
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return sorted(rng.choice(len(self.label_texts), size=k, replace=False).astype(int).tolist())

        self._ensure_query_encoder()
        if self.method == "tfidf":
            query = self._vectorizer.transform([text])
            scores = (query @ self._label_matrix.T).toarray().reshape(-1)

        elif self.method == "sbert":
            query = self._st_model.encode(
                [text],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].astype(np.float32)
            scores = np.matmul(self._label_embeddings, query)

        else:
            raise ValueError(f"Unknown shortlist method: {self.method}")

        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return top.astype(int).tolist()

    def topk_many(self, texts: Sequence[str], k: Optional[int] = None, batch_size: int = 64) -> List[List[int]]:
        self.fit()
        k = k or self.k
        if k is None or k >= len(self.label_texts):
            return [list(range(len(self.label_texts))) for _ in texts]

        if self.method == "random":
            return [self.topk(text, k=k) for text in texts]

        self._ensure_query_encoder()
        if self.method == "tfidf":
            query = self._vectorizer.transform(texts)
            rows = (query @ self._label_matrix.T).toarray()
        else:
            embeddings = self._st_model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            rows = embeddings @ self._label_embeddings.T

        all_shortlists = []
        for row in rows:
            top = np.argpartition(row, -k)[-k:]
            top = top[np.argsort(row[top])[::-1]]
            all_shortlists.append(top.astype(int).tolist())
        return all_shortlists


def compute_shortlists(
    bundle: DatasetBundle,
    split_name: str,
    config: ShortlistConfig,
    force_recompute: bool = False,
) -> List[List[int]]:
    texts = {"val": bundle.val_texts, "test": bundle.test_texts, "train": bundle.train_texts}[split_name]
    if config.k is None or config.k >= len(bundle.label_names):
        return [list(range(len(bundle.label_names))) for _ in texts]

    key = {
        "dataset": bundle.name,
        "split": split_name,
        "config": asdict(config),
        "label_count": len(bundle.label_names),
        "num_examples": len(texts),
    }
    path = CACHE_DIR / "shortlists" / f"{bundle.name}_{split_name}_{config_hash(key)}.pkl"
    if path.exists() and not force_recompute and not OVERWRITE_CACHE:
        return load_pickle(path)

    retriever = ShortlistRetriever(
        label_texts=bundle.label_texts,
        train_texts=bundle.train_texts,
        config=config,
        cache_namespace=f"{bundle.name}_{split_name}",
    )
    shortlists = retriever.topk_many(texts, k=config.k, batch_size=64)
    save_pickle(shortlists, path)
    return shortlists




def truncate_token_ids(token_ids: List[int], max_length: int, strategy: str = "head") -> List[int]:
    if max_length is None or len(token_ids) <= max_length:
        return token_ids
    if strategy == "head":
        return token_ids[:max_length]
    if strategy == "tail":
        return token_ids[-max_length:]
    if strategy in {"head_tail", "head+tail"}:
        head = max_length // 2
        tail = max_length - head
        return token_ids[:head] + token_ids[-tail:]
    if strategy == "middle":
        start = (len(token_ids) - max_length) // 2
        return token_ids[start:start + max_length]
    raise ValueError(f"Unknown truncation strategy: {strategy}")


def prepare_text_for_tokenizer(
    tokenizer: Any,
    text: str,
    max_length: int,
    strategy: str = "head",
) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = truncate_token_ids(token_ids, max_length=max_length, strategy=strategy)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class AnswerSlotPrompt:
    def __init__(
        self,
        tokenizer: Any,
        document: str,
        candidate_indices: List[int],
        label_texts: List[str],
        pos_token_id: int,
        neg_token_id: int,
        mask_token_id: int,
        prompt_config: PromptConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.document = document
        self.candidate_indices = list(candidate_indices)
        self.label_texts = label_texts
        self.pos_token_id = int(pos_token_id)
        self.neg_token_id = int(neg_token_id)
        self.mask_token_id = int(mask_token_id)
        self.prompt_config = prompt_config

        instruction = PROMPT_INSTRUCTIONS[prompt_config.instruction_key]
        doc_intro_ids = tokenizer.encode(instruction + "\n\nDocument:\n", add_special_tokens=False)
        labels_intro_ids = tokenizer.encode("\n\nCandidate labels:\n", add_special_tokens=False)
        label_line_ids = []
        for rank, label_idx in enumerate(self.candidate_indices):
            label_text = self.label_texts[label_idx]
            label_line_ids.extend(tokenizer.encode(f"{rank + 1}. {label_text}\n", add_special_tokens=False))
        answer_intro_ids = tokenizer.encode("\nAnswers:", add_special_tokens=False)

        sep_ids = tokenizer.encode(prompt_config.answer_separator, add_special_tokens=False)
        if len(sep_ids) == 0:
            raise ValueError("Answer separator must tokenize to at least one token.")
        self.separator_ids = sep_ids

        doc_ids_full = tokenizer.encode(maybe_trim_text(document), add_special_tokens=False)

        reserved = len(doc_intro_ids) + len(labels_intro_ids) + len(label_line_ids) + len(answer_intro_ids)
        reserved += len(self.candidate_indices) + (max(0, len(self.candidate_indices) - 1) * len(self.separator_ids))
        doc_budget = max(16, prompt_config.max_context_tokens - reserved)
        doc_ids = truncate_token_ids(doc_ids_full, max_length=doc_budget, strategy=prompt_config.truncation_strategy)

        self.prefix_ids = doc_intro_ids + doc_ids + labels_intro_ids + label_line_ids + answer_intro_ids
        self.template_ids = list(self.prefix_ids)
        self.slot_positions = []

        for i in range(len(self.candidate_indices)):
            self.template_ids.append(self.neg_token_id)
            self.slot_positions.append(len(self.template_ids) - 1)
            if i < len(self.candidate_indices) - 1:
                self.template_ids.extend(self.separator_ids)

    @property
    def num_slots(self) -> int:
        return len(self.slot_positions)

    def all_masked_ids(self) -> List[int]:
        ids = list(self.template_ids)
        for pos in self.slot_positions:
            ids[pos] = self.mask_token_id
        return ids

    def filled_ids(self, assignments: Sequence[int]) -> List[int]:
        ids = list(self.template_ids)
        for value, pos in zip(assignments, self.slot_positions):
            ids[pos] = self.pos_token_id if int(value) == 1 else self.neg_token_id
        return ids

    def sample_extra_mask_slots(
        self,
        query_slot: int,
        strategy: str,
        rng: np.random.Generator,
        probability: float = 0.5,
    ) -> List[int]:
        others = [i for i in range(self.num_slots) if i != query_slot]
        if strategy in {"query_only", "none"}:
            return []
        if strategy == "all_other_slots":
            return others
        if strategy == "uniform_count":
            r = int(rng.integers(0, len(others) + 1))
            if r == 0:
                return []
            chosen = rng.choice(others, size=r, replace=False)
            return sorted(int(x) for x in chosen.tolist())
        if strategy == "bernoulli":
            chosen = [i for i in others if rng.random() < probability]
            return sorted(chosen)
        raise ValueError(f"Unknown remask strategy: {strategy}")

    def masked_sequences_for_queries(
        self,
        assignments: Sequence[int],
        query_indices: Sequence[int],
        mc_samples: int,
        remask_strategy: str,
        remask_probability: float,
        seed: int,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        rng = make_rng(seed)
        filled = self.filled_ids(assignments)
        sequences: List[List[int]] = []
        metadata: List[Tuple[int, int]] = []
        for q in query_indices:
            for sample_id in range(mc_samples):
                ids = list(filled)
                mask_slots = {int(q)}
                mask_slots.update(self.sample_extra_mask_slots(
                    query_slot=int(q),
                    strategy=remask_strategy,
                    rng=rng,
                    probability=remask_probability,
                ))
                for slot in mask_slots:
                    ids[self.slot_positions[slot]] = self.mask_token_id
                sequences.append(ids)
                metadata.append((int(q), int(sample_id)))
        return sequences, metadata


def reorder_candidates(candidate_indices: List[int], randomize: bool, seed: int) -> List[int]:
    candidate_indices = list(candidate_indices)
    if not randomize:
        return candidate_indices
    rng = make_rng(seed)
    perm = rng.permutation(len(candidate_indices))
    return [candidate_indices[i] for i in perm]


class MaskedDiffusionBackbone:
    def __init__(self, backbone_name: str):
        self.backbone_name = backbone_name
        cfg = BACKBONE_REGISTRY[backbone_name]

        tokenizer_id = cfg["tokenizer_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=cfg["trust_remote_code"], use_fast=False)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        dtype_name = cfg["default_dtype"]
        torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32

        # Optional from_pretrained kwargs (used by BD3-LMs to force the
        # SDPA attention backend on accelerators that lack a working
        # flash-attn build, e.g. RTX 5090 sm_120 in our setting).
        extra_kwargs = dict(cfg.get("from_pretrained_kwargs", {}))

        if cfg["loader"] == "maskedlm":
            self.model = AutoModelForMaskedLM.from_pretrained(
                cfg["model_id"],
                trust_remote_code=cfg["trust_remote_code"],
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )
        else:
            self.model = AutoModel.from_pretrained(
                cfg["model_id"],
                trust_remote_code=cfg["trust_remote_code"],
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )

        self.model.eval()
        self.model.to(DEVICE)

        self.mask_token_id = self._infer_mask_token_id()
        self.max_context_tokens = int(cfg.get("max_context_tokens", 1024))

    def _infer_mask_token_id(self) -> int:
        candidates = [
            getattr(self.tokenizer, "mask_token_id", None),
            getattr(getattr(self.model, "config", None), "mask_token_id", None),
        ]
        for cand in candidates:
            if cand is not None and int(cand) >= 0:
                return int(cand)

        try:
            vocab_size = int(self.model.get_input_embeddings().weight.shape[0])
        except Exception:
            vocab_size = None

        # MDLM typically appends a mask token beyond the base GPT-2 vocabulary.
        if self.backbone_name == "mdlm" and vocab_size is not None:
            return vocab_size - 1

        # LLaDA fallback used in released code and model card discussions.
        for special in ["<mask>", "<|mask|>", "[MASK]"]:
            try:
                idx = self.tokenizer.convert_tokens_to_ids(special)
                if idx is not None and idx >= 0:
                    return int(idx)
            except Exception:
                pass

        if self.backbone_name == "llada":
            return 126336

        raise RuntimeError("Could not infer mask token id for diffusion backbone.")

    def verbalizer_token_ids(self, pair: Tuple[str, str]) -> Optional[Tuple[int, int]]:
        pos_ids = self.tokenizer.encode(pair[0], add_special_tokens=False)
        neg_ids = self.tokenizer.encode(pair[1], add_special_tokens=False)
        if len(pos_ids) == 1 and len(neg_ids) == 1:
            return int(pos_ids[0]), int(neg_ids[0])
        return None

    def pad_batch(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in sequences)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        for seq in sequences:
            padded = seq + [pad_id] * (max_len - len(seq))
            mask = [1] * len(seq) + [0] * (max_len - len(seq))
            input_ids.append(padded)
            attention_mask.append(mask)
        return (
            torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
            torch.tensor(attention_mask, dtype=torch.long, device=DEVICE),
        )

    @torch.inference_mode()
    def forward_logits(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw logits in the model's native dtype (bf16/fp32) and the attention mask.

        We deliberately do **not** materialise a float32 log_softmax tensor here — that
        copy is the largest single tensor in the pipeline (batch * seq * vocab) and was
        the OOM source for LLaDA-8B at large query batches.  Slot extraction in
        :meth:`score_unary_all_masked` and :meth:`score_queries` happens row-by-row in
        float32 after the slice.
        """
        input_ids, attention_mask = self.pad_batch(sequences)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits, attention_mask

    @torch.inference_mode()
    def forward_log_probs(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backwards-compatible wrapper that returns full log-probs (uses more memory)."""
        logits, attention_mask = self.forward_logits(sequences)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return log_probs, attention_mask

    @staticmethod
    def _slot_log_probs(logits_row: torch.Tensor, slot_positions: List[int],
                        pos_id: int, neg_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute log p(pos), log p(neg) at every slot position from a single row of logits.

        ``logits_row`` is shape ``[seq, vocab]`` in whatever dtype the model returns.
        We slice the seq axis first (cheap) and only then upcast to float32 for the
        softmax — keeping peak memory tiny.
        """
        rows = logits_row[slot_positions, :].float()  # [num_slots, vocab]
        log_probs = torch.log_softmax(rows, dim=-1)
        return (
            log_probs[:, pos_id].detach().cpu().numpy(),
            log_probs[:, neg_id].detach().cpu().numpy(),
        )

    def score_unary_all_masked(
        self,
        prompts: List[AnswerSlotPrompt],
        batch_size: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pos_rows: List[np.ndarray] = []
        neg_rows: List[np.ndarray] = []

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            sequences = [p.all_masked_ids() for p in batch_prompts]
            logits, _ = self.forward_logits(sequences)
            for i, prompt in enumerate(batch_prompts):
                pos_arr, neg_arr = self._slot_log_probs(
                    logits[i], prompt.slot_positions, prompt.pos_token_id, prompt.neg_token_id
                )
                pos_rows.append(pos_arr)
                neg_rows.append(neg_arr)

        return pos_rows, neg_rows

    def score_queries(
        self,
        prompt: AnswerSlotPrompt,
        assignments: Sequence[int],
        query_indices: Optional[Sequence[int]],
        mc_samples: int,
        remask_strategy: str,
        remask_probability: float,
        batch_size: int,
        seed: int,
    ) -> Dict[int, Tuple[float, float]]:
        if query_indices is None:
            query_indices = list(range(prompt.num_slots))
        sequences, metadata = prompt.masked_sequences_for_queries(
            assignments=assignments,
            query_indices=query_indices,
            mc_samples=mc_samples,
            remask_strategy=remask_strategy,
            remask_probability=remask_probability,
            seed=seed,
        )
        pos_scores: Dict[int, List[float]] = defaultdict(list)
        neg_scores: Dict[int, List[float]] = defaultdict(list)

        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start:start + batch_size]
            batch_meta = metadata[start:start + batch_size]
            logits, _ = self.forward_logits(batch_seqs)
            for b, (q, _) in enumerate(batch_meta):
                slot_logits = logits[b, prompt.slot_positions[q], :].float()
                slot_log_probs = torch.log_softmax(slot_logits, dim=-1)
                pos_scores[q].append(float(slot_log_probs[prompt.pos_token_id].item()))
                neg_scores[q].append(float(slot_log_probs[prompt.neg_token_id].item()))

        return {q: (float(np.mean(pos_scores[q])), float(np.mean(neg_scores[q]))) for q in query_indices}


def build_prompt_for_example(
    backbone: MaskedDiffusionBackbone,
    bundle: DatasetBundle,
    text: str,
    candidate_indices: List[int],
    verbalizer_pair: Tuple[str, str],
    prompt_config: PromptConfig,
    seed: int,
) -> AnswerSlotPrompt:
    ordered_candidates = reorder_candidates(candidate_indices, prompt_config.randomize_label_order, seed=seed)
    verbalizer_ids = backbone.verbalizer_token_ids(verbalizer_pair)
    if verbalizer_ids is None:
        raise ValueError(f"Verbalizer pair {verbalizer_pair} is not single-token for {backbone.backbone_name}.")
    pos_id, neg_id = verbalizer_ids
    return AnswerSlotPrompt(
        tokenizer=backbone.tokenizer,
        document=text,
        candidate_indices=ordered_candidates,
        label_texts=bundle.label_texts,
        pos_token_id=pos_id,
        neg_token_id=neg_id,
        mask_token_id=backbone.mask_token_id,
        prompt_config=prompt_config,
    )


def build_dllm_cache_prefix(
    bundle: DatasetBundle,
    backbone_name: str,
    verbalizer_pair: Tuple[str, str],
    prompt_config: PromptConfig,
    shortlist_config: ShortlistConfig,
) -> str:
    key = {
        "dataset": bundle.name,
        "backbone": backbone_name,
        "verbalizer_pair": verbalizer_pair,
        "prompt_config": asdict(prompt_config),
        "shortlist_config": asdict(shortlist_config),
    }
    return f"{bundle.name}_{backbone_name}_{config_hash(key)}"


def compute_dllm_unary_scores(
    backbone: MaskedDiffusionBackbone,
    bundle: DatasetBundle,
    split_name: str,
    shortlists: List[List[int]],
    shortlist_config: ShortlistConfig,
    verbalizer_pair: Tuple[str, str],
    prompt_config: PromptConfig,
    inference_config: InferenceConfig,
    use_cache: bool = True,
) -> np.ndarray:
    texts = {"val": bundle.val_texts, "test": bundle.test_texts}[split_name]
    num_labels = len(bundle.label_names)
    cache_prefix = build_dllm_cache_prefix(bundle, backbone.backbone_name, verbalizer_pair, prompt_config, shortlist_config)
    # Cache the unary scores by the *unary*-only relevant inference settings so that
    # changing JSR-only knobs (refine_sweeps, mc_samples_refine) doesn't invalidate
    # them. mc_samples_refine controls the JSR step that runs *after* this function.
    cache_key = {
        "unary_mode": inference_config.unary_mode,
        "mc_samples_unary": inference_config.mc_samples_unary,
        "remask_strategy": inference_config.remask_strategy,
        "remask_probability": inference_config.remask_probability,
        "shortlist_config": asdict(shortlist_config),
    }
    path = CACHE_DIR / "dllm_unary" / f"{cache_prefix}_{split_name}_{config_hash(cache_key)}.npz"

    if path.exists() and use_cache and not OVERWRITE_CACHE:
        return load_npz(path)["scores"]

    scores = np.full((len(texts), num_labels), -20.0, dtype=np.float32)

    if inference_config.unary_mode == "all_masked":
        prompts = []
        for i, text in enumerate(texts):
            prompts.append(
                build_prompt_for_example(
                    backbone=backbone,
                    bundle=bundle,
                    text=text,
                    candidate_indices=shortlists[i],
                    verbalizer_pair=verbalizer_pair,
                    prompt_config=prompt_config,
                    seed=GLOBAL_SEED + i,
                )
            )
        pos_rows, neg_rows = backbone.score_unary_all_masked(prompts, batch_size=inference_config.unary_batch_size)
        for i, prompt in enumerate(prompts):
            logits = pos_rows[i] - neg_rows[i]
            for slot_id, label_idx in enumerate(prompt.candidate_indices):
                scores[i, label_idx] = logits[slot_id]

    elif inference_config.unary_mode == "negative_fill_mc":
        for i, text in enumerate(tqdm(texts, desc=f"{bundle.name}:{split_name}:dllm-unary")):
            prompt = build_prompt_for_example(
                backbone=backbone,
                bundle=bundle,
                text=text,
                candidate_indices=shortlists[i],
                verbalizer_pair=verbalizer_pair,
                prompt_config=prompt_config,
                seed=GLOBAL_SEED + i,
            )
            base_assignments = [0] * prompt.num_slots
            query_scores = backbone.score_queries(
                prompt=prompt,
                assignments=base_assignments,
                query_indices=list(range(prompt.num_slots)),
                mc_samples=inference_config.mc_samples_unary,
                remask_strategy=inference_config.remask_strategy,
                remask_probability=inference_config.remask_probability,
                batch_size=inference_config.query_batch_size,
                seed=inference_config.seed + i,
            )
            for slot_id, label_idx in enumerate(prompt.candidate_indices):
                pos, neg = query_scores[slot_id]
                scores[i, label_idx] = pos - neg
    else:
        raise ValueError(f"Unknown unary mode: {inference_config.unary_mode}")

    save_npz(path, scores=scores)
    return scores


def search_best_verbalizer_pair(
    backbone: MaskedDiffusionBackbone,
    bundle: DatasetBundle,
    shortlists_val: List[List[int]],
    prompt_config: PromptConfig,
    shortlist_config: ShortlistConfig,
    calibration_config: CalibrationConfig,
    max_val_examples: int = 256,
) -> Tuple[str, str]:
    y_val = labels_to_multihot(bundle.val_labels, len(bundle.label_names))
    if len(bundle.val_texts) > max_val_examples:
        idx = np.arange(len(bundle.val_texts))[:max_val_examples]
        val_texts = [bundle.val_texts[i] for i in idx]
        val_labels = [bundle.val_labels[i] for i in idx]
        shortlists = [shortlists_val[i] for i in idx]
        y_small = labels_to_multihot(val_labels, len(bundle.label_names))
    else:
        idx = np.arange(len(bundle.val_texts))
        val_texts = bundle.val_texts
        shortlists = shortlists_val
        y_small = y_val

    temp_bundle = DatasetBundle(
        name=bundle.name,
        train_texts=bundle.train_texts,
        val_texts=val_texts,
        test_texts=bundle.test_texts,
        train_labels=bundle.train_labels,
        val_labels=[bundle.val_labels[i] for i in idx],
        test_labels=bundle.test_labels,
        label_names=bundle.label_names,
        label_texts=bundle.label_texts,
        hypothesis_template=bundle.hypothesis_template,
        metadata=bundle.metadata,
    )

    best_pair = None
    best_score = -1.0
    for pair in VERBALIZER_CANDIDATES:
        if backbone.verbalizer_token_ids(pair) is None:
            continue
        inf_cfg = InferenceConfig(
            unary_mode="all_masked",
            mc_samples_unary=1,
            mc_samples_refine=1,
            refine_sweeps=0,
            query_batch_size=16,
            unary_batch_size=2 if backbone.backbone_name == "llada" else 8,
        )
        scores = compute_dllm_unary_scores(
            backbone=backbone,
            bundle=temp_bundle,
            split_name="val",
            shortlists=shortlists,
            shortlist_config=shortlist_config,
            verbalizer_pair=pair,
            prompt_config=prompt_config,
            inference_config=inf_cfg,
            use_cache=True,
        )
        calib = tune_temperature_and_threshold(scores, y_small, calibration_config, scores_are_logits=True)
        pred, _ = predict_from_scores(scores, calib, scores_are_logits=True)
        micro = f1_score(y_small, pred, average="micro", zero_division=0)
        if micro > best_score:
            best_score = micro
            best_pair = pair

    if best_pair is None:
        raise RuntimeError(f"No valid single-token verbalizer pair was found for {backbone.backbone_name}.")
    return best_pair


def run_dllm_setscore(
    bundle: DatasetBundle,
    backbone_name: str,
    prompt_config: PromptConfig,
    shortlist_config: ShortlistConfig,
    inference_config: InferenceConfig,
    calibration_config: CalibrationConfig,
    verbalizer_pair: Optional[Tuple[str, str]] = None,
    method_name: Optional[str] = None,
) -> Dict[str, Any]:
    backbone = MaskedDiffusionBackbone(backbone_name)

    if shortlist_config.k is None and DATASET_DEFAULTS.get(bundle.name, {}).get("shortlist_k") is not None:
        shortlist_config = ShortlistConfig(
            method=shortlist_config.method,
            k=DATASET_DEFAULTS[bundle.name]["shortlist_k"],
            retriever_model=shortlist_config.retriever_model,
            fit_on_train_texts=shortlist_config.fit_on_train_texts,
            seed=shortlist_config.seed,
        )

    shortlists_val = compute_shortlists(bundle, "val", shortlist_config)
    shortlists_test = compute_shortlists(bundle, "test", shortlist_config)

    if verbalizer_pair is None:
        verbalizer_pair = search_best_verbalizer_pair(
            backbone=backbone,
            bundle=bundle,
            shortlists_val=shortlists_val,
            prompt_config=prompt_config,
            shortlist_config=shortlist_config,
            calibration_config=calibration_config,
        )

    y_val = labels_to_multihot(bundle.val_labels, len(bundle.label_names))
    y_test = labels_to_multihot(bundle.test_labels, len(bundle.label_names))

    val_scores = compute_dllm_unary_scores(
        backbone=backbone,
        bundle=bundle,
        split_name="val",
        shortlists=shortlists_val,
        shortlist_config=shortlist_config,
        verbalizer_pair=verbalizer_pair,
        prompt_config=prompt_config,
        inference_config=inference_config,
        use_cache=True,
    )
    calibrator = tune_temperature_and_threshold(val_scores, y_val, calibration_config, scores_are_logits=True)
    val_pred, val_prob = predict_from_scores(
        val_scores, calibrator, scores_are_logits=True
    )

    reset_cuda_stats()
    t0 = timer()
    test_scores = compute_dllm_unary_scores(
        backbone=backbone,
        bundle=bundle,
        split_name="test",
        shortlists=shortlists_test,
        shortlist_config=shortlist_config,
        verbalizer_pair=verbalizer_pair,
        prompt_config=prompt_config,
        inference_config=inference_config,
        use_cache=True,
    )
    y_pred, y_prob = predict_from_scores(test_scores, calibrator, scores_are_logits=True)

    if inference_config.refine_sweeps > 0:
        for i, text in enumerate(tqdm(bundle.test_texts, desc=f"{bundle.name}:{backbone_name}:refine")):
            prompt = build_prompt_for_example(
                backbone=backbone,
                bundle=bundle,
                text=text,
                candidate_indices=shortlists_test[i],
                verbalizer_pair=verbalizer_pair,
                prompt_config=prompt_config,
                seed=GLOBAL_SEED + i,
            )
            current = [int(y_pred[i, label_idx]) for label_idx in prompt.candidate_indices]
            for sweep in range(inference_config.refine_sweeps):
                query_scores = backbone.score_queries(
                    prompt=prompt,
                    assignments=current,
                    query_indices=list(range(prompt.num_slots)),
                    mc_samples=inference_config.mc_samples_refine,
                    remask_strategy=inference_config.remask_strategy,
                    remask_probability=inference_config.remask_probability,
                    batch_size=inference_config.query_batch_size,
                    seed=inference_config.seed + i * 17 + sweep,
                )
                current = [1 if query_scores[q][0] >= query_scores[q][1] else 0 for q in range(prompt.num_slots)]
            for slot_id, label_idx in enumerate(prompt.candidate_indices):
                y_pred[i, label_idx] = current[slot_id]

    elapsed = timer() - t0
    peak_memory = current_peak_memory_mb()

    metrics = compute_multilabel_metrics(y_test, y_pred, y_prob, ece_bins=calibration_config.ece_bins)

    run_name = method_name or f"{backbone_name}_{'jsr' if inference_config.refine_sweeps > 0 else 'unary'}"
    run_cfg = {
        "dataset": bundle.name,
        "method": run_name,
        "backbone": backbone_name,
        "prompt_config": asdict(prompt_config),
        "shortlist_config": asdict(shortlist_config),
        "inference_config": asdict(inference_config),
        "calibration_config": asdict(calibration_config),
        "verbalizer_pair": verbalizer_pair,
        "calibrator": calibrator,
    }
    run_id = config_hash(run_cfg)
    artifact_dir = RESULTS_DIR / "predictions" / f"{bundle.name}_{run_name}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_npz(
        artifact_dir / "predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        scores=test_scores,
        val_y_true=y_val,
        val_y_pred=val_pred,
        val_y_prob=val_prob,
        val_scores=val_scores,
    )
    save_json(run_cfg, artifact_dir / "config.json")

    result = {
        "dataset": bundle.name,
        "method": run_name,
        "backbone": backbone_name,
        "shortlist_method": shortlist_config.method,
        "shortlist_k": shortlist_config.k if shortlist_config.k is not None else len(bundle.label_names),
        "verbalizer_pair": f"{verbalizer_pair[0].strip()}/{verbalizer_pair[1].strip()}",
        "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
        "peak_memory_mb": peak_memory,
        "artifact_dir": str(artifact_dir),
        **metrics,
    }
    write_result_row(result, "main_results")
    return result




class BartMNLIZeroShotRunner:
    def __init__(self, model_name: str = SUPERIVSED_MODEL_REGISTRY["bart_mnli"]):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entailment_id = label2id.get("entailment", 2)
        self.contradiction_id = label2id.get("contradiction", 0)

    @torch.inference_mode()
    def score_pairs(
        self,
        premises: List[str],
        hypotheses: List[str],
        max_length: int,
        batch_size: int = 16,
    ) -> np.ndarray:
        scores = []
        for start in range(0, len(premises), batch_size):
            p = premises[start:start + batch_size]
            h = hypotheses[start:start + batch_size]
            enc = self.tokenizer(
                p,
                h,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = self.model(**enc).logits.float()
            pair_logits = logits[:, [self.contradiction_id, self.entailment_id]]
            probs = torch.softmax(pair_logits, dim=-1)[:, 1]
            scores.append(probs.detach().cpu().numpy())
        return np.concatenate(scores, axis=0)

    def score_dataset(
        self,
        bundle: DatasetBundle,
        split_name: str,
        shortlists: List[List[int]],
        max_length: int,
    ) -> np.ndarray:
        texts = {"val": bundle.val_texts, "test": bundle.test_texts}[split_name]
        num_labels = len(bundle.label_names)
        probs = np.zeros((len(texts), num_labels), dtype=np.float32)

        for i, text in enumerate(tqdm(texts, desc=f"{bundle.name}:{split_name}:bart-mnli")):
            candidate_indices = shortlists[i]
            label_texts = [bundle.label_texts[idx] for idx in candidate_indices]
            hypotheses = [bundle.hypothesis_template.format(label_text) for label_text in label_texts]
            premise = prepare_text_for_tokenizer(self.tokenizer, text, max_length=max_length, strategy="head_tail")
            pair_scores = self.score_pairs(
                premises=[premise] * len(hypotheses),
                hypotheses=hypotheses,
                max_length=max_length,
                batch_size=min(16, len(hypotheses)),
            )
            probs[i, candidate_indices] = pair_scores.astype(np.float32)

        return probs


def run_bart_mnli_zero_shot(
    bundle: DatasetBundle,
    shortlist_config: ShortlistConfig,
    calibration_config: CalibrationConfig,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    max_length = max_length or DATASET_DEFAULTS[bundle.name]["bart_max_length"]
    if shortlist_config.k is None and DATASET_DEFAULTS.get(bundle.name, {}).get("shortlist_k") is not None:
        shortlist_config = ShortlistConfig(
            method=shortlist_config.method,
            k=DATASET_DEFAULTS[bundle.name]["shortlist_k"],
            retriever_model=shortlist_config.retriever_model,
            fit_on_train_texts=shortlist_config.fit_on_train_texts,
            seed=shortlist_config.seed,
        )

    shortlists_val = compute_shortlists(bundle, "val", shortlist_config)
    shortlists_test = compute_shortlists(bundle, "test", shortlist_config)
    runner = BartMNLIZeroShotRunner()

    y_val = labels_to_multihot(bundle.val_labels, len(bundle.label_names))
    y_test = labels_to_multihot(bundle.test_labels, len(bundle.label_names))

    val_probs = runner.score_dataset(bundle, "val", shortlists_val, max_length=max_length)
    calibrator = tune_temperature_and_threshold(val_probs, y_val, calibration_config, scores_are_logits=False)
    val_pred, val_y_prob = predict_from_scores(
        val_probs, calibrator, scores_are_logits=False
    )

    reset_cuda_stats()
    t0 = timer()
    test_probs = runner.score_dataset(bundle, "test", shortlists_test, max_length=max_length)
    elapsed = timer() - t0
    peak_memory = current_peak_memory_mb()

    y_pred, y_prob = predict_from_scores(test_probs, calibrator, scores_are_logits=False)
    metrics = compute_multilabel_metrics(y_test, y_pred, y_prob, ece_bins=calibration_config.ece_bins)

    run_cfg = {
        "dataset": bundle.name,
        "method": "bart_mnli_zero_shot",
        "max_length": max_length,
        "shortlist_config": asdict(shortlist_config),
        "calibration_config": asdict(calibration_config),
        "calibrator": calibrator,
    }
    run_id = config_hash(run_cfg)
    artifact_dir = RESULTS_DIR / "predictions" / f"{bundle.name}_bart_mnli_zero_shot_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_npz(
        artifact_dir / "predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        val_y_true=y_val,
        val_y_pred=val_pred,
        val_y_prob=val_y_prob,
    )
    save_json(run_cfg, artifact_dir / "config.json")

    result = {
        "dataset": bundle.name,
        "method": "bart_mnli_zero_shot",
        "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
        "peak_memory_mb": peak_memory,
        "shortlist_method": shortlist_config.method,
        "shortlist_k": shortlist_config.k if shortlist_config.k is not None else len(bundle.label_names),
        "artifact_dir": str(artifact_dir),
        **metrics,
    }
    write_result_row(result, "main_results")
    return result




class RobustOVRLogReg:
    def __init__(self, C: float = 4.0, max_iter: int = 2000, class_weight: str = "balanced"):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.models: List[Any] = []
        self.constants: List[Optional[float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustOVRLogReg":
        self.models = []
        self.constants = []
        for j in tqdm(range(y.shape[1]), desc="OVR heads", leave=False):
            target = y[:, j]
            positives = int(target.sum())
            if positives == 0:
                self.models.append(None)
                self.constants.append(0.0)
            elif positives == len(target):
                self.models.append(None)
                self.constants.append(1.0)
            else:
                model = LogisticRegression(
                    C=self.C,
                    max_iter=self.max_iter,
                    solver="liblinear",
                    class_weight=self.class_weight,
                )
                model.fit(X, target)
                self.models.append(model)
                self.constants.append(None)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), len(self.models)), dtype=np.float32)
        for j, (model, constant) in enumerate(zip(self.models, self.constants)):
            if model is None:
                probs[:, j] = float(constant)
            else:
                probs[:, j] = model.predict_proba(X)[:, 1].astype(np.float32)
        return probs


def sample_fewshot_multilabel_subset(
    y_train: np.ndarray,
    shots_per_label: int,
    seed: int = GLOBAL_SEED,
    max_examples: Optional[int] = None,
) -> List[int]:
    rng = np.random.default_rng(seed)
    num_labels = y_train.shape[1]
    label_pos = [np.flatnonzero(y_train[:, j] > 0).astype(int).tolist() for j in range(num_labels)]
    for row in label_pos:
        rng.shuffle(row)

    targets = np.minimum(np.sum(y_train > 0, axis=0).astype(int), shots_per_label)
    counts = np.zeros(num_labels, dtype=int)
    selected: List[int] = []
    selected_set = set()

    def _coverage_gain(i: int) -> int:
        row = y_train[i]
        return int(np.sum((targets > counts) & (row > 0)))

    while np.any(counts < targets):
        deficit_labels = np.flatnonzero(counts < targets)
        label_choice = int(deficit_labels[np.argmin(counts[deficit_labels])])
        candidates = [i for i in label_pos[label_choice] if i not in selected_set]
        if not candidates:
            counts[label_choice] = targets[label_choice]
            continue
        if len(candidates) > 256:
            candidates = candidates[:256]
        best = max(candidates, key=_coverage_gain)
        selected.append(best)
        selected_set.add(best)
        counts += y_train[best].astype(int)
        if max_examples is not None and len(selected) >= max_examples:
            break

    selected = sorted(set(selected))
    return selected


def build_setfit_pairs(
    texts: List[str],
    y: np.ndarray,
    max_pairs: int = 4096,
    seed: int = GLOBAL_SEED,
) -> List[InputExample]:
    if InputExample is None:
        return []
    rng = np.random.default_rng(seed)
    n = len(texts)
    if n < 2:
        return []

    pair_examples: List[InputExample] = []

    # Positive pairs from shared labels.
    label_to_indices = [np.flatnonzero(y[:, j] > 0).astype(int).tolist() for j in range(y.shape[1])]
    for idxs in label_to_indices:
        rng.shuffle(idxs)
        for a, b in zip(idxs[::2], idxs[1::2]):
            pair_examples.append(InputExample(texts=[texts[a], texts[b]], label=1.0))
            if len(pair_examples) >= max_pairs // 2:
                break
        if len(pair_examples) >= max_pairs // 2:
            break

    # Negative pairs without shared labels.
    attempts = 0
    while len(pair_examples) < max_pairs and attempts < max_pairs * 20:
        a, b = rng.choice(n, size=2, replace=False)
        if not np.any((y[a] > 0) & (y[b] > 0)):
            pair_examples.append(InputExample(texts=[texts[a], texts[b]], label=0.0))
        attempts += 1

    return pair_examples


def train_setfit_style_baseline(
    bundle: DatasetBundle,
    shortlist_config: ShortlistConfig,
    calibration_config: CalibrationConfig,
    encoder_name: str = SUPERIVSED_MODEL_REGISTRY["setfit_encoder"],
    shots_per_label: Optional[int] = None,
    contrastive_epochs: int = 1,
    batch_size: int = 16,
) -> Dict[str, Any]:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for the SetFit-style baseline.")

    shots_per_label = shots_per_label or DATASET_DEFAULTS[bundle.name]["fewshot_shots_per_label"]
    y_train = labels_to_multihot(bundle.train_labels, len(bundle.label_names))
    y_val = labels_to_multihot(bundle.val_labels, len(bundle.label_names))
    y_test = labels_to_multihot(bundle.test_labels, len(bundle.label_names))

    subset_idx = sample_fewshot_multilabel_subset(y_train, shots_per_label=shots_per_label, seed=GLOBAL_SEED)
    train_texts_fs = [bundle.train_texts[i] for i in subset_idx]
    y_train_fs = y_train[subset_idx]

    model = SentenceTransformer(encoder_name, device=DEVICE)

    if contrastive_epochs > 0 and InputExample is not None and losses is not None and DataLoader is not None:
        pair_examples = build_setfit_pairs(train_texts_fs, y_train_fs, max_pairs=4096, seed=GLOBAL_SEED)
        if pair_examples:
            train_loader = DataLoader(pair_examples, shuffle=True, batch_size=batch_size)
            train_loss = losses.CosineSimilarityLoss(model)
            model.fit(
                train_objectives=[(train_loader, train_loss)],
                epochs=contrastive_epochs,
                warmup_steps=max(1, len(train_loader) // 10),
                show_progress_bar=True,
            )

    X_train = model.encode(train_texts_fs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    X_val = model.encode(bundle.val_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    X_test = model.encode(bundle.test_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    clf = RobustOVRLogReg(C=4.0, max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train_fs)

    val_probs = clf.predict_proba(X_val)
    calibrator = tune_temperature_and_threshold(val_probs, y_val, calibration_config, scores_are_logits=False)
    val_pred, val_y_prob = predict_from_scores(
        val_probs, calibrator, scores_are_logits=False
    )

    reset_cuda_stats()
    t0 = timer()
    test_probs = clf.predict_proba(X_test)
    elapsed = timer() - t0
    peak_memory = current_peak_memory_mb()

    y_pred, y_prob = predict_from_scores(test_probs, calibrator, scores_are_logits=False)
    metrics = compute_multilabel_metrics(y_test, y_pred, y_prob, ece_bins=calibration_config.ece_bins)

    run_cfg = {
        "dataset": bundle.name,
        "method": "setfit_fewshot",
        "encoder_name": encoder_name,
        "shots_per_label": shots_per_label,
        "contrastive_epochs": contrastive_epochs,
        "subset_size": len(subset_idx),
        "calibrator": calibrator,
        "shortlist_config": asdict(shortlist_config),
    }
    run_id = config_hash(run_cfg)
    artifact_dir = RESULTS_DIR / "predictions" / f"{bundle.name}_setfit_fewshot_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_npz(
        artifact_dir / "predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        val_y_true=y_val,
        val_y_pred=val_pred,
        val_y_prob=val_y_prob,
        subset_idx=np.asarray(subset_idx),
    )
    save_json(run_cfg, artifact_dir / "config.json")

    result = {
        "dataset": bundle.name,
        "method": "setfit_fewshot",
        "fewshot_subset_size": len(subset_idx),
        "shots_per_label": shots_per_label,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
        "peak_memory_mb": peak_memory,
        "artifact_dir": str(artifact_dir),
        **metrics,
    }
    write_result_row(result, "main_results")
    return result




def encode_text_with_strategy(
    tokenizer: Any,
    text: str,
    max_length: int,
    strategy: str = "head",
) -> Dict[str, List[int]]:
    if strategy == "head":
        enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        return {k: list(v) for k, v in enc.items()}

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    token_ids = truncate_token_ids(token_ids, max_length=max_length - n_special, strategy=strategy)
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    attention_mask = [1] * len(input_ids)

    encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
    if hasattr(tokenizer, "create_token_type_ids_from_sequences"):
        try:
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids)
            if token_type_ids is not None:
                encoded["token_type_ids"] = token_type_ids
        except Exception:
            pass
    return encoded


class MultiLabelCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.float32)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def build_encoder_dataset(
    texts: List[str],
    label_lists: List[List[int]],
    tokenizer: Any,
    num_labels: int,
    max_length: int,
    truncation_strategy: str,
) -> Dataset:
    rows = []
    for text, labels in tqdm(list(zip(texts, label_lists)), desc="Tokenizing", leave=False):
        enc = encode_text_with_strategy(tokenizer, text, max_length=max_length, strategy=truncation_strategy)
        enc["labels"] = labels_to_multihot([labels], num_labels=num_labels)[0].astype(np.float32).tolist()
        rows.append(enc)
    return Dataset.from_list(rows)


def train_encoder_multilabel(
    bundle: DatasetBundle,
    method_name: str,
    model_name: str,
    train_config: EncoderTrainConfig,
    calibration_config: CalibrationConfig,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    train_ds = build_encoder_dataset(
        bundle.train_texts,
        bundle.train_labels,
        tokenizer,
        num_labels=len(bundle.label_names),
        max_length=train_config.max_length,
        truncation_strategy=train_config.truncation_strategy,
    )
    val_ds = build_encoder_dataset(
        bundle.val_texts,
        bundle.val_labels,
        tokenizer,
        num_labels=len(bundle.label_names),
        max_length=train_config.max_length,
        truncation_strategy=train_config.truncation_strategy,
    )
    test_ds = build_encoder_dataset(
        bundle.test_texts,
        bundle.test_labels,
        tokenizer,
        num_labels=len(bundle.label_names),
        max_length=train_config.max_length,
        truncation_strategy=train_config.truncation_strategy,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(bundle.label_names),
        problem_type="multi_label_classification",
    )
    if train_config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    out_dir = MODELS_DIR / f"{bundle.name}_{slugify(method_name)}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.eval_batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        num_train_epochs=train_config.epochs,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        bf16=BF16_OK,
        fp16=(torch.cuda.is_available() and not BF16_OK),
        seed=train_config.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=MultiLabelCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()

    val_logits = trainer.predict(val_ds).predictions
    val_probs = expit(val_logits.astype(np.float32))
    y_val = labels_to_multihot(bundle.val_labels, len(bundle.label_names))
    calibrator = tune_temperature_and_threshold(val_probs, y_val, calibration_config, scores_are_logits=False)

    reset_cuda_stats()
    t0 = timer()
    test_logits = trainer.predict(test_ds).predictions
    elapsed = timer() - t0
    peak_memory = current_peak_memory_mb()
    test_probs = expit(test_logits.astype(np.float32))

    y_test = labels_to_multihot(bundle.test_labels, len(bundle.label_names))
    y_pred, y_prob = predict_from_scores(test_probs, calibrator, scores_are_logits=False)
    metrics = compute_multilabel_metrics(y_test, y_pred, y_prob, ece_bins=calibration_config.ece_bins)

    run_cfg = {
        "dataset": bundle.name,
        "method": method_name,
        "model_name": model_name,
        "train_config": asdict(train_config),
        "calibration_config": asdict(calibration_config),
        "calibrator": calibrator,
    }
    run_id = config_hash(run_cfg)
    artifact_dir = RESULTS_DIR / "predictions" / f"{bundle.name}_{slugify(method_name)}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_npz(artifact_dir / "predictions.npz", y_true=y_test, y_pred=y_pred, y_prob=y_prob, logits=test_logits)
    save_json(run_cfg, artifact_dir / "config.json")
    if train_config.save_model:
        trainer.save_model(str(artifact_dir / "model"))

    result = {
        "dataset": bundle.name,
        "method": method_name,
        "model_name": model_name,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
        "peak_memory_mb": peak_memory,
        "artifact_dir": str(artifact_dir),
        **metrics,
    }
    write_result_row(result, "supervised_results")
    return result


def build_t5_targets(bundle: DatasetBundle) -> Tuple[Dict[int, str], Dict[str, int]]:
    y_train = labels_to_multihot(bundle.train_labels, len(bundle.label_names))
    freq = np.asarray(y_train.sum(axis=0)).reshape(-1)
    order = np.argsort(-freq)
    idx_to_text = {int(i): str(bundle.label_texts[i]) for i in range(len(bundle.label_names))}
    norm_to_idx = {normalize_label_text(idx_to_text[int(i)]): int(i) for i in order}
    return idx_to_text, norm_to_idx


def normalize_label_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\- /]+", "", text)
    return text.strip()


def labels_to_t5_target(label_ids: List[int], idx_to_text: Dict[int, str]) -> str:
    texts = [idx_to_text[int(i)] for i in label_ids]
    texts = sorted(texts, key=lambda x: normalize_label_text(x))
    return " ; ".join(texts)


def parse_t5_predictions(pred_text: str, norm_to_idx: Dict[str, int]) -> List[int]:
    parts = [normalize_label_text(p) for p in pred_text.split(";")]
    labels = set()
    for part in parts:
        if not part:
            continue
        if part in norm_to_idx:
            labels.add(norm_to_idx[part])
            continue
        # fuzzy fallback
        from difflib import get_close_matches
        match = get_close_matches(part, list(norm_to_idx.keys()), n=1, cutoff=0.88)
        if match:
            labels.add(norm_to_idx[match[0]])
    return sorted(labels)


def build_t5_dataset(
    bundle: DatasetBundle,
    split_name: str,
    tokenizer: Any,
    idx_to_text: Dict[int, str],
    max_source_length: int,
    max_target_length: int,
) -> Dataset:
    texts = {"train": bundle.train_texts, "val": bundle.val_texts, "test": bundle.test_texts}[split_name]
    labels = {"train": bundle.train_labels, "val": bundle.val_labels, "test": bundle.test_labels}[split_name]
    rows = []
    for text, label_ids in tqdm(list(zip(texts, labels)), desc=f"Preparing {split_name}", leave=False):
        source = "Assign all applicable labels as a semicolon-separated list.\n\nDocument:\n" + maybe_trim_text(text)
        target = labels_to_t5_target(label_ids, idx_to_text)
        model_inputs = tokenizer(source, truncation=True, max_length=max_source_length)
        try:
            label_inputs = tokenizer(text_target=target, truncation=True, max_length=max_target_length)
        except TypeError:
            with tokenizer.as_target_tokenizer():
                label_inputs = tokenizer(target, truncation=True, max_length=max_target_length)
        model_inputs["labels"] = label_inputs["input_ids"]
        rows.append(model_inputs)
    return Dataset.from_list(rows)


def train_t5_text_to_set(
    bundle: DatasetBundle,
    model_name: str,
    train_config: T5TrainConfig,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    idx_to_text, norm_to_idx = build_t5_targets(bundle)
    train_ds = build_t5_dataset(bundle, "train", tokenizer, idx_to_text, train_config.max_source_length, train_config.max_target_length)
    val_ds = build_t5_dataset(bundle, "val", tokenizer, idx_to_text, train_config.max_source_length, train_config.max_target_length)
    test_ds = build_t5_dataset(bundle, "test", tokenizer, idx_to_text, train_config.max_source_length, train_config.max_target_length)

    out_dir = MODELS_DIR / f"{bundle.name}_t5_text_to_set"
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.eval_batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        num_train_epochs=train_config.epochs,
        predict_with_generate=True,
        generation_max_length=train_config.max_target_length,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        bf16=BF16_OK,
        fp16=(torch.cuda.is_available() and not BF16_OK),
        seed=train_config.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()

    reset_cuda_stats()
    t0 = timer()
    pred_out = trainer.predict(test_ds, max_length=train_config.max_target_length)
    elapsed = timer() - t0
    peak_memory = current_peak_memory_mb()

    pred_ids = pred_out.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    y_test = labels_to_multihot(bundle.test_labels, len(bundle.label_names))
    pred_labels = [parse_t5_predictions(text, norm_to_idx) for text in pred_texts]
    y_pred = labels_to_multihot(pred_labels, len(bundle.label_names))
    metrics = compute_multilabel_metrics(y_test, y_pred, y_prob=None)

    run_cfg = {
        "dataset": bundle.name,
        "method": "t5_text_to_set",
        "model_name": model_name,
        "train_config": asdict(train_config),
    }
    run_id = config_hash(run_cfg)
    artifact_dir = RESULTS_DIR / "predictions" / f"{bundle.name}_t5_text_to_set_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_npz(artifact_dir / "predictions.npz", y_true=y_test, y_pred=y_pred)
    with open(artifact_dir / "generated_predictions.txt", "w", encoding="utf-8") as f:
        for text in pred_texts:
            f.write(text + "\n")
    save_json(run_cfg, artifact_dir / "config.json")
    if train_config.save_model:
        trainer.save_model(str(artifact_dir / "model"))

    result = {
        "dataset": bundle.name,
        "method": "t5_text_to_set",
        "model_name": model_name,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
        "peak_memory_mb": peak_memory,
        "artifact_dir": str(artifact_dir),
        **metrics,
    }
    write_result_row(result, "supervised_results")
    return result




METHOD_DISPLAY_NAMES = {
    "bart_mnli_zero_shot": "BART-MNLI zero-shot",
    "setfit_fewshot": "SetFit few-shot",
    "mdlm_unary": "MDLM unary",
    "mdlm_jsr": "MDLM + JSR",
    "llada_unary": "LLaDA unary",
    "llada_jsr": "LLaDA + JSR",
    "bert_base_supervised": "BERT-base supervised",
    "roberta_base_supervised": "RoBERTa-base supervised",
    "t5_text_to_set": "T5 text-to-set",
}


def export_results_tables() -> Dict[str, Path]:
    main_df = read_jsonl(RESULTS_DIR / "main_results.jsonl")
    sup_df = read_jsonl(RESULTS_DIR / "supervised_results.jsonl")
    out_paths: Dict[str, Path] = {}

    if not main_df.empty:
        main_df["method_display"] = main_df["method"].map(METHOD_DISPLAY_NAMES).fillna(main_df["method"])
        main_csv = TABLES_DIR / "main_results.csv"
        main_df.to_csv(main_csv, index=False)
        out_paths["main_csv"] = main_csv

        pivot = main_df.pivot_table(
            index="method_display",
            columns="dataset",
            values=["micro_f1", "macro_f1"],
            aggfunc="mean",
        )
        pivot.columns = [f"{metric}_{dataset}" for metric, dataset in pivot.columns]
        pivot = pivot.reset_index()
        main_tex = TABLES_DIR / "main_results.tex"
        with open(main_tex, "w", encoding="utf-8") as f:
            f.write(pivot.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=False))
        out_paths["main_tex"] = main_tex

    if not sup_df.empty:
        sup_df["method_display"] = sup_df["method"].map(METHOD_DISPLAY_NAMES).fillna(sup_df["method"])
        sup_csv = TABLES_DIR / "supervised_results.csv"
        sup_df.to_csv(sup_csv, index=False)
        out_paths["supervised_csv"] = sup_csv

        pivot = sup_df.pivot_table(
            index="method_display",
            columns="dataset",
            values=["micro_f1", "macro_f1"],
            aggfunc="mean",
        )
        pivot.columns = [f"{metric}_{dataset}" for metric, dataset in pivot.columns]
        pivot = pivot.reset_index()
        sup_tex = TABLES_DIR / "supervised_results.tex"
        with open(sup_tex, "w", encoding="utf-8") as f:
            f.write(pivot.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=False))
        out_paths["supervised_tex"] = sup_tex

    return out_paths


def reliability_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            rows.append({"bin_lo": lo, "bin_hi": hi, "confidence": np.nan, "accuracy": np.nan, "count": 0})
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "confidence": float(y_prob[mask].mean()),
            "accuracy": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def plot_reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path, n_bins: int = 15) -> Path:
    curve = reliability_curve_points(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(5.0, 5.0))
    valid = curve["count"] > 0
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(curve.loc[valid, "confidence"], curve.loc[valid, "accuracy"], marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_pareto_front(results_df: pd.DataFrame, out_path: Path) -> Path:
    plot_df = results_df.dropna(subset=["latency_ms_per_example", "micro_f1"]).copy()
    plt.figure(figsize=(6.0, 4.5))
    plt.scatter(plot_df["latency_ms_per_example"], plot_df["micro_f1"])
    for _, row in plot_df.iterrows():
        plt.annotate(str(row["method"]), (row["latency_ms_per_example"], row["micro_f1"]), fontsize=8)
    plt.xlabel("Latency (ms / example)")
    plt.ylabel("Micro-F1")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def per_label_error_report(bundle: DatasetBundle, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> Path:
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    rows = []
    for i, label_name in enumerate(bundle.label_names):
        rows.append({
            "label_idx": i,
            "label_name": label_name,
            "label_text": bundle.label_texts[i],
            "support": int(supports[i]),
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
        })
    df = pd.DataFrame(rows).sort_values(["support", "f1"], ascending=[False, True])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path




def default_prompt_config(bundle: DatasetBundle) -> PromptConfig:
    return PromptConfig(
        instruction_key="default",
        answer_separator=";",
        include_label_descriptions=True,
        randomize_label_order=False,
        truncation_strategy="head_tail" if bundle.name in {"eurlex57k", "multieurlex_en"} else "head",
        max_context_tokens=DATASET_DEFAULTS[bundle.name]["prompt_max_context_tokens"],
    )


def default_shortlist_config(bundle: DatasetBundle) -> ShortlistConfig:
    return ShortlistConfig(
        method="sbert",
        k=DATASET_DEFAULTS[bundle.name]["shortlist_k"],
        retriever_model=SUPERIVSED_MODEL_REGISTRY["retriever"],
        fit_on_train_texts=True,
        seed=GLOBAL_SEED,
    )


def default_calibration_config(strategy: str = "auto") -> CalibrationConfig:
    """Default to *auto* (val-pick the best strategy among global / labelwise /
    expected_cardinality) so the manuscript headline numbers reflect a fair
    upper-envelope across calibration choices.  Pass an explicit strategy when
    you want a specific calibration cell of the ablation table."""
    return CalibrationConfig(strategy=strategy)


def default_inference_config(backbone_name: str, use_refinement: bool, *, bundle_name: Optional[str] = None) -> InferenceConfig:
    """Per-backbone defaults that fit in 32 GB VRAM and run within a few-hours budget.

    For LLaDA-8B-Base on the RTX 5090 we keep mc_samples_refine=2 / sweeps=1 by default
    so JSR runs in roughly 1 sec / example with k=28 labels.  Ablations override these.
    Long-document, large-shortlist datasets (e.g. EURLEX57K, k=32 with 1024-token
    documents) need smaller query batches because the final linear projection over
    LLaDA's 126k vocab is the OOM bottleneck — every doubling of sequences in the
    batch is +8 GB.
    """
    if backbone_name == "llada":
        long_doc = bundle_name in {"eurlex57k", "multieurlex_en"}
        return InferenceConfig(
            unary_mode="all_masked",
            mc_samples_unary=1,
            mc_samples_refine=2 if use_refinement else 1,
            refine_sweeps=1 if use_refinement else 0,
            remask_strategy="uniform_count",
            remask_probability=0.5,
            unary_batch_size=2 if long_doc else 4,
            query_batch_size=16 if long_doc else 64,
            seed=GLOBAL_SEED,
        )
    return InferenceConfig(
        unary_mode="all_masked",
        mc_samples_unary=1,
        mc_samples_refine=4 if use_refinement else 1,
        refine_sweeps=2 if use_refinement else 0,
        remask_strategy="uniform_count",
        remask_probability=0.5,
        unary_batch_size=8,
        query_batch_size=64,
        seed=GLOBAL_SEED,
    )


def default_encoder_train_config(bundle: DatasetBundle) -> EncoderTrainConfig:
    max_len = DATASET_DEFAULTS[bundle.name]["encoder_max_length"]
    # Aggressive defaults so the supervised stage finishes within a few hours
    # on a single accelerator. The cached predictions allow re-tuning thresholds
    # without retraining.  Long-document datasets (EURLEX57K) only get one pass
    # through the train split.
    epochs = 2 if bundle.name != "eurlex57k" else 1
    batch_size = 16 if bundle.name not in {"eurlex57k", "multieurlex_en"} else 8
    return EncoderTrainConfig(
        max_length=max_len,
        truncation_strategy="head_tail" if bundle.name in {"eurlex57k", "multieurlex_en"} else "head",
        batch_size=batch_size,
        eval_batch_size=batch_size * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        epochs=epochs,
        gradient_accumulation_steps=1,
        save_model=True,  # write the trained encoder under the artifact dir for HF upload
        gradient_checkpointing=(bundle.name in {"eurlex57k", "multieurlex_en"}),
    )


def default_t5_train_config(bundle: DatasetBundle) -> T5TrainConfig:
    return T5TrainConfig(
        max_source_length=DATASET_DEFAULTS[bundle.name]["t5_max_source_length"],
        max_target_length=DATASET_DEFAULTS[bundle.name]["t5_max_target_length"],
        batch_size=16 if bundle.name not in {"eurlex57k", "multieurlex_en"} else 4,
        eval_batch_size=16 if bundle.name not in {"eurlex57k", "multieurlex_en"} else 4,
        learning_rate=3e-4,
        weight_decay=0.01,
        epochs=2 if bundle.name != "eurlex57k" else 1,
        save_model=True,
    )


def run_main_comparison_for_bundle(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    results = []
    shortlist_cfg = default_shortlist_config(bundle)
    calibration_cfg = default_calibration_config("auto")
    prompt_cfg = default_prompt_config(bundle)

    print(f"=== Main comparison: {bundle.name} ===")

    results.append(run_bart_mnli_zero_shot(bundle, shortlist_cfg=shortlist_cfg, calibration_config=calibration_cfg))
    results.append(
        train_setfit_style_baseline(
            bundle=bundle,
            shortlist_config=shortlist_cfg,
            calibration_config=calibration_cfg,
            shots_per_label=DATASET_DEFAULTS[bundle.name]["fewshot_shots_per_label"],
            contrastive_epochs=1,
        )
    )

    results.append(
        run_dllm_setscore(
            bundle=bundle,
            backbone_name="mdlm",
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=default_inference_config("mdlm", use_refinement=False),
            calibration_config=calibration_cfg,
            method_name="mdlm_unary",
        )
    )
    results.append(
        run_dllm_setscore(
            bundle=bundle,
            backbone_name="mdlm",
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=default_inference_config("mdlm", use_refinement=True),
            calibration_config=calibration_cfg,
            method_name="mdlm_jsr",
        )
    )
    results.append(
        run_dllm_setscore(
            bundle=bundle,
            backbone_name="llada",
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=default_inference_config("llada", use_refinement=False),
            calibration_config=calibration_cfg,
            method_name="llada_unary",
        )
    )
    results.append(
        run_dllm_setscore(
            bundle=bundle,
            backbone_name="llada",
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=default_inference_config("llada", use_refinement=True),
            calibration_config=calibration_cfg,
            method_name="llada_jsr",
        )
    )
    return results


def run_supervised_comparison_for_bundle(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    results = []
    calibration_cfg = default_calibration_config("auto")
    enc_cfg = default_encoder_train_config(bundle)
    t5_cfg = default_t5_train_config(bundle)

    print(f"=== Supervised comparison: {bundle.name} ===")
    results.append(
        train_encoder_multilabel(
            bundle=bundle,
            method_name="bert_base_supervised",
            model_name=SUPERIVSED_MODEL_REGISTRY["bert_base"],
            train_config=enc_cfg,
            calibration_config=calibration_cfg,
        )
    )
    results.append(
        train_encoder_multilabel(
            bundle=bundle,
            method_name="roberta_base_supervised",
            model_name=SUPERIVSED_MODEL_REGISTRY["roberta_base"],
            train_config=enc_cfg,
            calibration_config=calibration_cfg,
        )
    )
    results.append(
        train_t5_text_to_set(
            bundle=bundle,
            model_name=SUPERIVSED_MODEL_REGISTRY["t5_base"],
            train_config=t5_cfg,
        )
    )
    return results


def run_ablation_suite_for_bundle(bundle: DatasetBundle, backbone_name: str = "mdlm") -> pd.DataFrame:
    prompt_cfg = default_prompt_config(bundle)
    shortlist_cfg = default_shortlist_config(bundle)
    base_calibration = default_calibration_config("global")
    results = []

    # Refinement sweeps
    for sweeps in [0, 1, 2, 3]:
        infer_cfg = default_inference_config(backbone_name, use_refinement=sweeps > 0)
        infer_cfg.refine_sweeps = sweeps
        infer_cfg.mc_samples_refine = 4
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=infer_cfg,
            calibration_config=base_calibration,
            method_name=f"{backbone_name}_ablation_sweeps_{sweeps}",
        )
        out["ablation_name"] = "refinement_sweeps"
        out["ablation_value"] = sweeps
        results.append(out)

    # Monte Carlo / mask context
    for mc in [1, 4, 8, 16]:
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        infer_cfg.mc_samples_refine = mc
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=infer_cfg,
            calibration_config=base_calibration,
            method_name=f"{backbone_name}_ablation_mc_{mc}",
        )
        out["ablation_name"] = "mc_samples_refine"
        out["ablation_value"] = mc
        results.append(out)

    for strategy in ["none", "global", "labelwise", "expected_cardinality"]:
        calib_cfg = default_calibration_config(strategy)
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_cfg,
            inference_config=infer_cfg,
            calibration_config=calib_cfg,
            method_name=f"{backbone_name}_ablation_cal_{strategy}",
        )
        out["ablation_name"] = "calibration_strategy"
        out["ablation_value"] = strategy
        results.append(out)

    for verbalizer_pair in VERBALIZER_CANDIDATES:
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        try:
            out = run_dllm_setscore(
                bundle=bundle,
                backbone_name=backbone_name,
                prompt_config=prompt_cfg,
                shortlist_config=shortlist_cfg,
                inference_config=infer_cfg,
                calibration_config=base_calibration,
                verbalizer_pair=verbalizer_pair,
                method_name=f"{backbone_name}_ablation_verbalizer_{slugify('_'.join(verbalizer_pair))}",
            )
            out["ablation_name"] = "verbalizer_pair"
            out["ablation_value"] = "/".join(v.strip() for v in verbalizer_pair)
            results.append(out)
        except Exception as exc:
            print(f"Skipping verbalizer pair {verbalizer_pair} for {backbone_name}: {exc}")

    for randomize in [False, True]:
        prompt_cfg_variant = default_prompt_config(bundle)
        prompt_cfg_variant.randomize_label_order = randomize
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg_variant,
            shortlist_config=shortlist_cfg,
            inference_config=infer_cfg,
            calibration_config=base_calibration,
            method_name=f"{backbone_name}_ablation_label_order_{'random' if randomize else 'fixed'}",
        )
        out["ablation_name"] = "label_order_randomization"
        out["ablation_value"] = randomize
        results.append(out)

    for sep in [";", ",", "|"]:
        prompt_cfg_variant = default_prompt_config(bundle)
        prompt_cfg_variant.answer_separator = sep
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg_variant,
            shortlist_config=shortlist_cfg,
            inference_config=infer_cfg,
            calibration_config=base_calibration,
            method_name=f"{backbone_name}_ablation_separator_{slugify(sep)}",
        )
        out["ablation_name"] = "answer_separator"
        out["ablation_value"] = sep
        results.append(out)

    for k in [16, 32, 64]:
        shortlist_variant = ShortlistConfig(
            method=shortlist_cfg.method,
            k=min(k, len(bundle.label_names)),
            retriever_model=shortlist_cfg.retriever_model,
            fit_on_train_texts=shortlist_cfg.fit_on_train_texts,
            seed=shortlist_cfg.seed,
        )
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_variant,
            inference_config=infer_cfg,
            calibration_config=base_calibration,
            method_name=f"{backbone_name}_ablation_shortlist_{k}",
        )
        out["ablation_name"] = "shortlist_k"
        out["ablation_value"] = k
        results.append(out)

    for retriever_method in ["tfidf", "sbert", "random"]:
        shortlist_variant = ShortlistConfig(
            method=retriever_method,
            k=shortlist_cfg.k,
            retriever_model=shortlist_cfg.retriever_model,
            fit_on_train_texts=shortlist_cfg.fit_on_train_texts,
            seed=shortlist_cfg.seed,
        )
        infer_cfg = default_inference_config(backbone_name, use_refinement=True)
        out = run_dllm_setscore(
            bundle=bundle,
            backbone_name=backbone_name,
            prompt_config=prompt_cfg,
            shortlist_config=shortlist_variant,
            inference_config=infer_cfg,
            calibration_config=base_calibration,
            method_name=f"{backbone_name}_ablation_retriever_{retriever_method}",
        )
        out["ablation_name"] = "retriever_method"
        out["ablation_value"] = retriever_method
        results.append(out)

    ablation_df = pd.DataFrame(results)
    ablation_path = TABLES_DIR / f"ablation_{bundle.name}_{backbone_name}.csv"
    ablation_df.to_csv(ablation_path, index=False)
    return ablation_df


def run_all_experiments() -> Dict[str, Any]:
    summary: Dict[str, Any] = {"main": [], "supervised": [], "ablations": {}}
    core_datasets = ["goemotions", "reuters21578_top20", "eurlex57k"]

    for dataset_name in core_datasets:
        bundle = load_dataset_bundle(dataset_name)
        if RUN_MAIN_EXPERIMENTS:
            summary["main"].extend(run_main_comparison_for_bundle(bundle))
        if RUN_SUPERVISED_EXPERIMENTS:
            summary["supervised"].extend(run_supervised_comparison_for_bundle(bundle))
        if RUN_ABLATIONS:
            summary["ablations"][dataset_name] = {}
            for backbone_name in ["mdlm", "llada"]:
                ablation_df = run_ablation_suite_for_bundle(bundle, backbone_name=backbone_name)
                summary["ablations"][dataset_name][backbone_name] = ablation_df

    if RUN_OPTIONAL_EXTENSIONS:
        try:
            bundle = load_dataset_bundle("multieurlex_en")
            summary.setdefault("optional", {})
            if RUN_MAIN_EXPERIMENTS:
                summary["optional"]["multieurlex_main"] = run_main_comparison_for_bundle(bundle)
            if RUN_SUPERVISED_EXPERIMENTS:
                summary["optional"]["multieurlex_supervised"] = run_supervised_comparison_for_bundle(bundle)
        except Exception as exc:
            print(f"Skipping optional MultiEURLEX-English extension: {exc}")

    exported = export_results_tables()
    summary["exported_tables"] = {k: str(v) for k, v in exported.items()}
    return summary




def export_bundle_for_external_repos(bundle: DatasetBundle, export_name: Optional[str] = None) -> Path:
    export_name = export_name or bundle.name
    out_dir = EXPORT_DIR / export_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(split_name: str, texts: List[str], labels: List[List[int]]) -> None:
        df = pd.DataFrame({
            "text": texts,
            "label_ids": [json.dumps(x) for x in labels],
            "label_texts": [json.dumps([bundle.label_texts[i] for i in x], ensure_ascii=False) for x in labels],
        })
        df.to_csv(out_dir / f"{split_name}.csv", index=False)

    _write_split("train", bundle.train_texts, bundle.train_labels)
    _write_split("val", bundle.val_texts, bundle.val_labels)
    _write_split("test", bundle.test_texts, bundle.test_labels)

    label_df = pd.DataFrame({
        "label_idx": list(range(len(bundle.label_names))),
        "label_name": bundle.label_names,
        "label_text": bundle.label_texts,
    })
    label_df.to_csv(out_dir / "labels.csv", index=False)
    return out_dir


def scan_prediction_artifacts() -> pd.DataFrame:
    rows = []
    pred_root = RESULTS_DIR / "predictions"
    if not pred_root.exists():
        return pd.DataFrame()
    for run_dir in pred_root.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        pred_path = run_dir / "predictions.npz"
        if not cfg_path.exists() or not pred_path.exists():
            continue
        cfg = load_json(cfg_path)
        rows.append({
            "run_dir": str(run_dir),
            "dataset": cfg.get("dataset"),
            "method": cfg.get("method"),
        })
    return pd.DataFrame(rows)


def render_pareto_plot() -> Optional[Path]:
    """Generate (or update) the accuracy/latency Pareto plot from saved results."""
    all_results = pd.concat(
        [
            read_jsonl(RESULTS_DIR / "main_results.jsonl"),
            read_jsonl(RESULTS_DIR / "supervised_results.jsonl"),
        ],
        ignore_index=True,
    )
    if all_results.empty:
        return None
    return plot_pareto_front(all_results, PLOTS_DIR / "pareto_front.png")
