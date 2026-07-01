"""Configuration dataclasses for the starling transfer model.

A single YAML file (see ``ml/configs/default.yaml``) is loaded into :class:`Config`.
Every field has a default so partial YAML overrides are fine. CLI scripts accept
``--config path.yaml`` plus ``--set key.subkey=value`` overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

# The free-text metadata fields used by the model, in a fixed order. `support_text` is
# deliberately EXCLUDED as a label leak (the bioavailability label is a percentage and
# ~93% of support_text rows contain a '%'). `oral_bioavailability_value` is never included.
METADATA_FIELDS: tuple[str, ...] = (
    "molecule_name",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
)


@dataclass
class PathsConfig:
    # Short dataset label used to namespace ml/results/<dataset>/{runs,plots,tables}.
    dataset: str = "same_species_v2"
    base_parquet: str = "datasets/base/Oral_bioavailability_cleaned_v2/train.parquet"
    splits_dir: str = "datasets/pairs_split_full/oral_bioavailability_pair_splits_same_species_v2_full"
    embeddings_dir: str = "ml/artifacts/embeddings_same_species_v2"
    memmap_dir: str = "ml/artifacts/memmap_same_species_v2"
    output_dir: str = "ml/artifacts/runs/same_species_v2"


@dataclass
class EmbeddingConfig:
    molformer_model: str = "ibm-research/MoLFormer-XL-both-10pct"
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    metadata_fields: list[str] = field(default_factory=lambda: list(METADATA_FIELDS))
    mol_emb_dim: int = 768
    text_emb_dim: int = 384
    n_meta_fields: int = len(METADATA_FIELDS)
    smiles_batch_size: int = 256
    text_batch_size: int = 1024
    # MolFormer's tokenizer caps SMILES length; longer ones are truncated.
    max_smiles_tokens: int = 202

    def __post_init__(self) -> None:
        self.metadata_fields = list(self.metadata_fields)
        self.n_meta_fields = len(self.metadata_fields)


@dataclass
class ModelConfig:
    # Per-molecule MolFormer branch (siamese 2-layer MLP). Keep the full 768-d MolFormer
    # representation (mol_out=768, no compression) and let the head do the heavy lifting.
    mol_hidden: int = 1024
    mol_out: int = 768
    # Per-field metadata projection (siamese, field-specific): 384 -> meta_field_proj each.
    meta_field_proj: int = 64
    # Residual SwiGLU head.
    d_model: int = 1024
    d_ff: int = 4096
    n_blocks: int = 32
    dropout: float = 0.1
    # LayerScale init for residual branches (per-channel learnable scale); <=0 disables.
    layerscale_init: float = 0.0
    # Directional variant: append the source-side oral bioavailability scalar to the head input.
    use_source_value: bool = False
    source_value_scale: float = 100.0


@dataclass
class LossConfig:
    # one of: "bce", "focal"
    kind: str = "bce"
    # positive class is ~26% -> pos_weight ~ 0.74/0.26 ~= 2.85 for plain reweighting,
    # but a milder 1.85 (negatives/positives capped) is the documented default; tune as needed.
    pos_weight: float = 1.85
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # label smoothing epsilon (0 disables); composes with bce/focal.
    label_smoothing: float = 0.0


@dataclass
class TrainConfig:
    per_device_batch_size: int = 65536  # batch-sweep winner (global 524288 on 8 GPUs)
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    # If num_train_epochs > 0 it takes precedence over max_steps (max_steps set to -1).
    num_train_epochs: float = 0.0
    max_steps: int = 20000
    eval_steps: int = 100
    logging_steps: int = 2
    dataloader_num_workers: int = 8
    bf16: bool = True
    tf32: bool = True
    torch_compile: bool = True
    seed: int = 0
    # AUROC on this many random train pairs is logged each eval as the capacity signal.
    # Set to 0 to evaluate/log only the validation split.
    train_eval_samples: int = 30000
    report_to: str = "wandb"  # "none" | "tensorboard" | "wandb"
    wandb_project: str = "starling_assay_transfer_mlp"  # used only when report_to == "wandb"
    run_name: str = ""  # wandb/Trainer run name; empty -> Trainer default (output_dir)
    # Mirror full-val metrics to the wandb `val/*` section. Disable for HP sweeps so they only
    # populate the default `eval` section, keeping `val/*` reserved for the final full run.
    wandb_val_mirror: bool = True
    # "binary" keeps the historical broad metric set; "simple_transfer" reports only the
    # compact benchmark metrics.
    metric_set: str = "binary"
    # When enabled, validation memmaps retain eval_subset and final evaluation includes
    # no_overlap, a_seen_only, and both_seen slices.
    eval_subset_metrics: bool = False
    eval_subset_names: list[str] = field(
        default_factory=lambda: ["no_overlap", "a_seen_only", "both_seen"]
    )
    # When enabled, validation/test memmaps retain fixed-width Tanimoto similarity
    # bucket codes and evaluation reports per-bucket metrics from the same val pass.
    eval_similarity_bucket_metrics: bool = False
    eval_similarity_bucket_names: list[str] = field(
        default_factory=lambda: [
            "tanimoto_0_0p2",
            "tanimoto_0p2_0p4",
            "tanimoto_0p4_0p6",
            "tanimoto_0p6_0p8",
            "tanimoto_0p8_1",
        ]
    )
    # For benchmark final runs, bypass HF's broad W&B integration and log only the
    # whitelisted validation metric keys.
    wandb_simple_validation_only: bool = False
    # Metric key from Trainer evaluation metrics used for the rolling best checkpoint.
    best_metric: str = "eval_val_macro_f1"
    tdc_eval_enabled: bool = False
    tdc_eval_path: str = "tdc/official_tianang/train/Bioavailability_Ma.jsonl"
    tdc_eval_valid_path: str = "tdc/official_tianang/valid/Bioavailability_Ma.jsonl"
    tdc_eval_valid_on_best_val: bool = True
    tdc_eval_cache_dir: str = "ml/artifacts/tdc_knn_eval"
    tdc_eval_steps: int = 250
    tdc_eval_top_fraction: float = 0.25
    tdc_eval_k: int = 10
    tdc_eval_batch_size: int = 65536
    record_knn_eval_enabled: bool = False
    record_knn_eval_dataset_dir: str = "datasets/starling_eval/condition_key_v3_record_splits_hf"
    record_knn_eval_dataset_config: str = "full_metadata"
    record_knn_eval_splits: list[str] = field(default_factory=lambda: ["validation_1"])
    record_knn_final_splits: list[str] = field(default_factory=lambda: ["validation_1", "validation_2"])
    record_knn_eval_cache_dir: str = "ml/artifacts/record_knn_eval_cache/condition_key_v3_record_splits_hf"
    record_knn_eval_steps: int = 500
    record_knn_eval_top_fraction: float = 0.10
    record_knn_eval_k: int = 10
    record_knn_eval_batch_size: int = 4096
    record_knn_eval_max_queries: int = 0


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- (de)serialization helpers ----
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml

        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def apply_overrides(self, overrides: list[str]) -> "Config":
        """Apply ``--set a.b=value`` style dotted overrides in place."""
        import json

        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got: {item!r}")
            key, raw = item.split("=", 1)
            try:
                value = json.loads(raw)  # parses ints/floats/bools/json; falls back to str
            except json.JSONDecodeError:
                value = raw
            obj: Any = self
            parts = key.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if not hasattr(obj, parts[-1]):
                raise AttributeError(f"unknown config key: {key}")
            setattr(obj, parts[-1], value)
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    import typing

    # Resolve string annotations (PEP 563 / `from __future__ import annotations`).
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _build(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
