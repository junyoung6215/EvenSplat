"""Training-only parameter policy; no wrappers or checkpoint key changes."""
from torch import nn


HEAD_MODULES = (
    "downstream_head1", "downstream_head2",
    "gaussian_param_head", "gaussian_param_head2",
)
SCOPES = ("full", "heads")
CHECKPOINT_SCOPE_KEY = "noposplat_training_scope"


def validate_training_scope(scope: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"Unknown finetune_scope {scope!r}; expected {SCOPES}")


def validate_training_initialization(scope, mode, pretrained_weights, resume_checkpoint):
    validate_training_scope(scope)
    if scope == "heads" and mode == "train" and not (pretrained_weights or resume_checkpoint):
        raise ValueError(
            "heads training requires model.encoder.pretrained_weights or checkpointing.load; "
            "do not freeze a randomly initialized backbone."
        )


def apply_training_scope(encoder: nn.Module, scope: str) -> None:
    validate_training_scope(scope)
    if scope == "full":
        # Preserve all legacy requires_grad settings, including intentional freezes.
        return
    names = ("backbone", *HEAD_MODULES)
    if any(not isinstance(getattr(encoder, name, None), nn.Module) for name in names):
        raise ValueError("heads scope requires the two-view NoPoSplat backbone and four output heads")
    frozen_ids = {id(p) for p in encoder.backbone.parameters()}
    if any(id(p) in frozen_ids for name in HEAD_MODULES for p in getattr(encoder, name).parameters()):
        raise ValueError("Cannot freeze a backbone sharing parameters with output heads")
    encoder.requires_grad_(False)
    for name in HEAD_MODULES:
        getattr(encoder, name).requires_grad_(True)
    encoder.backbone.eval()


def training_scope_report(encoder: nn.Module, scope: str) -> dict:
    groups = {}
    trainable, frozen = [], []
    for name, param in encoder.named_parameters():
        (trainable if param.requires_grad else frozen).append(name)
        group = groups.setdefault(name.split(".", 1)[0], {"trainable": 0, "frozen": 0})
        group["trainable" if param.requires_grad else "frozen"] += param.numel()
    return {
        "finetune_scope": scope,
        "trainable_parameters": sum(g["trainable"] for g in groups.values()),
        "frozen_parameters": sum(g["frozen"] for g in groups.values()),
        "groups": groups,
        "trainable_names": trainable,
        "frozen_names": frozen,
    }


def validate_resume_scope(checkpoint: dict, scope: str) -> None:
    # Old checkpoints used the legacy full-training policy.
    saved = checkpoint.get(CHECKPOINT_SCOPE_KEY, "full")
    if saved != scope:
        raise ValueError(
            f"Cannot resume optimizer state from scope={saved!r} into scope={scope!r}. "
            "Use model.encoder.pretrained_weights and checkpointing.load=null to start "
            "a new fine-tuning run with a fresh optimizer."
        )
