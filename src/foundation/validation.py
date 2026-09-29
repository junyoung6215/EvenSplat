"""Fail before training setup when an opt-in prior experiment is inconsistent."""
from .prior import PriorProvider


def validate_prior_pipeline(cfg):
    if cfg.mode != "train":
        return  # inference is independent of training cache availability
    prior_losses = [wrapper.reliability_prior for wrapper in cfg.loss
                    if hasattr(wrapper, "reliability_prior")]
    active = any(loss.enabled and loss.weight != 0 for loss in prior_losses)
    if not active:
        return
    if len(cfg.dataset) != 1 or not hasattr(cfg.dataset[0], "re10k"):
        raise ValueError("Prior MVP requires a single RE10K dataset.")
    dataset = cfg.dataset[0].re10k
    prior = dataset.foundation_prior
    if not prior.enabled:
        raise ValueError("Active reliability_prior requires dataset.re10k.foundation_prior.enabled=true.")
    if dataset.view_sampler.name != "fixed_pairs" or dataset.overfit_to_scene is not None:
        raise ValueError("Prior experiments require explicit training fixed_pairs, not overfit/evaluation pairs.")
    if cfg.model.anchor_selection.enabled:
        raise ValueError("Prior experiments must retain the original first-view canonical frame.")
    if not cfg.model.encoder.gaussian_reliability.enabled:
        raise ValueError("Prior loss requires the reliability head enabled.")
    for wrapper in cfg.loss:
        if hasattr(wrapper, "reliability") and wrapper.reliability.enabled and wrapper.reliability.weight != 0:
            raise ValueError("Do not combine legacy residual and prior auxiliary losses in the MVP.")
    PriorProvider(prior, "train", augment=dataset.augment, fixed_pairs=True)
    if (prior.source == "mock" or prior.allow_mock) and any(not loss.allow_mock for loss in prior_losses if loss.enabled):
        raise ValueError("Diagnostic mock runs require explicit allow_mock in both dataset and loss.")
