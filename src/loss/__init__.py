from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_geo import LossGeo, LossGeoCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_reliability import LossReliability, LossReliabilityCfgWrapper
from .loss_reliability_prior import LossReliabilityPrior, LossReliabilityPriorCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossGeoCfgWrapper: LossGeo,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossReliabilityCfgWrapper: LossReliability,
    LossReliabilityPriorCfgWrapper: LossReliabilityPrior,
}

LossCfgWrapper = (
    LossDepthCfgWrapper
    | LossGeoCfgWrapper
    | LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossReliabilityCfgWrapper
    | LossReliabilityPriorCfgWrapper
)


def _enabled(cfg: LossCfgWrapper) -> bool:
    if isinstance(cfg, LossReliabilityPriorCfgWrapper):
        return bool(cfg.reliability_prior.enabled)
    if isinstance(cfg, LossGeoCfgWrapper):
        return bool(cfg.geo.enabled)
    if isinstance(cfg, LossReliabilityCfgWrapper):
        return bool(cfg.reliability.enabled)
    return True


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs if _enabled(cfg)]
