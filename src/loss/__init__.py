from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_reliability import LossReliability, LossReliabilityCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossReliabilityCfgWrapper: LossReliability,
}

LossCfgWrapper = (
    LossDepthCfgWrapper
    | LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossReliabilityCfgWrapper
)


def _enabled(cfg: LossCfgWrapper) -> bool:
    if isinstance(cfg, LossReliabilityCfgWrapper):
        return bool(cfg.reliability.enabled)
    return True


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs if _enabled(cfg)]
