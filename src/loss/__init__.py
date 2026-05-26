from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_geo import LossGeo, LossGeoCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossGeoCfgWrapper: LossGeo,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = (
    LossDepthCfgWrapper | LossGeoCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [
        LOSSES[type(cfg)](cfg)
        for cfg in cfgs
        if not isinstance(cfg, LossGeoCfgWrapper) or cfg.geo.enabled
    ]
