from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_fixed_pairs import ViewSamplerFixedPairs, ViewSamplerFixedPairsCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "fixed_pairs": ViewSamplerFixedPairs,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | ViewSamplerFixedPairsCfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    if isinstance(cfg, ViewSamplerFixedPairsCfg) and stage != "train":
        cfg = ViewSamplerEvaluationCfg("evaluation", cfg.evaluation_index_path, 2)
    elif (isinstance(cfg, ViewSamplerBoundedCfg) and stage != "train"
          and cfg.evaluation_index_path is not None):
        cfg = ViewSamplerEvaluationCfg("evaluation", cfg.evaluation_index_path, 2)
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
