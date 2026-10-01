import json
from pathlib import Path
from typing import Any, Optional

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self, output_dir: Path = LOG_PATH) -> None:
        super().__init__()
        self.experiment = None
        # A run must never delete a previous run's local artifacts.
        self.output_dir = Path(output_dir)

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "metrics": metrics}, default=str) + "\n")

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.output_dir / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
