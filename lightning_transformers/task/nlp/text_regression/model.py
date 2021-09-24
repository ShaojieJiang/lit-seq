# Copyright The PyTorch Lightning team and Shaojie Jiang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict

import torch

from lightning_transformers.core.nlp import HFTransformer


class TextRegressionTransformer(HFTransformer):
    """Defines ``LightningModule`` for the Text Regression Task. This is a modification
    of the Text Classification Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSequenceClassification``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(
        self, *args, downstream_model_type: str = "transformers.AutoModelForSequenceClassification", **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        # self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        # preds = torch.argmax(logits, dim=1)
        # metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        # self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def configure_metrics(self, _) -> None:
        # TODO: add correlation metric?
        pass

    @property
    def num_classes(self) -> int:
        return 1
        # return self.trainer.datamodule.num_classes

    # def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
    #     # Not required by all models. Only required for classification
    #     return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}

    @property
    def hf_pipeline_task(self) -> str:
        return "sentiment-analysis"
