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
from typing import Any

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
        self.criterion = torch.nn.MSELoss()
        # self.metrics = {}

    def common_step(self, batch: Any) -> torch.Tensor:
        labels = batch.pop('labels') # don't pass labels so that it doesn't calculate loss inside the model
        outputs = self.model(**batch)
        logits = outputs.logits
        scores_relu10 = logits.clamp(0, 10)
        loss = self.criterion(scores_relu10, labels)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.common_step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self.common_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self.common_step(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return loss

    def configure_metrics(self, _) -> None:
        # TODO: add correlation metric?
        pass
    
    def interact(self):
        self.eval()
        while True:
            user_message = input("Your Message: ")
            output = self.hf_pipeline(user_message)
            ntl = output[0]['score'] * 10 # (0, 10)
            
            print(f'{ntl:.2f} turns left.')
    
    # def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
    #     # Not required by all models. Only required for classification
    #     return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}

    @property
    def hf_pipeline_task(self) -> str:
        return 'text-classification'
