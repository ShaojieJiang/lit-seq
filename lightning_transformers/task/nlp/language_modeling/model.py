# Copyright The PyTorch Lightning team.
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
from torch.nn import CrossEntropyLoss

from lightning_transformers.core.nlp import HFTransformer


class LanguageModelingTransformer(HFTransformer):
    """Defines ``LightningModule`` for the Language Modeling Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForCausalLM``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(self, *args, downstream_model_type: str = "transformers.AutoModelForCausalLM", **kwargs) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.criterion = CrossEntropyLoss(reduction='none')

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch, batch_idx):
        labels = batch.pop('labels')
        outputs = self.model(**batch)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # loss = outputs[0]
        return loss[-self.cfg.lm_stride:].mean()

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"
