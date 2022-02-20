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
from typing import List
import torch
from torch.nn import CrossEntropyLoss

from lightning_transformers.core.nlp import HFTransformer
from lightning_transformers.core.utils import calc_rep_tf_and_acc


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
    
    def setup(self, stage: str):
        self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        return super().setup(stage)

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def common_step(self, prefix, batch):
        labels = batch.pop('labels')
        outputs = self.model(output_hidden_states=True, **batch)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

        self.log(
            f"{prefix}_loss", loss[:, -self.cfg.lm_stride:].mean(), # only log the second part of the losses
            add_dataloader_idx=False,
        )

        final_loss = loss.mean() + self.calc_aux_loss(prefix, batch, shift_logits, outputs.hidden_states[-1][:, :-1, :], shift_labels)

        non_padding = labels != self.criterion.ignore_index
        
        if self.training:
            return final_loss
        else:
            return calc_rep_tf_and_acc(logits, non_padding, labels)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        # max_length = self.cfg.val_target_max_length if self.cfg.val_target_max_length else self.model.config.max_length
        num_beams = self.cfg.num_beams if self.cfg.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, # max_length=max_length,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=input_ids.size(1) + self.cfg.generation_length,
        )
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        prefix_len = input_ids.size(1)
        return pred_str, generated_tokens[:, prefix_len:]

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"
