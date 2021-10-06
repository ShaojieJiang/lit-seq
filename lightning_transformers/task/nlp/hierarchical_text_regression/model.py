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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

import torch
from hydra.utils import get_class
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.stats import pearsonr
from transformers import pipeline as hf_transformers_pipeline
from transformers.pipelines.base import Pipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModel, BertModel

from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.core.nlp.model import HFTransformer


class HierarchicalBert(torch.nn.Module):
    def __init__(self, downstream_model_type: str, backbone: HFBackboneConfig, **model_data_kwargs):
        super().__init__()
        model_cls: Type["AutoModel"] = get_class(downstream_model_type)
        self.turn_encoder = BertModel.from_pretrained(backbone.pretrained_model_name_or_path)
        self.hier_encoder = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)

    def forward(self, turn_batches, **kwargs):
        turn_outputs = []
        for turn_batch in turn_batches:
            turn_output = self.turn_encoder(**turn_batch)['pooler_output'].unsqueeze(1)
            turn_outputs.append(turn_output)

        hier_input = torch.cat(turn_outputs, dim=1)
        output = self.hier_encoder(inputs_embeds=hier_input)

        return output


class HierarchicalTextRegressionTransformer(TaskTransformer):

    def __init__(
        self,
        downstream_model_type: str,
        backbone: HFBackboneConfig,
        optimizer: OptimizerConfig = OptimizerConfig(),
        scheduler: SchedulerConfig = SchedulerConfig(),
        instantiator: Optional[Instantiator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        pipeline_kwargs: Optional[dict] = None,
        cfg: Optional[LitTaskConfig] = None,
        **model_data_kwargs,
    ) -> None:
        self.save_hyperparameters()
        model = HierarchicalBert(downstream_model_type, backbone, **model_data_kwargs)
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}
        self.criterion = torch.nn.MSELoss()

    @property
    def tokenizer(self) -> Optional["PreTrainedTokenizerBase"]:
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")  # noqa: W503
            and hasattr(self.trainer, "datamodule")  # noqa: W503
            and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        self._tokenizer = tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]

    def common_step(self, batch: Any, return_scores=False) -> torch.Tensor:
        outputs = self.model(**batch)
        logits = outputs.logits.squeeze(-1)
        scores_relu10 = logits.clamp(0, 10)
        loss = self.criterion(scores_relu10, batch['labels'])
        if return_scores:
            return loss, scores_relu10

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
        loss, scores = self.common_step(batch, return_scores=True)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return {
            'loss': loss,
            'scores': scores.cpu().tolist(),
            'text': "Dummy text.", # self.tokenizer.batch_decode([turn[input_ids] for], skip_special_tokens=True),
            'labels': batch['labels'].cpu().tolist(),
            'dialog_id': batch['dialog_id'].cpu().tolist(),
            'turn_id': batch['turn_id'].cpu().tolist(),
        }
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # collate all results in the format of List[dialogs[turns]]
        dialogs = defaultdict(list)
        for batch_output in outputs:
            for res_pair in zip(batch_output['text'], batch_output['labels'], batch_output['scores'], 
                batch_output['dialog_id'], batch_output['turn_id']):
                text, label, score, dialog_id, turn_id = list(res_pair)
                dialogs[dialog_id].append((text, label, score, turn_id))
        
        ordered_dialogs = []
        for dialog_id, dialog in dialogs.items():
            # sort dialog turns in case they're not in the correct order
            dialog = sorted(dialog, key=lambda x: int(x[-1]))
            ordered_dialogs.append(dialog)

        self.get_greetings_farewells(ordered_dialogs)
        self.calc_pearsonr_first_last_n(ordered_dialogs)
        self.calc_pearsonr_first_last_n(ordered_dialogs, n=[3, 2, 1])

    def get_greetings_farewells(self, ordered_dialogs):
        GREETING_THRESH = 8.5
        FAREWELL_THRESH = 1.5
        greetings_file = open('./greetings.txt', 'w')
        farewells_file = open('./farewells.txt', 'w')
        for dialog in ordered_dialogs:
            for i, turn in enumerate(dialog):
                if turn[2] >= GREETING_THRESH: # greetings
                    self._write_filtered_dialogs(greetings_file, dialog, i)
                elif turn[2] <= FAREWELL_THRESH: # farewells
                    if i > 1 and dialog[i - 1][2] <= FAREWELL_THRESH:
                        # direct previous turn has been recorded, skip this one
                        continue
                    self._write_filtered_dialogs(farewells_file, dialog, i)

        greetings_file.close()
        farewells_file.close()
    
    def _write_filtered_dialogs(self, fp, dialog, turn_id):
        # write all previous turns and current
        for i in range(turn_id + 1):
            turn = dialog[i]
            fp.write(f'{turn[0]}\n')
            # fp.write(f'{turn[0]}\tGT: {turn[1]:.2f}\tPred: {turn[2]:.2f}\n')
        fp.write('\n')
        # fp.write('============\n\n')

    def calc_pearsonr(self, ordered_dialogs):
        all_labels_scores = [[(turn[2], turn[1]) for turn in dialog] for dialog in ordered_dialogs]

        flatten_scores = [turn for dialog in all_labels_scores for turn in dialog]
        scores, labels = zip(*flatten_scores)
        return pearsonr(scores, labels)
    
    def calc_pearsonr_first_last_n(self, ordered_dialogs, n=None):
        if n is None:
            res = self.calc_pearsonr(ordered_dialogs)
            print(f'Overall: {res[0]:.2f}, pval: {res[1]}')
        elif type(n) is int:
            removed_intermediate = [dialog[:n] + dialog[-n:] if len(dialog) > 2*n else dialog for dialog in ordered_dialogs]
            res = self.calc_pearsonr(removed_intermediate)
            print(f'First/Last {n}: {res[0]:.2f}, pval: {res[1]}')
        elif type(n) is list:
            for N in n:
                self.calc_pearsonr_first_last_n(ordered_dialogs, n=N)
    
    @torch.no_grad()
    def interact(self):
        self.eval()
        while True:
            user_message = input("Your Message: ")
            sentences = user_message.split('[SEP]')
            tokenized = [self.tokenizer(sent, return_tensors='pt').to(self.device) for sent in sentences]
            model_input = {'turn_batches': tokenized}
            output = self.model(**model_input)
            ntl = output.logits.clamp(0, 10).cpu().item()
            
            print(f'{ntl:.2f} turns left.')