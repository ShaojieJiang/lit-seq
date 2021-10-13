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
import random
from collections import defaultdict
from typing import Any, List

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.stats import pearsonr
from scipy.stats.stats import spearmanr

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
        if self.cfg.pooling_method != 'cls':
            self.linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        # self.metrics = {}

    def common_step(self, batch: Any, return_scores=False) -> torch.Tensor:
        input_keys = set(batch.keys()) - {'dialog_id', 'turn_id', 'labels'}
        inputs = {key: batch[key] for key in input_keys}
        outputs = self.model(**inputs)
        # pooling: mean/max/min/cls
        if self.cfg.pooling_method == 'cls':
            logits = outputs.logits.squeeze(-1)
        else:
            # apply attention mask
            masked = outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1) # bsz * seq_len * hidden_sz
            if self.cfg.pooling_method == 'mean':
                pooled = masked.mean(dim=1)
            elif self.cfg.pooling_method == 'max':
                pooled = masked.max(dim=1)[0]
            elif self.cfg.pooling_method == 'min':
                pooled = masked.min(dim=1)[0]
            logits = self.linear(pooled).squeeze(-1)
        scores_relu10 = logits.clamp(0, 10)
        # Avg baseline
        # scores_relu10 = (7.84 - batch['turn_id']).clamp(min=0.0) / 0.784
        loss = self.criterion(scores_relu10, batch['labels'])
        if return_scores:
            return loss, scores_relu10

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if type(batch) == list: # multi-tasking
            choice = random.randrange(0, len(batch))
            batch = batch[choice]
        loss = self.common_step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self.common_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True, add_dataloader_idx=False)

        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss, scores = self.common_step(batch, return_scores=True)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return {
            'loss': loss,
            'scores': scores.cpu().tolist(),
            'text': self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
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

        self.calc_correlations_first_last_n(ordered_dialogs)
        self.calc_correlations_first_last_n(ordered_dialogs, n=[3, 2, 1])
        self.get_greetings_farewells(ordered_dialogs)

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

    def calc_correlations(self, ordered_dialogs):
        all_labels_scores = [[(turn[2], turn[1]) for turn in dialog] for dialog in ordered_dialogs]

        flatten_scores = [turn for dialog in all_labels_scores for turn in dialog]
        scores, labels = zip(*flatten_scores)
        return pearsonr(scores, labels), spearmanr(scores, labels)
    
    def calc_correlations_first_last_n(self, ordered_dialogs, n=None):
        if n is None:
            pearson, spearman = self.calc_correlations(ordered_dialogs)
            print(f'Overall Pearson: {pearson[0]:.2f}, pval: {pearson[1]}')
            print(f'Overall Spearman: {spearman[0]:.2f}, pval: {spearman[1]}')
        elif type(n) is int:
            removed_intermediate = [dialog[:n] + dialog[-n:] if len(dialog) > 2*n else dialog for dialog in ordered_dialogs]
            pearson, spearman = self.calc_correlations(removed_intermediate)
            print(f'First/Last {n} P: {pearson[0]:.2f}, pval: {pearson[1]}')
            print(f'First/Last {n} S: {spearman[0]:.2f}, pval: {spearman[1]}')
        elif type(n) is list:
            for N in n:
                self.calc_correlations_first_last_n(ordered_dialogs, n=N)

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
