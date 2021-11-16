import random
from collections import defaultdict
from typing import Any, List, Optional

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.stats import pearsonr
from scipy.stats.stats import spearmanr
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator

from lightning_transformers.core.nlp import HFTransformer
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.task.nlp.ntm.modules import EncapsulatedNTM


class NTM(HFTransformer):
    """Defines ``LightningModule`` for the Text Regression Task. This is a modification
    of the Text Classification Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.BertModel``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

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
        model = EncapsulatedNTM(**model_data_kwargs['ntm'])
        super(HFTransformer, self).__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}
        self.criterion = torch.nn.BCELoss()

    def common_step(self, batch: Any, return_scores=False) -> torch.Tensor:
        self.model.init_sequence(batch.size(1))
        for i in range(batch.size(0)):
            self.model(batch[i])
        
        label = batch[:-1, :, :-1]

        y_out = torch.zeros(label.size())
        for i in range(label.size(0)):
            y_out[i], _ = self.model()

        loss = self.criterion(y_out, label)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if type(batch) == list: # multi-tasking
            choice = random.randrange(0, len(batch))
            batch = batch[choice]
        loss = self.common_step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        val_dataloader = self.val_dataloader()
        if isinstance(val_dataloader, list) and dataloader_idx == len(val_dataloader) - 1:# only run common_eval for last dataset
            return self.common_step_return_scores(batch, stage='val')
        else:
            loss = self.common_step(batch)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True, add_dataloader_idx=False)
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        val_dataloader = self.val_dataloader()
        if isinstance(val_dataloader, list) and len(val_dataloader) > 1: # the last dataloader is for correlations
            pearson, spearman = self.eval_correlations(outputs[-1])
            self.log("pearson", pearson[0])
            self.log("spearman", spearman[0])
    
    def common_step_return_scores(self, batch, stage='val'):
        loss, scores = self.common_step(batch, return_scores=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return {
            'loss': loss,
            'scores': scores.cpu().tolist(),
            'text': self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
            'labels': batch['labels'].cpu().tolist(),
            'dialog_id': batch['dialog_id'].cpu().tolist(),
            'turn_id': batch['turn_id'].cpu().tolist(),
        }
    
    def eval_correlations(self, outputs: EPOCH_OUTPUT) -> None:
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

        pearson, spearman = self.calc_correlations_first_last_n(ordered_dialogs)
        self.calc_correlations_first_last_n(ordered_dialogs, n=[3, 2, 1])
        # self.get_greetings_farewells(ordered_dialogs)

        # fp = open('./predictions.txt', 'w')
        # for dialog in ordered_dialogs:
        #     for turn in dialog:
        #         fp.write(f'{turn[0]}\tGT: {turn[1]:.2f}\tPred: {turn[2]:.2f}\n')
        #     fp.write('\n')
        #     fp.write('============\n\n')
        # fp.close()

        return pearson, spearman

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step_return_scores(batch, stage='test')
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.eval_correlations(outputs) # no logging or return needed

    def get_greetings_farewells(self, ordered_dialogs):
        GREETING_THRESH = 0
        FAREWELL_THRESH = 0.15
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

    def pearson_spearman_correlations(self, ordered_dialogs):
        all_labels_scores = [[(turn[2], turn[1]) for turn in dialog] for dialog in ordered_dialogs]

        flatten_scores = [turn for dialog in all_labels_scores for turn in dialog]
        scores, labels = zip(*flatten_scores)
        return pearsonr(scores, labels), spearmanr(scores, labels)
    
    def calc_correlations_first_last_n(self, ordered_dialogs, n=None):
        if n is None:
            pearson, spearman = self.pearson_spearman_correlations(ordered_dialogs)
            print(f'Overall Pearson: {pearson[0]:.2f}, pval: {pearson[1]}')
            print(f'Overall Spearman: {spearman[0]:.2f}, pval: {spearman[1]}')
        elif type(n) is int:
            removed_intermediate = [dialog[:n] + dialog[-n:] if len(dialog) > 2*n else dialog for dialog in ordered_dialogs]
            pearson, spearman = self.pearson_spearman_correlations(removed_intermediate)
            print(f'First/Last {n} P: {pearson[0]:.2f}, pval: {pearson[1]}')
            print(f'First/Last {n} S: {spearman[0]:.2f}, pval: {spearman[1]}')
        elif type(n) is list:
            for N in n:
                pearson, spearman = self.calc_correlations_first_last_n(ordered_dialogs, n=N)
        
        return pearson, spearman

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
