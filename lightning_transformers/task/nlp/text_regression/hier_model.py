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
from lightning_transformers.task.nlp.text_regression.model import TextRegressionTransformer


class HierarchicalBert(torch.nn.Module):
    def __init__(self, downstream_model_type: str, backbone: HFBackboneConfig, pooling_method, **model_data_kwargs):
        super().__init__()
        # model_cls: Type["AutoModel"] = get_class(downstream_model_type)
        self.turn_encoder = BertModel.from_pretrained(backbone.pretrained_model_name_or_path)
        # self.hier_encoder = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
        self.pooling_method = pooling_method
        self.linear = torch.nn.Linear(self.turn_encoder.config.hidden_size, 1)

    def forward(self, turn_batches, **kwargs):
        turn_outputs = []
        dialog_attention_mask = []
        for turn_batch in turn_batches:
            turn_output = self.turn_encoder(**turn_batch)
            if self.pooling_method == 'first':
                turn_output = turn_output['pooler_output'].unsqueeze(1)
            else:
                masked = turn_output['last_hidden_state'] * turn_batch['attention_mask'].unsqueeze(-1) # bsz * seq_len * hidden_sz
                if self.pooling_method == 'mean':
                    turn_output = masked.mean(dim=1, keepdim=True)
                elif self.pooling_method == 'max':
                    turn_output = masked.max(dim=1, keepdim=True)[0]
                elif self.pooling_method == 'min':
                    turn_output = masked.min(dim=1, keepdim=True)[0]

            turn_outputs.append(turn_output)
            turn_attn_mask = (turn_batch['attention_mask'].sum(dim=1, keepdim=True) > 1).long()
            dialog_attention_mask.append(turn_attn_mask)

        hier_input = torch.cat(turn_outputs, dim=1)
        dialog_attention_mask = torch.cat(dialog_attention_mask, dim=-1)
        masked = hier_input * dialog_attention_mask.unsqueeze(-1)
        logits = masked.mean(dim=1)

        # output = self.hier_encoder(inputs_embeds=hier_input, attention_mask=dialog_attention_mask)
        # if self.pooling_method == 'first':
        #     logits = output['pooler_output']
        # else:
        #     masked = output['last_hidden_state'] * dialog_attention_mask.unsqueeze(-1)
        #     if self.pooling_method == 'mean':
        #         logits = masked.mean(dim=1)
        #     elif self.pooling_method == 'max':
        #         logits = masked.max(dim=1)[0]
        #     elif self.pooling_method == 'min':
        #         logits = masked.min(dim=1)[0]

        scores = self.linear(logits)

        return scores.squeeze(-1)


class HierarchicalTextRegressionTransformer(TextRegressionTransformer):

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
        model = HierarchicalBert(downstream_model_type, backbone, cfg.pooling_method, **model_data_kwargs)
        super(HFTransformer, self).__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}
        self.criterion = torch.nn.MSELoss()
    
    def common_step(self, batch: Any, return_scores=False) -> torch.Tensor:
        logits = self.model(**batch)
        scores_relu1 = logits.clamp(0, 1)
        loss = 100 * self.criterion(scores_relu1, batch['labels'])
        
        if return_scores:
            return loss, scores_relu1

        return loss
    
    def common_step_return_scores(self, batch, stage='val'):
        loss, scores = self.common_step(batch, return_scores=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        texts = [[] for _ in range(len(batch['labels']))]
        for turn_batch in batch['turn_batches']:
            turn_texts = self.tokenizer.batch_decode(turn_batch['input_ids'], skip_special_tokens=True)
            for i, turn_text in enumerate(turn_texts):
                if turn_text != '':
                    texts[i].append(turn_text)
        
        texts = [turn_texts[-1] for turn_texts in texts] # only keep the last turn

        return {
            'loss': loss,
            'scores': scores.cpu().tolist(),
            'text': texts,
            'labels': batch['labels'].cpu().tolist(),
            'dialog_id': batch['dialog_id'].cpu().tolist(),
            'turn_id': batch['turn_id'].cpu().tolist(),
        }
        