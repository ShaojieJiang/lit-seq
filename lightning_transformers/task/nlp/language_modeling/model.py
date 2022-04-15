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
from typing import List, Optional

import torch
from hydra.utils import get_class
from transformers import AutoConfig, PreTrainedTokenizerBase

from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.nlp import HFTransformer
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.core.utils import calc_rep_tf_and_acc, get_unique_total_ngrams, repeated_ngrams


class LanguageModelingTransformer(HFTransformer):
    """Defines ``LightningModule`` for the Language Modeling Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForCausalLM``)
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
        model_cls = get_class(downstream_model_type)
            
        if cfg.scratch:
            config = AutoConfig.from_pretrained(backbone.pretrained_model_name_or_path)
            model = model_cls.from_config(config)
        else:
            model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path)

        super(HFTransformer, self).__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
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

        if not self.cfg.negative_method.startswith('ul') and self.cfg.preced_m_negatives:
            wsz = self.cfg.ct_seq_len
            logits_ct = shift_logits[..., :wsz, :]
            labels_ct = shift_labels[..., :wsz]
            final_loss = loss.mean() + self.calc_aux_loss(prefix, batch, logits_ct, outputs.hidden_states[-1][:, :wsz, :], labels_ct)
        else:
            final_loss = loss.mean() + self.calc_aux_loss(prefix, batch, shift_logits, outputs.hidden_states[-1][:, :-1, :], shift_labels)
        
        # from lightning_transformers.core.utils import ContrastiveTokenLoss
        # ct_criterion = ContrastiveTokenLoss(pad_id=self.tokenizer.pad_token_id)
        # final_loss = loss.mean() + ct_criterion(shift_logits, shift_labels)

        non_padding = shift_labels != self.criterion.ignore_index
        
        if self.training:
            return final_loss
        else:
            return calc_rep_tf_and_acc(shift_logits, non_padding, shift_labels)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        # max_length = self.cfg.val_target_max_length if self.cfg.val_target_max_length else self.model.config.max_length
        num_beams = self.cfg.num_beams if self.cfg.num_beams else self.model.config.num_beams
        prefix_len = self.cfg.prefix_length
        input_ids=input_ids[:, :prefix_len]
        attention_mask=attention_mask[:, :prefix_len]
        generated_tokens = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, # max_length=max_length,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.cfg.val_target_max_length,
            # do_sample=True, top_k=50, # top_p=0.9
        )
        return generated_tokens[:, :prefix_len], generated_tokens[:, prefix_len:]
    
    def write_generations_to_file(self, input_ids, generated_tokens):
        f = open(self.cfg.save_generation_path, 'a')
        bos_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_token_id
        rep_rates = []
        for i in range(len(generated_tokens)):
            counts = get_unique_total_ngrams(generated_tokens[i:i+1, :], bos_id, eos_id, pad_id)
            rep1 = round(1 - counts['uniq_unigrams'] / (counts['num_unigrams'] + 1e-8), 2)
            rep2 = round(1 - counts['uniq_bigrams'] / (counts['num_bigrams'] + 1e-8), 2)
            rep3 = round(1 - counts['uniq_trigrams'] / (counts['num_trigrams'] + 1e-8), 2)
            rep4 = round(1 - counts['uniq_fourgrams'] / (counts['num_fourgrams'] + 1e-8), 2)
            rep = f"Rep1: {rep1}, Rep2: {rep2}, Rep3: {rep3}, Rep4: {rep4}"
            rep_rates.append(rep)

        prefixes = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        continues = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        for pref, cont, rep in zip(prefixes, continues, rep_rates):
            f.write(f"Prefix: {pref}\nContinuation: {cont}\n{rep}\n\n")
        f.close()

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"
    
    def interact(self):
        self.eval()
        while True:
            prefix = input("Prefix: ")
            output = self.hf_pipeline(
                prefix,
                max_length=150,
                do_sample=False,
                num_beams=self.cfg.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            print("Continuation: ", output[0]['generated_text'])
    
    def compute_seq_ul(self, batch):
        pad_id = self.tokenizer.pad_token_id
        prefix_len = self.cfg.prefix_length
        generation_len = 90
        generated = self.model.generate.__wrapped__(
            self.model,
            input_ids=batch['input_ids'][:, :prefix_len],
            attention_mask=batch['attention_mask'][:, :prefix_len],
            num_beams=1,
            max_length=50 + generation_len,
            no_repeat_ngram_size=0,
            # encoder_no_repeat_ngram_size=0,
            # min_length=min_length,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_id,
        )
        gen_logits = torch.cat([scores.unsqueeze(1) for scores in generated.scores], dim=1)
        gen_probs = gen_logits.softmax(dim=-1)
        pred_probs = gen_probs.gather(2, generated.sequences[:, prefix_len:].unsqueeze(-1))
        one_minus_probs = torch.clamp(1 - pred_probs, min=1e-20).squeeze(-1)
        repeated = repeated_ngrams(generated.sequences[:, prefix_len:], n=4)
        repeated *= generated.sequences[:, prefix_len:] != pad_id
        seq_ul = -torch.log(one_minus_probs) * repeated
        seq_ul = seq_ul.sum()
        
        return seq_ul

    def compute_ct_seq(self, batch):
        pad_id = self.tokenizer.pad_token_id
        prefix_len = self.cfg.prefix_length
        generated = self.model.generate.__wrapped__(
            self.model,
            input_ids=batch['input_ids'][:, :prefix_len],
            attention_mask=batch['attention_mask'][:, :prefix_len],
            num_beams=1,
            max_length=140,
            no_repeat_ngram_size=0,
            # encoder_no_repeat_ngram_size=0,
            # min_length=140,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_id,
        )
        gen_logits = torch.cat([scores.unsqueeze(1) for scores in generated.scores], dim=1)
        gen_seqs = generated.sequences[:, -90:]
        repeated = repeated_ngrams(gen_seqs, n=4)
        neg_scores = gen_logits.gather(2, gen_seqs.unsqueeze(-1))
        pos_scores, _ = gen_logits.topk(k=2)
        neg_minus_pos = neg_scores.unsqueeze(-1) - pos_scores[..., -1:].unsqueeze(-2)
        exp = neg_minus_pos.exp()
        # exp = exp * false_positive_mask
        # pad_mask *= (exp <= pos_hardness).int() # don't use too hard negatives

        # ours
        sum_exp = exp.sum(dim=-1).sum(dim=-1) # don't use pad tokens as negatives
        losses = (1 + sum_exp).log() * repeated.int()
        repeat_loss = losses.sum() / (repeated.int().sum() + 1e-8)
        
        return repeat_loss
