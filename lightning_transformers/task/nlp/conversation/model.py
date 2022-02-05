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
import random
from collections import Counter
from typing import Any, List, Optional, Type

import torch
import torch.nn.functional as F
from hydra.utils import get_class
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from transformers import AutoConfig, AutoModel, BlenderbotForConditionalGeneration, BlenderbotModel, Conversation
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.blenderbot.configuration_blenderbot import BlenderbotConfig
from transformers.models.blenderbot.modeling_blenderbot import (
    BlenderbotDecoder,
    BlenderbotEncoder,
    _expand_mask,
    shift_tokens_right,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.core.nlp.model import HFTransformer
from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer


class ConversationTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Conversation Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        **kwargs: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
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
        model_cls = None
        if cfg.clr:
            if cfg.strengthen_position:
                model_cls = BlenderbotForConditionalGenerationClrSPos
            else:
                model_cls = BlenderbotForConditionalGenerationClr
        else:
            if cfg.strengthen_position:
                model_cls = BlenderbotForConditionalGenerationSPos
            else:
                model_cls = BlenderbotForConditionalGeneration
            
        if cfg.scratch:
            config = AutoConfig.from_pretrained(backbone.pretrained_model_name_or_path)
            model = model_cls(config)
        else:
            model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path)

        super(HFTransformer, self).__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if type(batch) == list: # multi-tasking
            choice = random.randrange(0, len(batch))
            batch = batch[choice]
        return self.common_step('train', batch)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        rep_tf = self.common_step("val", batch)
        if self.should_generate:
            rep_gen = self.common_eval_step("val", batch)
        else:
            rep_gen = Counter()
        rep_gen.update(rep_tf)

        return rep_gen
    
    @property
    def should_generate(self):
        if self.trainer.global_step / max(self.trainer.max_steps, 1e-8) >= self.cfg.generate_after_progress:
            return True
        return False

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        rep_tf = self.common_step("test", batch)
        rep_gen = self.common_eval_step("test", batch)
        rep_gen.update(rep_tf)

        return rep_gen
    
    def compute_seq_ul(self, batch):
        pad_id = self.tokenizer.pad_token_id

        generated = self.model.generate.__wrapped__(
            self.model,
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_beams=1,
            max_length=50,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            min_length=self.cfg.min_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_logits = torch.cat([scores.unsqueeze(1) for scores in generated.scores], dim=1)
        gen_probs = gen_logits.softmax(dim=-1)
        pred_probs = gen_probs.gather(2, generated.sequences[:, 1:].unsqueeze(-1))
        one_minus_probs = torch.clamp(1 - pred_probs, min=1e-20).squeeze(-1)
        repeated = self.repeated_ngrams(generated.sequences[:, 1:], n=4)
        repeated *= generated.sequences[:, 1:] != pad_id
        seq_ul = -torch.log(one_minus_probs) * repeated
        seq_ul = seq_ul.sum()
        
        return seq_ul
    
    def repeated_ngrams(self, tensor, n):
        mask = torch.zeros_like(tensor)
        for i, x in enumerate(tensor):
            seen = set()
            xl = x.tolist()
            for j in range(len(x)-n):
                ng = tuple(xl[j:j+n])
                if ng in seen:
                    mask[i, j:j+n] = 1
                seen.add(ng)
        return mask
    
    def calc_similarity(self, hidden_states, indices, padding_id=0):
        non_padding = indices != padding_id

        sim_mask = 1 - torch.eye(indices.size(1)).to(indices.device) # don't penalise self similarity
        sim_mask = sim_mask.repeat(indices.size(0), 1, 1)
        if self.cfg.padding_mask: # don't calc similarity for padding tokens
            tokens_mask = non_padding.float().unsqueeze(-1).bmm(non_padding.float().unsqueeze(1)).int()
            sim_mask *= tokens_mask
        
        if self.cfg.identical_mask: # id_mask entails padding mask
            different_tokens = (indices.unsqueeze(-1) != indices.unsqueeze(1)).int()
            sim_mask *= different_tokens
            
        vector_represen = F.normalize(hidden_states, dim=-1)
        pair_sim = vector_represen.bmm(vector_represen.transpose(1, 2))
        # report the avg similarity of all
        cos_sim = (pair_sim * sim_mask).sum() / (sim_mask.sum() + 1e-8) # get average cosine similarity

        cut_cos_sim = None
        if self.cfg.disparate:
            sim_mask *= (pair_sim >= self.cfg.sim_threshold).int()
            # report the avg similarity of all
            cut_cos_sim = (pair_sim * sim_mask).sum() / (sim_mask.sum() + 1e-8) # get average cosine similarity

        return cos_sim, cut_cos_sim
    
    def negative_sampling(self, logits=None, labels=None, pad_id=0):
        neg_exs = None
        topk_preds = None
        if self.cfg.topk_hard_negatives > 0:
            _, topk_preds = logits.topk(k=self.cfg.topk_hard_negatives)
        
        if topk_preds is not None:
            neg_exs = topk_preds
        
        preced_tokens = None
        if self.cfg.preced_k_negatives >= 0: # use previous k tokens as negatives
            preced_tokens = labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)).tril(-1)
            if self.cfg.preced_k_negatives > 0:
                preced_tokens = preced_tokens.triu(- self.cfg.preced_k_negatives) # discard further preced tokens

        if preced_tokens is not None:
            if neg_exs is None:
                neg_exs = preced_tokens
            else:
                neg_exs = torch.cat([neg_exs, preced_tokens], dim=-1)

        neg_exs = neg_exs.masked_fill(neg_exs == labels.unsqueeze(-1), pad_id) # exclude same label tokens as negatives

        return neg_exs
    
    def negative_loss(self, logits, target_inds, orig_pad_id=0, method='cl2'):
        non_padding = target_inds != orig_pad_id
        labels = target_inds * (target_inds >= 0).int()
        pad_id = self.tokenizer.pad_token_id

        neg_exs = self.negative_sampling(logits=logits, labels=labels, pad_id=pad_id)

        negative_targets = torch.zeros_like(logits).scatter_(2, neg_exs, 1)
        negative_targets.scatter_(2, torch.zeros_like(labels).unsqueeze(-1) + pad_id, 0) # don't treat the pad_id as negative example

        if method == 'ul':
            # penalise previous tokens
            probs = logits.softmax(dim=-1)
            token_ul = -torch.log(torch.clamp(1 - probs, min=1e-20)) * negative_targets
            token_ul = token_ul.sum(dim=-1) * non_padding.int()
            ul_loss = token_ul.sum() / non_padding.int().sum()

            return ul_loss
        else:
            gt_scores = logits.gather(2, labels.unsqueeze(-1))

            if method == 'cl1':
                # contrastive with previous tokens
                # this implementation doesn't consider repeated negatives
                neg_minus_pos = logits - gt_scores
                exp = neg_minus_pos.exp() * negative_targets
                sum_exp = exp.sum(dim=-1)
            elif method == 'cl2':
                # this cl implementation consideres repeated netatives
                pad_mask = (neg_exs != pad_id).int()
                neg_scores = logits.gather(2, neg_exs)
                neg_minus_pos = neg_scores - gt_scores
                exp = neg_minus_pos.exp()
                sum_exp = (exp * pad_mask).sum(dim=-1) # don't use pad tokens as negatives

            losses = (1 + sum_exp).log() * non_padding.int()
            cl_loss = losses.sum() / non_padding.int().sum()

            return cl_loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        labels = batch.pop('labels')
        decoder_input_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        outputs = self.model(decoder_input_ids=decoder_input_ids, output_hidden_states=True, **batch)
        # loss = outputs[0]
        logits = outputs.logits

        non_padding = labels != self.criterion.ignore_index

        # calculate CE loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
        ce_loss = loss.sum() / non_padding.int().sum()

        if self.cfg.disparate:
            mean_sim, sim_loss = self.calc_similarity(
                outputs.decoder_hidden_states[-1],
                labels,
                padding_id=self.criterion.ignore_index,
            )
        else:
            with torch.no_grad():
                mean_sim, sim_loss = self.calc_similarity(
                    outputs.decoder_hidden_states[-1],
                    labels,
                    padding_id=self.criterion.ignore_index,
                )

        self.log_dict(
            {
                f"{prefix}_loss": ce_loss,
                f"{prefix}_similarity": mean_sim,
            },
            add_dataloader_idx=False,
        )
        # if prefix == 'val':
        #     best_val_loss = self.trainer.logged_metrics.get('val_loss', float('inf'))
        #     if ce_loss < best_val_loss:
        #         self.log(
        #             'best_val_loss', ce_loss,
        #             add_dataloader_idx=False,
        #         )

        final_loss = ce_loss
        if self.cfg.disparate: # report cosine when not using disparate regulariser
            final_loss += self.cfg.disparate_alpha * sim_loss

        if self.cfg.topk_hard_negatives > 0 or self.cfg.preced_k_negatives > -1:
            neg_loss = self.negative_loss(
                logits,
                labels,
                orig_pad_id=self.criterion.ignore_index,
                method=self.cfg.negative_method,
            )
            self.log(
                f"{prefix}_neg_loss", neg_loss,
                add_dataloader_idx=False,
            )

            final_loss += neg_loss # the actual loss to backprop
        
        if self.cfg.unlikelihood:
            seq_ul = self.compute_seq_ul(batch)
            self.log(
                f"{prefix}_seq_ul", seq_ul,
                add_dataloader_idx=False,
            )
            final_loss += self.cfg.ul_alpha * seq_ul
        
        if self.training:
            return final_loss
        else:
            preds = logits.argmax(dim=-1)
            # mask padding
            preds *= non_padding.int()
            # mask identical tokens
            different_tokens = (labels.unsqueeze(-1) != labels.unsqueeze(1)).int()
            different_tokens = different_tokens.tril()
            true_non_rep = different_tokens.sum(dim=-1) == torch.arange(preds.size(1)).to(preds.device)
            preds *= true_non_rep.int()
            # calculate rep-1
            different_preds = (preds.unsqueeze(-1) != preds.unsqueeze(1)).int()
            different_preds = different_preds.tril()
            rep_preds = different_preds.sum(dim=-1) < torch.arange(preds.size(1)).to(preds.device)
            rep_preds *= non_padding
            num_repeated = rep_preds.sum(-1)
            num_total = non_padding.sum(-1)

            return {
                'num_rep_tf': num_repeated.sum().item(),
                'num_total_tf': num_total.sum().item(),
            }

    def common_eval_step(self, prefix: str, batch: Any) -> torch.Tensor:
        ngram_counts = None
        if self.cfg.compute_generate_metrics:
            ngram_counts = self.compute_generate_metrics(batch, prefix)

        return ngram_counts
    
    def aggregate_outputs(self, outputs, prefix):
        # aggregation strategy 1: add all #ngrams, #uniq_ngrams together, then take the division
        counts = Counter()
        for batch in outputs:
            counts.update(batch)
        for key, val in counts.items():
            if type(val) is list:
                counts[key] = set(val)
        if self.should_generate or prefix == 'test':
            self.log_dict(
                {
                    f'{prefix}_rep_tf': (counts['num_rep_tf'] + 1e-8) / (counts['num_total_tf'] + 1e-8),
                    f'{prefix}_uniq_1': len(counts['unigrams']),
                    f'{prefix}_uniq_2': len(counts['bigrams']),
                    f'{prefix}_uniq_3': len(counts['trigrams']),
                    f'{prefix}_uniq_4': len(counts['fourgrams']),
                    f'{prefix}_distinct_1': (len(counts['unigrams']) + 1e-8) / (counts['num_unigrams'] + 1e-8),
                    f'{prefix}_distinct_2': (len(counts['bigrams']) + 1e-8) / (counts['num_bigrams'] + 1e-8),
                    f'{prefix}_distinct_3': (len(counts['trigrams']) + 1e-8) / (counts['num_trigrams'] + 1e-8),
                    f'{prefix}_distinct_4': (len(counts['fourgrams']) + 1e-8) / (counts['num_fourgrams'] + 1e-8),
                    f'{prefix}_rep_1': 1 - (counts['uniq_unigrams'] + 1e-8) / (counts['num_unigrams'] + 1e-8),
                    f'{prefix}_rep_2': 1 - (counts['uniq_bigrams'] + 1e-8) / (counts['num_bigrams'] + 1e-8),
                    f'{prefix}_rep_3': 1 - (counts['uniq_trigrams'] + 1e-8) / (counts['num_trigrams'] + 1e-8),
                    f'{prefix}_rep_4': 1 - (counts['uniq_fourgrams'] + 1e-8) / (counts['num_fourgrams'] + 1e-8),
                },
                add_dataloader_idx=False,
            )
        else:
            self.log(
                f'{prefix}_rep_tf',
                (counts['num_rep_tf'] + 1e-8) / (counts['num_total_tf'] + 1e-8),
                add_dataloader_idx=False,
            )
        # aggregation strategy 2: calc the repetition rate of each example, then average
        # Update: This 2nd strategy doesn't expose the problem well

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        val_dataloader = self.val_dataloader()
        if isinstance(val_dataloader, list) and len(val_dataloader) > 1:
            # flatten the outputs for different datasets
            outputs = [batch for dset_output in outputs for batch in dset_output]
            
        self.aggregate_outputs(outputs, 'val')

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        test_dataloader = self.test_dataloader()
        if isinstance(test_dataloader, list) and len(test_dataloader) > 1:
            # flatten the outputs for different datasets
            outputs = [batch for dset_output in outputs for batch in dset_output]

        self.aggregate_outputs(outputs, 'test')
    
    def pad_and_gather(self, tensor):
        max_length = self.cfg.val_target_max_length
        pad_length = max_length - tensor.size(-1)
        # pad tensors to the same size, otherwise all_gather will be stuck
        tensor = F.pad(tensor, (0, pad_length), 'constant', 0)
        tensor = self.all_gather(tensor)
        tensor = tensor.view(-1, max_length)

        return tensor

    def compute_generate_metrics(self, batch, prefix):
        _, generated_tokens = self.generate(batch["input_ids"], batch["attention_mask"])
        input_ids = batch["input_ids"]
        if self.trainer.gpus > 1:
            generated_tokens = self.pad_and_gather(generated_tokens)
            input_ids = self.pad_and_gather(input_ids)
            
        if self.cfg.save_generation_path is not None and self.global_rank == 0:
            f = open(self.cfg.save_generation_path, 'a')
            contexts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for ctx, res in zip(contexts, responses):
                f.write(f"{ctx}\n{res}\n\n")
            f.close()

        ngram_counts = self.get_unique_total_ngrams(generated_tokens)
        self.log(
            f'{prefix}_pred_len', (generated_tokens[:, 1:] != 0).sum(dim=-1),
            add_dataloader_idx=False,
        )
        return ngram_counts

    def get_unique_total_ngrams(self, batch_generations):
        assert type(batch_generations) is torch.Tensor
        batch_generations = batch_generations.cpu().numpy()

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        res = Counter()
        ngrams = Counter()
        for pred in batch_generations:
            # trim special tokens
            pred = pred[pred != bos_id]
            pred = pred[pred != eos_id]
            pred = pred[pred != pad_id].tolist()

            # get ngrams
            bigrams = [tuple(pred[i:i+2]) for i in range(len(pred) - 1)]
            trigrams = [tuple(pred[i:i+3]) for i in range(len(pred) - 2)]
            fourgrams = [tuple(pred[i:i+4]) for i in range(len(pred) - 3)]

            # count total and distinct numbers
            res.update(
                {
                    'num_unigrams': len(pred),
                    'num_bigrams': len(bigrams),
                    'num_trigrams': len(trigrams),
                    'num_fourgrams': len(fourgrams),
                    'uniq_unigrams': len(set(pred)),
                    'uniq_bigrams': len(set(bigrams)),
                    'uniq_trigrams': len(set(trigrams)),
                    'uniq_fourgrams': len(set(fourgrams)),
                }
            )
            ngrams.update(
                {
                    'unigrams': pred,
                    'bigrams': bigrams,
                    'trigrams': trigrams,
                    'fourgrams': fourgrams,
                }
            )
        # reduce unique ngrams and add to res
        for key, val in ngrams.items():
            res[key] = list(set(val))
        return res

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        # max_length = self.cfg.val_target_max_length if self.cfg.val_target_max_length else self.model.config.max_length
        num_beams = self.cfg.num_beams if self.cfg.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, # max_length=max_length,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=self.cfg.encoder_no_repeat_ngram_size,
            min_length=self.cfg.min_length,
        )
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str, generated_tokens

    def hf_predict(self, *args, **kwargs) -> Any:
        conversation = Conversation(args[0])
        self.hf_pipeline(conversation)

        return conversation.generated_responses[0]
    
    def interact(self):
        self.eval()
        conv = Conversation()
        while True:
            user_message = input("Your Message: ")
            conv.add_user_input(user_message)
            self.hf_pipeline(
                conv,
                no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=self.cfg.encoder_no_repeat_ngram_size,
                min_length=self.cfg.min_length,
                num_beams=self.cfg.num_beams,
            )
            
            print("Blenderbot: ", conv.generated_responses[-1])

    @property
    def hf_pipeline_task(self) -> str:
        return "conversational"


class BlenderbotForConditionalGenerationClr(BlenderbotForConditionalGeneration):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = F.normalize(outputs[0], dim=-1)
        bsz, seq_len, hid_sz = hidden_states.size()
        w = F.normalize(self.lm_head.weight.data, dim=-1)
        # lm_logits = w.bmm(hidden_states.transpose(1, 2))
        lm_logits = hidden_states.view(-1, hid_sz).mm(w.T).view(bsz, seq_len, -1) / 0.02 # temperature

        masked_lm_loss = None

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BlenderbotForConditionalGenerationClrSPos(BlenderbotForConditionalGenerationClr):
    def __init__(self, config: BlenderbotConfig):
        super(BlenderbotForConditionalGeneration, self).__init__(config)
        self.model = BlenderbotModelSPOS(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()


class BlenderbotForConditionalGenerationSPos(BlenderbotForConditionalGeneration):
    def __init__(self, config: BlenderbotConfig):
        super(BlenderbotForConditionalGeneration, self).__init__(config)
        self.model = BlenderbotModelSPOS(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()


class BlenderbotModelSPOS(BlenderbotModel):
    def __init__(self, config: BlenderbotConfig):
        super(BlenderbotModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BlenderbotEncoder(config, self.shared)
        self.decoder = BlenderbotDecoderSPOS(config, self.shared)

        self.init_weights()


class BlenderbotDecoderSPOS(BlenderbotDecoder):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                # if use_cache:
                #     logger.warning(
                #         "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                #         "`use_cache=False`..."
                #     )
                #     use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            hidden_states += positions # strengthen the effect of positional embeddings at each layer

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
