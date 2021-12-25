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
from transformers import AutoModel, BlenderbotForConditionalGeneration, BlenderbotModel, Conversation
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
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
        if cfg.strengthen_position:
            model = BlenderbotForConditionalGenerationSPOS.from_pretrained(backbone.pretrained_model_name_or_path)
        else:
            model_cls: Type["AutoModel"] = get_class(downstream_model_type)
            model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
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
        loss = self.common_step("val", batch)
        return self.common_eval_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self.common_step("test", batch)
        return self.common_eval_step("test", batch)

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
        self.log(
            f"{prefix}_loss", ce_loss,
            add_dataloader_idx=False,
        )

        # penalize pairwise similarity of last decoder hidden states
        if self.cfg.disparate_space == 'hidden':
            output_vectors = outputs.decoder_hidden_states[-1] # last decoder hidden states
        elif self.cfg.disparate_space == 'logits':
            output_vectors = logits

        # we always need to normalize to unit vectors, because scaled hidden states results in the same rankings
        output_vectors = F.normalize(output_vectors, dim=-1)

        sim_mask = 1 - torch.eye(logits.size(1)).unsqueeze(0).to(output_vectors.device) # don't penalise self similarity
        sim_mask = sim_mask.repeat(output_vectors.size(0), 1, 1)
        if self.cfg.padding_mask:
            tokens_mask = non_padding.float().unsqueeze(-1).bmm(non_padding.float().unsqueeze(1)).int()
            sim_mask *= tokens_mask
        
        if self.cfg.identical_mask: # id_mask entails padding mask
            different_tokens = (labels.unsqueeze(-1) != labels.unsqueeze(1)).int()
            sim_mask *= different_tokens
            
        # pairwise cosine similarity
        if self.cfg.distance == 'cosine' or not self.cfg.disparate: # report cosine when not using disparate regulariser
            psim = output_vectors.bmm(output_vectors.transpose(1, 2))
            cos_sim = (psim * sim_mask).sum() / sim_mask.sum() # range [-1, 1]
            sim_loss = 1 + cos_sim # range [0, 2]
        # p-norm distance
        elif self.cfg.distance.endswith('-norm'):
            p = int(self.cfg.distance.split('-')[0])
            pdist = torch.cdist(output_vectors, output_vectors, p=p)
            distance = (pdist * sim_mask).sum() / sim_mask.sum()

            if self.cfg.distance == '2-norm': # report cosine similarity for 2-norm
                psim = (2 - pdist.pow(2)) / 2
                cos_sim = (psim * sim_mask).sum() / sim_mask.sum()
                sim_loss = 2 - distance # range [0, 2]
            else:
                similarity = - distance
                sim_loss = similarity

        if self.cfg.distance in ['cosine', '2-norm'] or not self.cfg.disparate: # log cosine similarity
            self.log(
                f"{prefix}_similarity", cos_sim,
                add_dataloader_idx=False,
            )
        else: # log negative n-norm
            self.log(
                f"{prefix}_neg_{self.cfg.distance}", similarity,
                add_dataloader_idx=False,
            )
        
        if self.cfg.disparate:
            return ce_loss + self.cfg.disparate_alpha * sim_loss # the actual loss to backprop
        else:
            return ce_loss

    def common_eval_step(self, prefix: str, batch: Any) -> torch.Tensor:
        if self.cfg.compute_generate_metrics:
            ngram_counts = self.compute_generate_metrics(batch, prefix)

        return ngram_counts
    
    def aggregate_outputs(self, outputs, prefix):
        # aggregation strategy 1: add all #ngrams, #uniq_ngrams together, then take the division
        counts = Counter()
        for batch in outputs:
            for res in batch:
                counts.update(res)
        self.log_dict(
            {
                f'{prefix}_rep_1': 1 - (counts['uniq_unigrams'] + 1e-8) / (counts['num_unigrams'] + 1e-8),
                f'{prefix}_rep_2': 1 - (counts['uniq_bigrams'] + 1e-8) / (counts['num_bigrams'] + 1e-8),
                f'{prefix}_rep_3': 1 - (counts['uniq_trigrams'] + 1e-8) / (counts['num_trigrams'] + 1e-8),
                f'{prefix}_rep_4': 1 - (counts['uniq_fourgrams'] + 1e-8) / (counts['num_fourgrams'] + 1e-8),
            },
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

    def compute_generate_metrics(self, batch, prefix):
        _, generated_tokens = self.generate(batch["input_ids"], batch["attention_mask"])
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

        res = []
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
            res.append(
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


class BlenderbotForConditionalGenerationSPOS(BlenderbotForConditionalGeneration):
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
