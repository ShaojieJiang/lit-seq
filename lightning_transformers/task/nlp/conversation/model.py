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
from typing import TYPE_CHECKING, Any, List, Optional, Type
import uuid

import torch
from hydra.utils import get_class
from transformers import AutoConfig, Conversation
from transformers.models.blenderbot.modeling_blenderbot import shift_tokens_right
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.core.nlp.model import HFTransformer
from lightning_transformers.core.utils import calc_rep_tf_and_acc, get_unique_total_ngrams, repeated_ngrams

if TYPE_CHECKING:
    from transformers import AutoModel


class ConversationTransformer(HFTransformer):
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
        model_cls: Type["AutoModel"] = get_class(downstream_model_type)
        
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

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        labels = batch.pop('labels')
        decoder_input_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        outputs = self.model(decoder_input_ids=decoder_input_ids, output_hidden_states=True, **batch)
        # batch['labels'] = decoder_input_ids
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
        
        if not self.cfg.negative_method.startswith('ul') and self.cfg.preced_m_negatives:
            wsz = self.cfg.ct_seq_len
            logits_ct = logits[..., :wsz, :]
            labels_ct = labels[..., :wsz]
            final_loss = loss.mean() + self.calc_aux_loss(prefix, batch, logits_ct, outputs.decoder_hidden_states[-1][:, :wsz, :], labels_ct)
        else:
            final_loss = ce_loss + self.calc_aux_loss(prefix, batch, logits, outputs.decoder_hidden_states[-1], labels)
        
        if self.training:
            return final_loss
        else:
            return calc_rep_tf_and_acc(logits, non_padding, labels)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        max_length = self.cfg.max_length if self.cfg.max_length else self.model.config.max_length
        num_beams = self.cfg.num_beams if self.cfg.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, max_length=max_length,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=self.cfg.encoder_no_repeat_ngram_size,
            min_length=self.cfg.min_length,
            # do_sample=True, top_p=0.9, # top_k=50,
        )
        return input_ids, generated_tokens
    
    def write_generations_to_file(self, input_ids, generated_tokens):
        f = open(self.cfg.save_generation_path, 'a')
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
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
            
        contexts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for ctx, res, rep in zip(contexts, responses, rep_rates):
            f.write(f"{ctx}\n{res}\n{rep}\n\n")
        f.close()

    def hf_predict(self, *args, **kwargs) -> Any:
        conversation = Conversation(args[0])
        self.hf_pipeline(conversation)

        return conversation.generated_responses[0]
    
    def interact(self):
        self.eval()
        conv = ReverseConversationFixed(history_size=self.cfg.history_size)
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
            
            print("Bot: ", conv.generated_responses[-1])

    @property
    def hf_pipeline_task(self) -> str:
        return "conversational"
    
    def compute_seq_ul(self, batch):
        pad_id = self.tokenizer.pad_token_id
        min_length = self.cfg.min_length
        generated = self.model.generate.__wrapped__(
            self.model,
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_beams=1,
            max_length=50,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            min_length=min_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_logits = torch.cat([scores.unsqueeze(1) for scores in generated.scores], dim=1)
        gen_probs = gen_logits.softmax(dim=-1)
        pred_probs = gen_probs.gather(2, generated.sequences[:, 1:].unsqueeze(-1))
        one_minus_probs = torch.clamp(1 - pred_probs, min=1e-20).squeeze(-1)
        repeated = repeated_ngrams(generated.sequences[:, 1:], n=4)
        repeated *= generated.sequences[:, 1:] != pad_id
        seq_ul = -torch.log(one_minus_probs) * repeated
        seq_ul = seq_ul.sum()
        
        return seq_ul


class ReverseConversationFixed(Conversation):
    def __init__(
        self, text: str = None, conversation_id: uuid.UUID = None,
        past_user_inputs=None, generated_responses=None, history_size=None,
    ):
        super().__init__(text, conversation_id, past_user_inputs, generated_responses)
        self.history_size = history_size
    
    def iter_texts(self):
        texts = list(super().iter_texts())
        history = texts[::-1]
        if self.history_size:
            history = history[:self.history_size]
        for (is_user, text) in history:
            yield is_user, text
