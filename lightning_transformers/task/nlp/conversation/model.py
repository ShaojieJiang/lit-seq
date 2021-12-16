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
from collections import Counter
from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from transformers import Conversation
from transformers.models.blenderbot.modeling_blenderbot import shift_tokens_right

from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer
from lightning_transformers.task.nlp.conversation.config import ConversationConfig


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
        *args,
        downstream_model_type: str = "transformers.AutoModelForSeq2SeqLM",
        cfg: ConversationConfig = ConversationConfig(),
        **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, cfg=cfg, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
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

        if self.cfg.distance == 'none':
            self.log_dict(
                {
                    f"{prefix}_loss": ce_loss,
                }
            )
            return ce_loss
        else:
            # penalize pairwise similarity of last decoder hidden states
            if self.cfg.disparate_space == 'hidden':
                output_vectors = outputs.decoder_hidden_states[-1] # last decoder hidden states
            elif self.cfg.disparate_space == 'logits':
                output_vectors = logits

            # normalize to unit vectors
            if self.cfg.norm_space or self.cfg.distance == 'cosine':
                output_vectors = F.normalize(output_vectors, dim=-1)
                
            # pairwise cosine similarity
            if self.cfg.distance == 'cosine':
                pdist = output_vectors.bmm(output_vectors.transpose(1, 2))
            # p-norm distance
            elif self.cfg.distance.endswith('-norm'):
                p = int(self.cfg.distance.split('-')[0])
                pdist = - torch.cdist(output_vectors, output_vectors, p=p) # negate the distance to learn to enlarge

            sim_mask = 1 - torch.eye(logits.size(1)).unsqueeze(0).to(pdist.device) # don't penalise self similarity
            sim_mask = sim_mask.repeat(pdist.size(0), 1, 1)
            if self.cfg.padding_mask:
                tokens_mask = non_padding.float().unsqueeze(-1).bmm(non_padding.float().unsqueeze(1)).int()
                sim_mask *= tokens_mask
            
            if self.cfg.identical_mask:
                different_tokens = (labels.unsqueeze(-1) != labels.unsqueeze(1)).int()
                sim_mask *= different_tokens

            similarity = (pdist * sim_mask).sum() / sim_mask.sum() # represents the similarity among last hidden states

            self.log_dict(
                {
                    f"{prefix}_loss": ce_loss,
                    f"{prefix}_similarity": similarity,
                }
            )
            return ce_loss + self.cfg.disparate_alpha * similarity # the actual loss to backprop

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
                f'{prefix}_rep_1': 1 - counts['uniq_unigrams'] / counts['num_unigrams'],
                f'{prefix}_rep_2': 1 - counts['uniq_bigrams'] / counts['num_bigrams'],
                f'{prefix}_rep_3': 1 - counts['uniq_trigrams'] / counts['num_trigrams'],
                f'{prefix}_rep_4': 1 - counts['uniq_fourgrams'] / counts['num_fourgrams'],
            }
        )
        # aggregation strategy 2: calc the repetition rate of each example, then average
        # Update: This 2nd strategy doesn't expose the problem well

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.aggregate_outputs(outputs, 'val')

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.aggregate_outputs(outputs, 'test')

    def compute_generate_metrics(self, batch, prefix):
        _, generated_tokens = self.generate(batch["input_ids"], batch["attention_mask"])
        ngram_counts = self.get_unique_total_ngrams(generated_tokens)
        self.log(f'{prefix}_pred_len', (generated_tokens[:, 1:] != 0).sum(dim=-1))
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
