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
from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer
from lightning_transformers.task.nlp.conversation.config import ConversationConfig

# from lightning_transformers.task.nlp.conversation.metric import RougeMetric


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
        # self.rouge = None

    def compute_generate_metrics(self, batch, prefix):
        pass
        # tgt_lns = self.tokenize_labels(batch["labels"])
        # pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        # result = self.rouge(pred_lns, tgt_lns)
        # self.log_dict(result, on_step=False, on_epoch=True)

    # def configure_metrics(self, stage: str):
    #     self.rouge = RougeMetric(
    #         rouge_newline_sep=self.cfg.rouge_newline_sep,
    #         use_stemmer=self.cfg.use_stemmer,
    #     )

    @property
    def hf_pipeline_task(self) -> str:
        return "conversation"
