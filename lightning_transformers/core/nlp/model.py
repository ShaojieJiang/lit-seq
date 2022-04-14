import random
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

import torch
import torch.nn.functional as F
from hydra.utils import get_class
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from transformers import PreTrainedTokenizerBase
from transformers import pipeline as hf_transformers_pipeline

from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.core.utils import calc_vector_similarity, contrastive_loss, get_unique_total_ngrams, nce_loss, negative_loss

if TYPE_CHECKING:
    from transformers import AutoModel, Pipeline


class HFTransformer(TaskTransformer):
    """Base class for task specific transformers, wrapping pre-trained language models for downstream tasks. The
    API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html

    Args:
        downstream_model_type: The AutoModel downstream model type.
            See https://huggingface.co/transformers/model_doc/auto.html
        backbone: Config containing backbone specific arguments.
        optimizer: Config containing optimizer specific arguments.
        scheduler: Config containing scheduler specific arguments.
        instantiator: Used to instantiate objects (when using Hydra).
            If Hydra is not being used the instantiator is not required,
            and functions that use instantiation such as ``configure_optimizers`` has been overridden.
        tokenizer: The pre-trained tokenizer.
        pipeline_kwargs: Arguments required for the HuggingFace inference pipeline class.
        **model_data_kwargs: Arguments passed from the data module to the class.
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
        model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}

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

    @property
    def hf_pipeline_task(self) -> Optional[str]:
        """Override to define what HuggingFace pipeline task to use.

        Returns: Optional string to define what pipeline task to use.
        """
        return None

    @property
    def hf_pipeline(self) -> "Pipeline":
        if self._hf_pipeline is None:
            if self.hf_pipeline_task is not None:
                self._hf_pipeline = hf_transformers_pipeline(
                    task=self.hf_pipeline_task, model=self.model, tokenizer=self.tokenizer, **self._hf_pipeline_kwargs
                )
            else:
                raise RuntimeError("No task was defined for this model. Try overriding `hf_pipeline_task`")
        return self._hf_pipeline

    @hf_pipeline.setter
    def hf_pipeline(self, pipeline: "Pipeline") -> None:
        self._hf_pipeline = pipeline

    def hf_predict(self, *args, **kwargs) -> Any:
        return self.hf_pipeline(*args, **kwargs)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        raise NotImplementedError

    def common_eval_step(self, prefix: str, batch: Any) -> torch.Tensor:
        ngram_counts = None
        if self.cfg.compute_generate_metrics:
            ngram_counts = self.compute_generate_metrics(batch, prefix)

        return ngram_counts
    
    @property
    def should_generate(self):
        if self.trainer.global_step / max(self.trainer.max_steps, 1e-8) >= self.cfg.generate_after_progress:
            return True
        return False
    
    def calc_aux_loss(self, prefix: str, batch: Any, logits, hidden_states, labels):
        aux_loss = 0.0

        if self.cfg.simctg:
            mean_sim, sim_loss = calc_vector_similarity(
                hidden_states,
                labels,
                padding_id=self.criterion.ignore_index,
                padding_mask=self.cfg.padding_mask,
                identical_mask=self.cfg.identical_mask,
            )

            self.log(
                f"{prefix}_similarity", mean_sim,
                add_dataloader_idx=False,
            )

            aux_loss += sim_loss

        if self.cfg.topk_negatives or self.cfg.preced_m_negatives:
            if self.cfg.negative_method.startswith('ul'):
                neg_loss = negative_loss(
                    logits,
                    labels,
                    orig_pad_id=self.criterion.ignore_index,
                    method=self.cfg.negative_method,
                    pad_id=self.tokenizer.pad_token_id,
                    topk_negatives=self.cfg.topk_negatives,
                    preced_m_negatives=self.cfg.preced_m_negatives,
                )
            elif self.cfg.negative_method.startswith('ct'):
                neg_loss = contrastive_loss(
                    logits,
                    labels,
                    orig_pad_id=self.criterion.ignore_index,
                    pad_id=self.tokenizer.pad_token_id,
                    topk_negatives=self.cfg.topk_negatives,
                    preced_m_negatives=self.cfg.preced_m_negatives,
                )
            elif self.cfg.negative_method == 'nce':
                neg_loss = nce_loss(
                    logits,
                    labels,
                    orig_pad_id=self.criterion.ignore_index,
                    pad_id=self.tokenizer.pad_token_id,
                    preced_m_negatives=self.cfg.preced_m_negatives,
                )
            self.log(
                f"{prefix}_neg_loss", neg_loss,
                add_dataloader_idx=False,
            )

            aux_loss += neg_loss # the actual loss to backprop
        
        if self.cfg.ul_seq and self.training:
            seq_ul = self.compute_seq_ul(batch)
            self.log(
                f"{prefix}_seq_ul", seq_ul,
                add_dataloader_idx=False,
            )
            aux_loss += seq_ul
        
        if self.cfg.ct_seq and self.training:
            seq_ct = self.compute_ct_seq(batch)
            self.log(
                f"{prefix}_seq_ct", seq_ct,
                add_dataloader_idx=False,
            )
            aux_loss += seq_ct
        
        return aux_loss

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

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        rep_tf = self.common_step("test", batch)
        rep_gen = self.common_eval_step("test", batch)
        rep_gen.update(rep_tf)

        return rep_gen

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
        tensor = F.pad(tensor, (0, pad_length), 'constant', self.tokenizer.pad_token_id)
        tensor = self.all_gather(tensor)
        tensor = tensor.view(-1, max_length)

        return tensor

    def compute_generate_metrics(self, batch, prefix):
        input_ids, generated_tokens = self.generate(batch["input_ids"], batch["attention_mask"])
        # generated_tokens = batch['labels']
        # input_ids = batch["input_ids"]
        if self.trainer.gpus > 1:
            generated_tokens = self.pad_and_gather(generated_tokens)
            input_ids = self.pad_and_gather(input_ids)
            
        if self.cfg.save_generation_path is not None and self.global_rank == 0:
            self.write_generations_to_file(input_ids, generated_tokens)

        ngram_counts = get_unique_total_ngrams(
            generated_tokens,
            bos_id=self.tokenizer.bos_token_id,
            eos_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        self.log(
            f'{prefix}_pred_len', (generated_tokens[:, 1:] != 0).sum(dim=-1),
            add_dataloader_idx=False,
        )
        return ngram_counts
    
    def aggregate_outputs(self, outputs, prefix):
        counts = Counter()
        for batch in outputs:
            counts.update(batch)
        for key, val in counts.items():
            if type(val) is list:
                counts[key] = set(val)
        self.log_dict(
            {
                f'{prefix}_accuracy': (counts['correct_tf'] + 1e-8) / (counts['num_total_tf'] + 1e-8),
                f'{prefix}_rep_tf': (counts['num_rep_tf'] + 1e-8) / (counts['num_total_tf'] + 1e-8),
            },
            add_dataloader_idx=False,
        )
        if self.should_generate or prefix == 'test':
            self.log_dict(
                {
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
