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
import inspect
import os
import re
import warnings
from collections import Counter
from typing import Mapping, Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F
from datasets.arrow_dataset import Dataset
from datasets.builder import DatasetBuilder
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.features import Features
from datasets.iterable_dataset import IterableDataset
from datasets.load import prepare_module
from datasets.metric import Metric
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, hash_python_lines
from datasets.splits import Split
from datasets.streaming import extend_module_for_streaming
from datasets.tasks.base import TaskTemplate
from datasets.utils.download_manager import GenerateMode
from datasets.utils.file_utils import DownloadConfig
from datasets.utils.info_utils import is_small_dataset
from datasets.utils.version import Version
from omegaconf.dictconfig import DictConfig


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"


def import_main_class(module, dataset=True) -> Optional[Union[Type[DatasetBuilder], Type[Metric]]]:

    if dataset:
        main_cls_type = DatasetBuilder
    else:
        main_cls_type = Metric

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            break

    return module_main_cls


def load_dataset_builder(
    dataset_module,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[GenerateMode] = None,
    script_version: Optional[Union[str, Version]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    **config_kwargs,
) -> DatasetBuilder:
    # Download/copy dataset processing script
    _, hash, base_path = prepare_module(
        dataset_module.__file__,
        script_version=script_version,
        download_config=download_config,
        download_mode=download_mode,
        dataset=True,
        return_associated_base_path=True,
        use_auth_token=use_auth_token,
        data_files=data_files,
    )
    rehash_fields = [hash, script_version]
    if 'history_delimiter' in config_kwargs:
        rehash_fields.append(config_kwargs['history_delimiter'])
    hash = hash_python_lines(rehash_fields) # rehasing and consider script version
    builder_cls = import_main_class(dataset_module) # import the class from our own file

    # Instantiate the dataset builder
    builder_instance: DatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        hash=hash,
        base_path=base_path,
        features=features,
        use_auth_token=use_auth_token,
        **config_kwargs,
    )

    return builder_instance


def load_my_dataset(
    dataset_module,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[GenerateMode] = None,
    ignore_verifications: bool = False,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    script_version: Optional[Union[str, Version]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    task: Optional[Union[str, TaskTemplate]] = None,
    streaming: bool = False,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """This is a modified version of datasets.load_dataset(), for easier debugging and configuration.
    """
    ignore_verifications = ignore_verifications or save_infos

    path = dataset_module.__file__
    # Create a dataset builder
    builder_instance = load_dataset_builder(
        dataset_module,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        script_version=script_version,
        use_auth_token=use_auth_token,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        # this extends the open and os.path.join functions for data streaming
        extend_module_for_streaming(builder_instance.__module__, use_auth_token=use_auth_token)
        return builder_instance.as_streaming_dataset(
            split=split,
            use_auth_token=use_auth_token,
        )

    # Some datasets are already processed on the HF google storage
    # Don't try downloading from google storage for the packaged datasets as text, json, csv or pandas
    try_from_hf_gcs = path not in _PACKAGED_DATASETS_MODULES

    # Download and prepare data
    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        ignore_verifications=ignore_verifications,
        try_from_hf_gcs=try_from_hf_gcs,
        use_auth_token=use_auth_token,
    )

    # Build dataset for splits
    keep_in_memory = (
        keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size)
    )
    ds = builder_instance.as_dataset(split=split, ignore_verifications=ignore_verifications, in_memory=keep_in_memory)
    # Rename and cast features to match task schema
    if task is not None:
        ds = ds.prepare_for_task(task)
    if save_infos:
        builder_instance._save_infos()

    return ds


def validate_resume_path(cfg: DictConfig):
    if not os.path.isfile(cfg.trainer.resume_from_checkpoint):
        cfg.trainer.resume_from_checkpoint = None
        
    return cfg


# This function is copied from ParlAI
def normalize_personachat_text(text: str, version=1) -> str:

    def uppercase(string: str) -> str:
        if len(string) == 0:
            return string
        else:
            return string[0].upper() + string[1:]
    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text


def normalize_dailydialog_text(text: str, version=1) -> str:
    replace_list = [
        ("’", "'"), ("\ '", " '"), ("”", '"'), ("“", '"'),
        (" ' d ", "'d "), (" ' ll ", "'ll "), (" ' s ", "'s "),
        (" ' t ", "'t "), (" ' re ", "'re "), (" ' Ve ", "'ve "),
        (" ' m ", "'m "),
    ]
    new_text = text
    for old, new in replace_list:
        new_text = new_text.replace(old, new)

    # quotes
    quotes = re.findall(r"( ' [\w ]+ ')|( \" [\w+ ]+ \")", new_text)
    for quote_tuple in quotes:
        if quote_tuple[0]:
            normalized = ' ' + quote_tuple[0].replace("' ", "'").replace(" '", "'")
            new_text = new_text.replace(quote_tuple[0], normalized)

        if quote_tuple[1]:
            normalized = ' ' + quote_tuple[1].replace('" ', '"').replace(' "', '"')
            new_text = new_text.replace(quote_tuple[1], normalized)
    
    # parantheses
    parans = re.findall(r" \( .+ \)", new_text)
    for paran in parans:
        normalized = ' ' + paran.replace("( ", "(").replace(" )", ")")
        new_text = new_text.replace(paran, normalized)
    
    switch_list = [
        ('.', '. '), (' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'),
        (' -', '-'), ('- ', '-'), ('$ ', '$'), (' %', '%'), (' / ', '/'),
        (" __eou__ ", "\t"), (" __eou__", ""),
    ]

    # abbreviattions
    # abbr = set(re.findall(r" [a-zA-Z]\.", new_text))

    # normalize in case of human:
    for old, new in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # get rid of surrounding whitespace
    new_text = new_text.strip()

    return new_text


def get_unique_total_ngrams(batch_generations, bos_id, eos_id, pad_id):
    assert type(batch_generations) is torch.Tensor
    batch_generations = batch_generations.cpu().numpy()

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

    
def repeated_ngrams(tensor, n):
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

    
def calc_vector_similarity(
    hidden_states,
    indices,
    padding_id=0,
    padding_mask=True,
    identical_mask=False,
    disparate=False,
    sim_threshold=0,
):
    non_padding = indices != padding_id

    sim_mask = 1 - torch.eye(indices.size(1)).to(indices.device) # don't penalise self similarity
    sim_mask = sim_mask.repeat(indices.size(0), 1, 1)
    if padding_mask: # don't calc similarity for padding tokens
        tokens_mask = non_padding.float().unsqueeze(-1).bmm(non_padding.float().unsqueeze(1)).int()
        sim_mask *= tokens_mask
    
    if identical_mask: # id_mask entails padding mask
        different_tokens = (indices.unsqueeze(-1) != indices.unsqueeze(1)).int()
        sim_mask *= different_tokens
        
    vector_represen = F.normalize(hidden_states, dim=-1)
    pair_sim = vector_represen.bmm(vector_represen.transpose(1, 2))
    # report the avg similarity of all
    cos_sim = (pair_sim * sim_mask).sum() / (sim_mask.sum() + 1e-8) # get average cosine similarity

    cut_cos_sim = None
    if disparate:
        sim_mask *= (pair_sim >= sim_threshold).int()
        # report the avg similarity of all
        cut_cos_sim = (pair_sim * sim_mask).sum() / (sim_mask.sum() + 1e-8) # get average cosine similarity

    return cos_sim, cut_cos_sim

    
def negative_sampling(
    logits=None,
    labels=None,
    pad_id=0,
    topk_negatives=0,
    preced_k_negatives=0,
):
    neg_exs = None
    topk_preds = None
    if topk_negatives > 0:
        _, topk_preds = logits.topk(k=topk_negatives)
    
    if topk_preds is not None:
        neg_exs = topk_preds
    
    preced_tokens = None
    if preced_k_negatives: # use previous k tokens as negatives
        preced_tokens = labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1))
        mask = torch.ones_like(preced_tokens).bool()
        mask = torch.ones_like(preced_tokens).tril(-1).bool()
        if preced_k_negatives > 0:
            mask = mask.triu(-preced_k_negatives)
        preced_tokens = preced_tokens.masked_fill(~mask, pad_id)

    if preced_tokens is not None:
        if neg_exs is None:
            neg_exs = preced_tokens
        else:
            neg_exs = torch.cat([neg_exs, preced_tokens], dim=-1)

    neg_exs = neg_exs.masked_fill(neg_exs == labels.unsqueeze(-1), pad_id) # exclude same label tokens as negatives

    return neg_exs


def preced_negatives(
    labels=None,
    preced_k_negatives=0,
    pad_id=0,
):
    preced_tokens = None
    if preced_k_negatives: # use previous k tokens as negatives
        preced_tokens = labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1))
        mask = torch.ones_like(preced_tokens).bool()
        mask = torch.ones_like(preced_tokens).tril(-1).bool()
        if preced_k_negatives > 0:
            mask = mask.triu(-preced_k_negatives)
        preced_tokens = preced_tokens.masked_fill(~mask, pad_id)

    if preced_tokens is not None:
        preced_tokens = preced_tokens.masked_fill(preced_tokens == labels.unsqueeze(-1), pad_id) # exclude same label tokens as negatives

    return preced_tokens


def contrastive_loss(
    logits, target_inds, orig_pad_id=0, # method='cl2',
    pad_id=0, preced_k_negatives=0,
    topk_positives=0, topk_negatives=0,
    neg_hardness=0,
):
    if preced_k_negatives:
        topk_positives = max(topk_positives, 1) # at least use top1 as positive examples

    topk = max(topk_positives, topk_negatives)
    if topk == 0: # no contrastive losses need to be calculated
        return 0.0

    labels = target_inds * (target_inds >= 0).int() # mask -100 padding tokens
    non_padding = target_inds != orig_pad_id
    topk_scores, topk_preds = logits.topk(k=topk)

    # repetition loss: using topk as positives
    preced_tokens = preced_negatives(labels, preced_k_negatives, pad_id)
    repeat_loss = 0.0
    if preced_k_negatives:
        positive_scores = topk_scores[..., :topk_positives] # using too many positives can be too harsh
        # label_scores = logits.gather(2, labels.unsqueeze(-1))
        # positive_scores = torch.cat([positive_scores, label_scores], dim=-1)
        # topk_preds = torch.cat([topk_preds, labels.unsqueeze(-1)], dim=-1)
        false_positive_mask = (preced_tokens.unsqueeze(-1) == topk_preds.unsqueeze(-2)).int()
        false_positive_mask = false_positive_mask.sum(-2).clamp(max=1)
        false_positive_mask *= (topk_preds != pad_id).int()
        false_positive_mask = false_positive_mask.unsqueeze(-2)
        pad_mask = (preced_tokens != pad_id).int()
        neg_scores = logits.gather(2, preced_tokens)
        neg_minus_pos = neg_scores.unsqueeze(-1) - positive_scores.unsqueeze(-2)
        exp = neg_minus_pos.exp()
        exp = exp * false_positive_mask
        # pad_mask *= (exp <= pos_hardness).int() # don't use too hard negatives
        sum_exp = (exp * pad_mask.unsqueeze(-1)).sum(dim=-1).sum(dim=-1) # don't use pad tokens as negatives
    
        losses = (1 + sum_exp).log() * non_padding.int()
        repeat_loss = losses.sum() / non_padding.int().sum()

    # prediction loss: using topk as negatives
    pred_loss = 0.0
    if topk_negatives:
        topk_preds = topk_preds.masked_fill(topk_preds == labels.unsqueeze(-1), pad_id) # exclude same label tokens as negatives
        pad_mask = (topk_preds != pad_id).int()
        neg_scores = topk_scores[..., :topk_negatives]
        positive_scores = logits.gather(2, labels.unsqueeze(-1))
        neg_minus_pos = neg_scores - positive_scores
        exp = neg_minus_pos.exp()
        pad_mask *= (exp <= neg_hardness).int() # using too hard negatives can be harmful
        sum_exp = (exp * pad_mask).sum(dim=-1) # don't use pad tokens as negatives

        losses = (1 + sum_exp).log() * non_padding.int()
        pred_loss = losses.sum() / non_padding.int().sum()

    return repeat_loss + pred_loss


def negative_loss(
    logits, target_inds, orig_pad_id=0, method='cl2',
    pad_id=0, topk_negatives=0, preced_k_negatives=-1,
):
    non_padding = target_inds != orig_pad_id
    labels = target_inds * (target_inds >= 0).int()

    neg_exs = negative_sampling(
        logits=logits, labels=labels, pad_id=pad_id,
        topk_negatives=topk_negatives,
        preced_k_negatives=preced_k_negatives,
    )

    negative_targets = torch.zeros_like(logits).scatter_(2, neg_exs, 1)
    negative_targets.scatter_(2, torch.zeros_like(labels).unsqueeze(-1) + pad_id, 0) # don't treat the pad_id as negative example

    if method == 'ul':
        # penalise previous tokens
        probs = logits.softmax(dim=-1)
        token_ul = -torch.log(torch.clamp(1 - probs, min=1e-20)) * negative_targets
        token_ul = token_ul.sum(dim=-1) * non_padding.int()
        ul_loss = token_ul.sum() / non_padding.int().sum()

        return ul_loss
    elif method == 'ul2':
        probs = logits.softmax(dim=-1)
        sum_neg_probs = (probs * negative_targets).sum(dim=-1)
        token_ul = -torch.log((1 - sum_neg_probs).clamp(min=1e-20)) * non_padding.int()
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
    
def compute_seq_ul(batch, model, pad_id, min_length):
    generated = model.generate.__wrapped__(
        model,
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


def calc_rep_tf_and_acc(logits, non_padding, labels):
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).int()
    incorrect = 1 - correct
    correct *= non_padding.int()
    incorrect *= non_padding.int()

    repeated = (preds.unsqueeze(-1) == preds.unsqueeze(1)).int()
    repeated = repeated.tril(-1).sum(-1).clamp(max=1)
    repeated *= incorrect

    return {
        'correct_tf': correct.sum().item(),
        'num_rep_tf': repeated.sum().item(),
        'num_total_tf': non_padding.sum().item(),
    }
