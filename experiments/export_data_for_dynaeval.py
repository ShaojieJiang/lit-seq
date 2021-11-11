import json
import os
from math import ceil

import numpy as np
import requests
import torch
import tqdm
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr
from transformers import BertModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from experiments.PredictiveEngagement.pytorch_src.engagement_classifier import BiLSTM, Engagement_cls
from lightning_transformers.core.utils import load_my_dataset

# other datasets
from lightning_transformers.task.nlp.text_regression.datasets import (
    daily_dialog_engaging,
    fed,
    my_blended_skill_talk,
    my_daily_dialog,
    my_empathetic_dialogues,
    my_personachat,
    my_wizard_of_wikipedia,
    personachat_engaging,
)

for module in [fed, daily_dialog_engaging, personachat_engaging]:
    model_scores = []
    human_scores = []
    hist_sz = -1
    name = module.__name__.split('.')[-1]
    dataset = load_my_dataset(
                        module,
                        name=name,
                        split='test',
                        history_delimeter= ' <|endoftext|> ',
                        history_size=hist_sz,
                        script_version=f'histsz_{hist_sz}',
                    )
    tgt_file = open(f'./data/{name}.jsonl', 'w')
    for data in tqdm.tqdm(dataset):
        text = data['text'].split(' <|endoftext|> ')
        speaker = ['A' if i % 2 == 0 else 'B' for i in range(len(text))]
        tgt_file.write(json.dumps({'text': text, 'label': data['label'], 'speaker': speaker}) + '\n')
    tgt_file.close()
