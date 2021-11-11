import json
from math import ceil

import numpy as np
import requests
import tqdm
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr

from experiments import fed_metric
from lightning_transformers.core.utils import load_my_dataset

# Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

model, tokenizer = fed_metric.load_models("microsoft/DialoGPT-large")
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

# other datasets
from lightning_transformers.task.nlp.text_regression.datasets import (
    my_blended_skill_talk,
    my_daily_dialog,
    my_empathetic_dialogues,
    my_personachat,
    my_wizard_of_wikipedia,
    daily_dialog_engaging,
    personachat_engaging,
    fed,
)

for module in [fed]:
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

    bsz = 5
    # dataset = dataset.sort('turn_id', reverse=True)
    num_batches = ceil(len(dataset) / bsz)
    for i in tqdm.tqdm(range(num_batches)):
        batch = dataset[i*bsz : (i+1)*bsz]
        texts = batch['text']
        human_scores.extend(batch['label'])
        
        scores = fed_metric.evaluate(texts,
                            model,
                            tokenizer)
        model_scores.extend(scores['engaging'])

    spearman = spearmanr(model_scores, human_scores)
    pearson = pearsonr(model_scores, human_scores)

    print(f'{name} Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
    print(f'{name} Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
