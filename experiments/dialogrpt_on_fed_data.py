import json
from math import ceil

import numpy as np
import requests
import torch
import tqdm
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr

from experiments.dialog_rpt_scorer import DialogRPTScorer
from lightning_transformers.core.utils import load_my_dataset

# Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

model = DialogRPTScorer(80, fp16=False).cuda()

@torch.no_grad()
def score(texts):
    score = model.score_text(texts)
    return score

data = json.loads(r.content)
model_scores = []
human_scores = []
for row in tqdm.tqdm(data):
    if 'response' in row: # this is a turn annotation, add it to dict
        text = f"{row['context']}\n{row['response']}".replace('User: ', '').replace('System: ', '')
        history = text.split('\n')[-2:]
        text = ' <|end_of_text|> '.join(history)
        # text = text.replace('\n', ' <|endoftext|> ') # using full history results in poor correlations
        engaging = row['annotations']['Engaging']
        avg_engaging = np.mean(engaging)
        human_scores.append(avg_engaging)
        
        engaging = score(text)
        model_scores.append(engaging)

spearman = spearmanr(model_scores, human_scores)
pearson = pearsonr(model_scores, human_scores)

print(f'Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
print(f'Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')

# other datasets
from lightning_transformers.task.nlp.text_regression.datasets import (
    my_blended_skill_talk,
    my_daily_dialog,
    my_empathetic_dialogues,
    my_personachat,
    my_wizard_of_wikipedia,
)

for module in [my_daily_dialog, my_personachat, my_blended_skill_talk, my_wizard_of_wikipedia, my_empathetic_dialogues]:
    model_scores = []
    human_scores = []
    hist_sz = 2
    name = module.__name__.split('.')[-1]
    dataset = load_my_dataset(
                        module,
                        name=name,
                        split='test',
                        history_delimeter= ' <|end_of_text|> ',
                        history_size=hist_sz,
                        script_version=f'histsz_{hist_sz}',
                    )

    bsz = 20
    dataset = dataset.sort('turn_id', reverse=True)
    num_batches = ceil(len(dataset) / bsz)
    for i in tqdm.tqdm(range(num_batches)):
        batch = dataset[i*bsz : (i+1)*bsz]
        texts = batch['text']
        human_scores.extend(batch['label'])
        
        engaging = score(texts)
        model_scores.extend(engaging)

    spearman = spearmanr(model_scores, human_scores)
    pearson = pearsonr(model_scores, human_scores)

    print(f'{name} Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
    print(f'{name} Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
