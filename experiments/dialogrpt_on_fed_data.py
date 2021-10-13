import json

import numpy as np
import requests
import torch
import tqdm
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.dialog_rpt_scorer import DialogRPTScorer


# Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

model = DialogRPTScorer(80, fp16=False).cuda()

@torch.no_grad()
def score(text):
    score = model.score_text(text)
    return score.item()

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
