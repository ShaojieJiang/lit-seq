import json

import numpy as np
import requests
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr
import tqdm
from experiments import fed_metric

# Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

model, tokenizer = fed_metric.load_models("microsoft/DialoGPT-large")

data = json.loads(r.content)
model_scores = []
human_scores = []
for row in tqdm.tqdm(data):
    if 'response' in row: # this is a turn annotation, add it to dict
        text = f"<|endoftext|> {row['context']}\n{row['response']}".replace('User: ', '').replace('System: ', '').replace('\n', ' <|endoftext|> ')
        engaging = row['annotations']['Engaging']
        avg_engaging = np.mean(engaging)
        model_scores.append(avg_engaging)
        
        scores = fed_metric.evaluate(text,
                            model,
                            tokenizer)
        human_scores.append(scores['engaging'])

spearman = spearmanr(model_scores, human_scores)
pearson = pearsonr(model_scores, human_scores)

print(f'Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
print(f'Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
