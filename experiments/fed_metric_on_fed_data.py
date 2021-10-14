import json
from math import ceil

import numpy as np
import requests
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr
import tqdm
from experiments import fed_metric
from lightning_transformers.core.utils import load_my_dataset

# Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

model, tokenizer = fed_metric.load_models("microsoft/DialoGPT-large")
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

data = json.loads(r.content)
model_scores = []
human_scores = []
for row in tqdm.tqdm(data):
    if 'response' in row: # this is a turn annotation, add it to dict
        text = f"<|endoftext|> {row['context']}\n{row['response']}".replace('User: ', '').replace('System: ', '').replace('\n', ' <|endoftext|> ')
        engaging = row['annotations']['Engaging']
        avg_engaging = np.mean(engaging)
        human_scores.append(avg_engaging)
        
        scores = fed_metric.evaluate(text,
                            model,
                            tokenizer)
        model_scores.append(scores['engaging'])

spearman = spearmanr(model_scores, human_scores)
pearson = pearsonr(model_scores, human_scores)

print(f'FED Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
print(f'FED Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')

# daily_dialog
from lightning_transformers.task.nlp.text_regression.datasets import my_daily_dialog, my_personachat, my_blended_skill_talk, my_wizard_of_wikipedia, my_empathetic_dialogues

for module in [my_daily_dialog, my_personachat, my_blended_skill_talk, my_wizard_of_wikipedia, my_empathetic_dialogues]:
    model_scores = []
    human_scores = []
    hist_sz = 2
    name = module.__name__.split('.')[-1]
    dataset = load_my_dataset(
                        module,
                        name=name,
                        split='test',
                        history_delimeter= ' [SEP] ',
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
        
        scores = fed_metric.evaluate(texts,
                            model,
                            tokenizer)
        model_scores.extend(scores['engaging'])

    spearman = spearmanr(model_scores, human_scores)
    pearson = pearsonr(model_scores, human_scores)

    print(f'{name} Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
    print(f'{name} Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
