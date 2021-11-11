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

# Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

data = json.loads(r.content)

model_path = os.path.dirname(__file__) + '/PredictiveEngagement/model/best_model_finetuned.pt'


model = BiLSTM(mlp_hidden_dim=[64, 32, 8], dropout=0.8)
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

bert_name = 'bert-base-uncased' # -cased not working
tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert = BertModel.from_pretrained(bert_name).cuda()

@torch.no_grad()
def score(context, response):
    context_input = tokenizer(list(context), padding=True, truncation=True, return_tensors='pt').to(bert.device)
    response_input = tokenizer(list(response), padding=True, truncation=True, return_tensors='pt').to(bert.device)
    context_emb = bert(**context_input).last_hidden_state.mean(dim=1).cpu().numpy()
    response_emb = bert(**response_input).last_hidden_state.mean(dim=1).cpu().numpy()

    context_emb_dict = {key: val for key, val in zip(context, context_emb)}
    response_emb_dict = {key: val for key, val in zip(response, response_emb)}
    model_output = model(context, response, context_emb_dict, response_emb_dict)
    if model_output.ndim < 2:
        model_output = model_output.unsqueeze(0)
    pred_eng = torch.nn.functional.softmax(model_output, dim=1)[:, 1]
    return pred_eng.cpu().numpy()

# model_scores = []
# human_scores = []
# for row in tqdm.tqdm(data):
#     if 'response' in row: # this is a turn annotation, add it to dict
#         context = row['context'].replace('User: ', '').replace('System: ', '')
#         context = context.split('\n')[-1]
#         response = row['response'].replace('System: ', '')
#         engaging = row['annotations']['Engaging']
#         avg_engaging = np.mean(engaging)
#         human_scores.append(avg_engaging)
        
#         engaging = score([context], [response])
#         model_scores.append(engaging)

# spearman = spearmanr(model_scores, human_scores)
# pearson = pearsonr(model_scores, human_scores)

# print(f'Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
# print(f'Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')

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

for module in [fed, daily_dialog_engaging]:
    model_scores = []
    human_scores = []
    hist_sz = 2
    name = module.__name__.split('.')[-1]
    dataset = load_my_dataset(
                        module,
                        name=name,
                        split='test',
                        history_delimeter= ' <|endoftext|> ',
                        history_size=hist_sz,
                        script_version=f'histsz_{hist_sz}',
                    )

    bsz = 20
    dataset = dataset.sort('turn_id', reverse=True)
    num_batches = ceil(len(dataset) / bsz)
    for i in tqdm.tqdm(range(num_batches)):
        batch = dataset[i*bsz : (i+1)*bsz]
        texts = batch['text']
        texts = [(text.split(' <|endoftext|> ')) for text in texts]
        texts = [text if len(text) == 2 else ('', text[0]) for text in texts]
        contexts, responses = zip(*texts)
        human_scores.extend(batch['label'])
        
        engaging = score(contexts, responses)
        model_scores.extend(engaging)

    spearman = spearmanr(model_scores, human_scores)
    pearson = pearsonr(model_scores, human_scores)

    print(f'{name} Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
    print(f'{name} Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
