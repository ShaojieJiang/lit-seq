import json
import os
import numpy as np
import requests
import torch
import tqdm
from scipy.stats.mstats_basic import pearsonr
from scipy.stats.stats import spearmanr
from transformers import BertModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from experiments.PredictiveEngagement.pytorch_src.engagement_classifier import BiLSTM, Engagement_cls


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
    context_input = tokenizer(context, truncation=True, return_tensors='pt').to(bert.device)
    response_input = tokenizer(response, truncation=True, return_tensors='pt').to(bert.device)
    context_emb, response_emb = {}, {}
    context_emb[context] = bert(**context_input).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    response_emb[response] = bert(**response_input).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    model_output = model([context], [response], context_emb, response_emb)
    pred_eng = torch.nn.functional.softmax(model_output, dim=1)[0, 1].item()
    return pred_eng

model_scores = []
human_scores = []
for row in tqdm.tqdm(data):
    if 'response' in row: # this is a turn annotation, add it to dict
        context = row['context'].replace('User: ', '').replace('System: ', '')
        context = context.split('\n')[-1]
        response = row['response'].replace('System: ', '')
        engaging = row['annotations']['Engaging']
        avg_engaging = np.mean(engaging)
        human_scores.append(avg_engaging)
        
        engaging = score(context, response)
        model_scores.append(engaging)

spearman = spearmanr(model_scores, human_scores)
pearson = pearsonr(model_scores, human_scores)

print(f'Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
print(f'Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
