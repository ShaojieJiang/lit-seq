import json

import numpy as np
import requests
from scipy.stats import pearsonr, spearmanr


_URL = "http://shikib.com/fed_data.json"

r = requests.get(_URL)

data = json.loads(r.content)
dialogues = []
turn_anno_dict = {}
for row in data:
    # if row['system'] != 'Human':
    #     continue
    if 'response' in row: # this is a turn annotation, add it to dict
        text = f"{row['context']}\n{row['response']}"
        engaging = row['annotations']['Engaging']
        avg_engaging = np.mean(engaging) * 5
        if text in turn_anno_dict:
            print(text)
        turn_anno_dict[text] = avg_engaging
    else: # this is a dialogue
        dialogues.append(row['context'])

depth_list = []
engaging_list = []
for text in turn_anno_dict.keys():
    for dialogue in dialogues:
        if text in dialogue:
            rest_dialogue = dialogue[len(text) + 1:]
            depth = len(rest_dialogue.split('\n'))
            depth_norm10 = depth / (len(dialogue.split('\n')) - 1) * 10
            depth_list.append(depth_norm10)
            engaging_list.append(turn_anno_dict[text])

spearman = spearmanr(depth_list, engaging_list)
pearson = pearsonr(depth_list, engaging_list)

print(f'Spearman correlation: {spearman[0]}, p-val: {spearman[1]}')
print(f'Pearson correlation: {pearson[0]}, p-val: {pearson[1]}')
