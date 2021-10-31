import json
from math import ceil

import tqdm

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

for module in [personachat_engaging]:
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
    tgt_file = open(f'./data/{name}.jsonl', 'w')
    for data in tqdm.tqdm(dataset):
        text = data['text'].split(' [SEP] ')
        if len(text) < 2:
            continue
        tgt_file.write(json.dumps({'context': text[0], 'response': text[1], 'label': data['label'], 'dialog_id': data['dialog_id'], 'turn_id': data['turn_id']}) + '\n')
    tgt_file.close()
