import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DialogRPTScorer(nn.Module):

    _PRIOR_MODEL_TO_WEIGHT = {
        'microsoft/DialogRPT-human-vs-rand': 0.5,
        'microsoft/DialogRPT-human-vs-machine': 0.5
    }  # yapf: disable

    _COND_MODEL_TO_WEIGHT = {
        'microsoft/DialogRPT-updown': 1,
        'microsoft/DialogRPT-width': -0.5,
        'microsoft/DialogRPT-depth': 0.48
    }  # yapf: disable

    _ALL_MODEL_NAMES = _PRIOR_MODEL_TO_WEIGHT.keys() | _COND_MODEL_TO_WEIGHT.keys()

    _SEPARATOR = '<|endoftext|>'

    def __init__(self, truncate, fp16=True):
        super().__init__()
        self.truncate = truncate
        self.fp16 = fp16

        self._models = nn.ModuleDict()
        # all tokenizers look the same, so only use one
        self._tokenizer = AutoTokenizer.from_pretrained(list(self._ALL_MODEL_NAMES)[0])
        self.separator_idx = [self._tokenizer.unk_token_id]
        for model_name in self._ALL_MODEL_NAMES:
            self._models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name)
            if self.fp16:
                self._models[model_name] = self._models[model_name].half()

    def score(self, model_inputs, model_to_weight):
        scores = {}
        # make use of past_key_values to make inference faster
        weighted_scores = 0.0
        for model_name, model_weight in model_to_weight.items():
            batch_results = self._models[model_name](**model_inputs, return_dict=True)
            batch_scores = torch.sigmoid(batch_results.logits.float()).squeeze(-1)
            scores[model_name] = np.round(batch_scores.data.cpu().numpy(), 4)
            weighted_scores += model_weight * batch_scores.data.cpu().numpy()
        return scores, weighted_scores

    @property
    def device(self):
        return next(self._models.parameters()).device
    
    @torch.no_grad()
    def score_text(self, batch_text):
        scores = {}
        model_inputs = self._tokenizer(batch_text, padding=True, max_length=self.truncate,
                            truncation='longest_first', return_tensors='pt').to(self.device)

        prior_scores, weighted_prior_scores = self.score(model_inputs, self._PRIOR_MODEL_TO_WEIGHT)
        cond_scores, weighted_cond_scores = self.score(model_inputs, self._COND_MODEL_TO_WEIGHT)
        scores.update(prior_scores)
        scores.update(cond_scores)
        scores['final'] = np.round(weighted_prior_scores * weighted_cond_scores, 4)

        return scores['final']

    @torch.no_grad()
    def batch_score(self, contexts, responses, mode='text'):
        scores = {}
        if mode == 'text':
            responses = [self._SEPARATOR + response for response in responses]
            model_inputs = self._tokenizer(contexts, responses, padding=True, max_length=self.truncate,
                                           truncation='longest_first', return_tensors='pt').to(self.device)
        elif mode == 'vector':
            input_ids = [context + self.separator_idx + response for context, response in zip(contexts, responses)]
            model_inputs = self._tokenizer.pad({'input_ids': input_ids}, return_tensors='pt').to(self.device)

        prior_scores, weighted_prior_scores = self.score(model_inputs, self._PRIOR_MODEL_TO_WEIGHT)
        cond_scores, weighted_cond_scores = self.score(model_inputs, self._COND_MODEL_TO_WEIGHT)
        scores.update(prior_scores)
        scores.update(cond_scores)
        scores['final'] = np.round(weighted_prior_scores * weighted_cond_scores, 4)

        return scores['final']
