import torch
import numpy as np
from collections import OrderedDict
from collections import Counter
import string
import re
import argparse
import json
import sys

from transformers import BertForTokenClassification
from transformers import BertTokenizer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def removeSublist(lst):
    curr_result = []
    result = []
    for ele in sorted(map(OrderedDict.fromkeys, lst), key=len, reverse=True):
        if not any(ele.keys() <= req.keys() for req in curr_result):
            curr_result.append(ele)
            result.append(' '.join(list(ele)))
    return result


def find( haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.

    # >>> find([1, 1, 2], [1, 2])
    # 1

    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1


def get_lables():
    labels = ['B-Focus', 'I-Focus', 'O']
    label2id = {value: index for index, value in enumerate(labels)}
    return label2id

focus_model = BertForTokenClassification.from_pretrained('focus_model/epochs-9', num_labels =3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
labels2id = get_lables()
focus_model.to(device)
focus_model.eval()

def preprocess_dataset_for_focus(items, max_len=20):
    inputs = []
    attention_masks = []
    for text in items:
        line = text.strip()
        tokenized_inputs = tokenizer.tokenize(line)
        attention_mask = [1] * len(tokenized_inputs)
        tokenized_inputs = tokenizer.convert_tokens_to_ids(tokenized_inputs)

        for i in range(len(tokenized_inputs), max_len):
            tokenized_inputs.append(tokenizer.pad_token_id)
            attention_mask.append(0)

        tokenized_inputs = tokenized_inputs[:max_len]
        attention_mask = attention_mask[:max_len]
        assert len(tokenized_inputs) == len(attention_mask)
        inputs.append(tokenized_inputs)
        attention_masks.append(attention_mask)

    source_ids =torch.tensor(inputs, dtype=torch.long)
    src_mask = torch.tensor(attention_masks, dtype=torch.long)  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask}


def merge_entities(inputs, predictions):
    entities = []
    prev_words = []
    for word, label in zip(inputs, predictions):
        if label.startswith('B'):
            if len(prev_words) > 0:
                ent_string = ''
                for item in prev_words:
                    if item.startswith('##'):
                        ent_string += item.replace('##', '')
                    else:
                        ent_string += ' ' + item
                entities.append(ent_string)
                prev_words=[]

            prev_words.append(word)
        elif label.startswith('I'):
            prev_words.append(word)
    if len(prev_words) > 0:
        ent_string = ''
        for item in prev_words:
            if item.startswith('##'):
                ent_string+=item.replace('##','')
            else:
                ent_string +=' '+item
        entities.append(ent_string)
    return entities


def filter_null(focus_list):
    filtered_list = []
    for item in focus_list:
        if item != 'NULL':
            filtered_list.append(item)
    return filtered_list


def get_question_focus_reward(sample_sents, target_summary, gold_focus=None):
    focus_model.eval()  # Turn on the evaluation mode
    gold_label = []
    idx2tag = {value: key for key, value in labels2id.items()}
    gold_focus = filter_null(gold_focus)
    for text, targets in zip(target_summary, gold_focus):
        new_targets = []
        for item in targets:
            new_targets.append(item.strip().split(' '))
        new_targets = removeSublist(new_targets)
        # print(f'original Focus after removing duplicates: {new_targets}')

        gold_label.append(' '.join(new_targets))


    scores=[]

    batch = preprocess_dataset_for_focus(sample_sents)
    with torch.no_grad():
        input_ids = batch['source_ids'].to(device)
        attention_mask = batch['source_mask'].to(device)
        outputs = focus_model(input_ids=input_ids, attention_mask=attention_mask
                              )
        logits = torch.softmax(outputs.logits, dim=-1)
        batch_output = logits.detach().cpu().numpy()
        batch_output = np.argmax(batch_output, axis=2)
        predicted_label = []
        for item in batch_output:
            temp_label=[]
            for ind in item:
                temp_label.append(idx2tag.get(ind))
            predicted_label.append(temp_label)

    input_tokens = []
    for inp in input_ids:
        tokens = tokenizer.convert_ids_to_tokens(inp)
        input_tokens.append(tokens)

    index = 0
    for input, label in zip(input_tokens, predicted_label):
        predicted_entity = merge_entities(input, label)
        # print(f"Original Predicted focus: {predicted_entity} ")
        new_predicted_enty = []
        for item in predicted_entity:
            new_predicted_enty.append(item.strip().split(' '))
        new_predicted_enty = removeSublist(new_predicted_enty)
        # print(f'predicted Focus after removing duplicates: {new_predicted_enty}')

        gold_ent = gold_label[index]
        new_predicted_entity = ' '.join(new_predicted_enty)
        # print('predicted: '+new_predicted_entity)
        # print('gold: '+gold_ent)
        f1 = f1_score(new_predicted_entity, gold_ent)
        # print(f"F1-score: {f1}")
        # print('*'*20)
        scores.append(f1)
        index += 1
    batch_reward = torch.FloatTensor(scores).to(device)
    return batch_reward




