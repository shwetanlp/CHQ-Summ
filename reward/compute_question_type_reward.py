import torch
import numpy as np
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from sklearn.metrics import f1_score
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_lables():
    label2id = {'doses': 0, 'drug': 1, 'diagnosis': 2, 'treatments': 3, 'duration': 4, 'testing': 5, 'symptom': 6, 'uses': 7, 'information': 8, 'causes': 9}
    # print(label2id)
    return label2id



class BertPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TypeClassifier(torch.nn.Module):
    def __init__(self, model, n_labels):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TypeClassifier, self).__init__()
        self.bert_model = model
        self.pooling = BertPooler(self.bert_model.base_model.config)
        self.dropout = torch.nn.Dropout(self.bert_model.base_model.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert_model.base_model.config.hidden_size, n_labels)
        self.loss = torch.nn.BCELoss()
    def forward(self, input_ids, attention_mask, target):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs['hidden_states'][-1]
        pooled_output = self.pooling(last_hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        predictions= torch.sigmoid(logits)
        if target is not None:
            loss = self.loss(predictions, target)
        else:
            loss = None
        return predictions, loss


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
labels2id = get_lables()
type_model = TypeClassifier(model=model, n_labels=len(labels2id))
type_model.to(device)
type_model.load_state_dict(torch.load('type_model/epochs-7'))
type_model.eval()

def preprocess_dataset_for_type(items, labels=None, max_len=20):
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

    source_ids = torch.tensor(inputs, dtype=torch.long)
    src_mask = torch.tensor(attention_masks, dtype=torch.long)  # might need to squeeze
    if labels is not None:
        gold_labels = torch.tensor(labels, dtype=torch.float)
    else:
        gold_labels = None
    return {"source_ids": source_ids, "source_mask": src_mask, "labels": gold_labels}


def filter_null(type_list):
    filtered_list = []
    for item in type_list:
        if item != 'NULL':
            filtered_list.append(item)
    return filtered_list

def get_question_type_reward(sample_sents, target_summary, gold_types=None):
    type_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    predicted_labels = []
    gold_labels = []
    gold_types = filter_null(gold_types)
    for text, target in zip(target_summary, gold_types):

        # tokenize targets
        new_target = []
        for item in target:
            if item not in labels2id:
                item = 'information'
            new_target.append(item)

        targets = [0] * len(labels2id)
        for key, value in labels2id.items():
            if key in new_target:
                targets[value] = 1
        gold_labels.append(targets)

    batch = preprocess_dataset_for_type(sample_sents)
    with torch.no_grad():
        input_ids = batch['source_ids'].to(device)
        attention_mask = batch['source_mask'].to(device)
        outputs, loss = type_model(input_ids, attention_mask, None)
    scores = []
    index = 0

    for item in outputs:
        pred = item.cpu().tolist()
        cap_pred = []
        for p in pred:
            if p > 0.4:
                cap_pred.append(1)
            else:
                cap_pred.append(0)

        gold = gold_labels[index]
        cap_pred = np.array(cap_pred)
        gold = np.array(gold)
        f1 = f1_score(gold, cap_pred, average='weighted')
        scores.append(f1)

    batch_reward = torch.FloatTensor(scores).to(device)
    return batch_reward
