import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, ProphetNetTokenizer
from transformers import ProphetNetForConditionalGeneration
from torch.distributions import Categorical
from torch import cuda
import math
import os
import argparse
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from data import CustomDataset, read_data, read_json
from utils import get_rouge
from reward.compute_question_focus_reward import get_question_focus_reward
from reward.compute_question_type_reward import get_question_type_reward

from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_tpu_sampler

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

device = 'cuda' if cuda.is_available() else 'cpu'


def train_batch_rl(input_ids, attention_mask, target_mask, tokenizer, max_decoding_step=15, greedy=True):
    all_targets = []
    log_probs = []
    decoder_start_token_id = tokenizer.sep_token_id
    decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
            * decoder_start_token_id
    )

    encoder = model.get_encoder()
    encoder_outputs = encoder(input_ids, return_dict=True)

    for di in range(max_decoding_step):
        outputs = model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        use_cache=False
                        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = torch.softmax(next_token_logits, dim=-1)
        if greedy is False:
            multi_dist = Categorical(next_token_logits)
            target = multi_dist.sample()  # perform multinomial sampling
            log_prob = multi_dist.log_prob(target)
            log_probs.append(log_prob)
        else:
            target = torch.argmax(next_token_logits, dim=-1)
        target = target.detach()
        decoder_input_ids = torch.cat([decoder_input_ids, target[:, None]], dim=-1)
        all_targets.append(target)

    inds = torch.stack(all_targets, dim=1)
    if greedy is False:  # If multinomial based sampling, compute log probabilites of sampled words
        log_probs = torch.stack(log_probs, dim=1)
        log_probs = log_probs * target_mask  # Not considering sampled words with padding mask = 0
        lens = torch.sum(target_mask, dim=1)  # Length of sampled sentence
        log_probs = torch.sum(log_probs,
                              dim=1) / lens  # (bs,)                                     #compute normalizied log probability of a sentence
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in inds]
    return preds, log_probs


def get_rouge_reward(sample_sents, target_summary):
    scores = []
    for i in range(len(sample_sents)):
        try:
            score = get_rouge(target_summary[i], sample_sents[i])
            if math.isnan(score):
                score = 0.0
        except Exception:
            print("Error occured in computing rewards:")
            score = 0.0
        scores.append(score)

    r_l = torch.FloatTensor(scores).to(device)

    return r_l


def joint_reward_function(sample_sents, target_summary, gold_types=None, gold_focus=None):
    r1 = get_question_type_reward(sample_sents, target_summary, gold_types=gold_types)
    r2 = get_question_focus_reward(sample_sents, target_summary, gold_focus=gold_focus)
    r = 0.4 * r1 + 0.6 * r2
    return r


def reward_function(sample_sents, target_summary, reward_type='rouge', gold_types=None, gold_focus=None):
    if reward_type == 'rouge':
        return get_rouge_reward(sample_sents, target_summary)
    if reward_type == 'question_type':
        return get_question_type_reward(sample_sents, target_summary, gold_types=gold_types)
    if reward_type == 'question_focus':
        return get_question_focus_reward(sample_sents, target_summary, gold_focus=gold_focus)


def train_rl(epoch, tokenizer, device, loader, optimizer,
             scheduler, accumulation_steps=4, max_decoding_step=15,
             semantic_rewards='question_type'
             ):
    model.train()
    optimizer.zero_grad()
    avg_mle_loss = 0.0
    avg_rl_loss = 0.0
    avg_total_loss = 0.0
    avg_rewards = 0.0

    for index, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        target_mask = data['target_mask'].to(device, dtype=torch.long)
        gold_types = data['types']
        gold_focus = data['focus']

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, use_cache=False)
        lm_logits = outputs["logits"]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        mle_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))

        sample_sents, RL_log_probs = train_batch_rl(ids, mask, target_mask, tokenizer,
                                                    max_decoding_step=max_decoding_step, greedy=False)

        with torch.autograd.no_grad():
            # greedy sampling
            greedy_sents, _ = train_batch_rl(ids, mask, target_mask, tokenizer, max_decoding_step=max_decoding_step,
                                             greedy=True)

        target_summary = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

        if semantic_rewards == 'question_type':
            sample_reward = reward_function(sample_sents, target_summary, reward_type='question_type', gold_types=gold_types)
            baseline_reward = reward_function(greedy_sents, target_summary, reward_type='question_type', gold_types=gold_types)

        elif semantic_rewards == 'question_focus':
            sample_reward = reward_function(sample_sents, target_summary, reward_type='question_focus',  gold_focus=gold_focus)
            baseline_reward = reward_function(greedy_sents, target_summary, reward_type='question_focus',  gold_focus=gold_focus)

        elif semantic_rewards == 'both':
            sample_reward = joint_reward_function(sample_sents, target_summary, gold_types=gold_types, gold_focus=gold_focus)
            baseline_reward = joint_reward_function(greedy_sents, target_summary, gold_types=gold_types, gold_focus=gold_focus)
        else:
            print("Something wrong with rewards...")
            exit()

        rl_loss = -(
                sample_reward - baseline_reward) * RL_log_probs  # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)

        rl_loss = torch.mean(rl_loss)

        if rl_loss < 0.0:
            rl_loss = torch.tensor(0.0).to(device)

        batch_reward = torch.mean(sample_reward).item()

        total_loss = 0.05 * mle_loss + 0.95 * (rl_loss)

        avg_mle_loss += mle_loss.item()
        avg_rl_loss += rl_loss.item()
        avg_rewards += batch_reward
        avg_total_loss += total_loss

        total_loss = total_loss / accumulation_steps

        total_loss.backward()

        if (index + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if index % 10 == 0:
            print(f"Completed {index} for epoch {epoch}")

    return avg_mle_loss / len(loader), avg_rl_loss / len(loader), avg_total_loss / len(loader), avg_rewards / len(
        loader)


def train_mle(epoch, tokenizer, device, loader, optimizer, scheduler):
    model.train()
    avg_mle_loss = 0.0
    for index, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, use_cache=False)
        lm_logits = outputs["logits"]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))

        avg_mle_loss += loss.item()
        if index % 10 == 0:
            print(f"Completed {index} for epoch {epoch}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return avg_mle_loss / len(loader)


def validate(tokenizer, device, loader, max_generated_length=15):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_generated_length,
                num_beams=4,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 10 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_src_len", type=int, default=120)
    parser.add_argument("--summary_len", type=int, default=20)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size for one process.", )
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Learning rate.")
    parser.add_argument("--seed", type=int, help="Random seed.", default=42)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default='models/meqsum')
    parser.add_argument("--dataset_dir", type=str, help="Directory for dataset.", default='dataset/meqsum')
    parser.add_argument("--model_filename", type=str, help="Model filename to save the model.", default='meqsum-model')
    parser.add_argument("--trained_model_path", type=str, help="Path to the train model using MLE.")

    parser.add_argument("--semantic_rewards", type=str, help="Semantic rewards (question_type/question_focus/both)",
                        default='question_type')

    parser.add_argument("--mode", default='train', type=str, help="train/test")
    parser.add_argument("--train_mode", default='mle', type=str, help="mle/rl")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    global tokenizer, model
    if args.mode == 'train':

        DATA_TRAIN = os.path.join(args.dataset_dir, 'train.json')
        DATA_VAL = os.path.join(args.dataset_dir, 'train.json')

        if args.train_mode == 'mle':
            # tokenzier for encoding the text
            tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
            model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")
            model = model.to(device)
        else:
            tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
            if os.path.exists(args.trained_model_path):
                model = ProphetNetForConditionalGeneration.from_pretrained(args.trained_model_path)
                model = model.to(device)
            else:
                print("Please provide the path to trained MLE model to train MLE+RL model")
                exit()


        train_dataset, max_src_train, max_tgt_train = read_json(DATA_TRAIN, max_src=args.max_src_len,
                                                                max_tgt=args.summary_len)
        val_dataset, max_src_dev, max_tgt_dev = read_json(DATA_VAL, max_src=args.max_src_len, max_tgt=args.summary_len)
        # Creating the Training, Validation and Test dataset for further creation of Dataloader
        original_length = len(train_dataset)
        num = int(original_length / args.batch_size)
        new_length = num * args.batch_size
        print(f"Original length:{original_length}")
        print(f"New length: {new_length}")
        training_set = CustomDataset(train_dataset[:new_length], tokenizer, args.max_src_len, args.summary_len)
        val_set = CustomDataset(val_dataset, tokenizer, args.max_src_len, args.summary_len)

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 0
        }

        val_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 0
        }

        # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
        training_loader = DataLoader(training_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)

        # Defining the model.
        # Further this model is sent to device (GPU/TPU) for using the hardware.

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # Defining the optimizer that will be used to tune the weights of the network in the training session.
        optimizer = torch.optim.Adam(params=optimizer_grouped_parameters, lr=args.learning_rate)
        num_training_steps = args.batch_size * (len(training_loader) / args.batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(num_training_steps))

        if args.train_mode == 'mle':
            print("Starting MLE training...")
            best_r_l = 0.0
            for epoch in range(args.num_epochs):
                try:
                    mle_loss = train_mle(epoch, tokenizer, device, training_loader, optimizer, scheduler)
                    print(f"MLE loss: {mle_loss}")

                except KeyboardInterrupt:
                    print("-------------------Keyboard Interrupt------------------")
                    exit(0)

                print("Evaluating on dev set...")
                predictions, actuals = validate(tokenizer, device, val_loader,
                                                max_generated_length=args.summary_len)

                r_l = get_rouge(predictions, actuals, is_print=True)
                if r_l > best_r_l:
                    best_r_l = r_l
                    print("Saving the best MLE model.")
                    model_file_name = os.path.join(args.model_dir, args.model_filename)
                    model_file_name = model_file_name + '-epoch-' + str(epoch)
                    print(f"Saved the best MLE model as epoch {model_file_name}.")

        if args.train_mode == 'rl':
            print('Initiating Fine-Tuning for the model on our dataset')
            print('Best MLE model results:')
            predictions, actuals = validate(tokenizer, device, val_loader,
                                            max_generated_length=args.summary_len)
            print(f"Validation output length: {len(predictions)}")
            print("Evaluating on val set...")
            r_l = get_rouge(predictions, actuals, is_print=True)

            print("Done with evaluation, starting MLE + RL training...")
            best_r_l = 0.0
            for epoch in range(args.num_epochs):
                try:
                    mle_loss, rl_loss, total_loss, reward = train_rl(epoch, tokenizer, device,
                                                                     training_loader, optimizer,
                                                                     scheduler,
                                                                     accumulation_steps=args.accumulation_steps,
                                                                     max_decoding_step=args.summary_len,
                                                                     semantic_rewards=args.semantic_rewards

                                                                     )

                    print(f"MLE loss: {mle_loss}, RL_loss: {rl_loss}, Total_loss: {total_loss}, reward: {reward}")

                except KeyboardInterrupt:
                    print("-------------------Keyboard Interrupt------------------")
                    exit(0)

                print("Evaluating on dev set...")
                predictions, actuals = validate(tokenizer, device, val_loader,
                                                max_generated_length=args.summary_len)

                r_l = get_rouge(predictions, actuals, is_print=True)
                if r_l > best_r_l:
                    best_r_l = r_l
                    print("Saving the best MLE+RL model.")
                    model_file_name = os.path.join(args.model_dir, args.model_filename)
                    model_file_name = model_file_name + '-epoch-' + str(epoch)
                    print(f"Saved the best MLE + RL model as epoch {model_file_name}.")

    if args.mode == 'test':
        SOURCE_TEST = os.path.join(args.dataset_dir, 'test.source')
        TARGET_TEST = os.path.join(args.dataset_dir, 'test.target')
        tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        if os.path.exists(args.trained_model_path):
            model = ProphetNetForConditionalGeneration.from_pretrained(args.trained_model_path)
            model = model.to(device)
        else:
            print("Model path does not exist...")
            exit()

        file_test = (SOURCE_TEST, TARGET_TEST)
        test_dataset, max_src_test, max_tgt_test = read_data(file_test, max_src=args.max_src_len,
                                                             max_tgt=args.summary_len)
        test_set = CustomDataset(test_dataset, tokenizer, args.max_src_len, args.summary_len)

        test_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 0
        }

        test_loader = DataLoader(test_set, **test_params)

        print('Evaluating the model performance:')
        predictions, actuals = validate(tokenizer, device, test_loader,
                                        max_generated_length=args.summary_len)
        print(f"Test output length: {len(predictions)}")
        r_l = get_rouge(predictions, actuals, is_print=True)
        print("Done...")


if __name__ == '__main__':
    main()
