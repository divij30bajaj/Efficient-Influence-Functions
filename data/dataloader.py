import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class GPT2Dataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __iter__(self):
        for item in self.tokenized_dataset:
            input_ids = item['input_ids']
            data = input_ids[:-1]
            target = input_ids[1:]
            yield torch.tensor(data, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# def load_train_data(sample_size):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokenizer.pad_token = tokenizer.eos_token
#
#     def preprocess_function(examples):
#         return tokenizer(examples["text"], truncation=True, max_length=10, padding="max_length")
#
#     raw_dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, streaming=True)['train']
#     tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
#     dataset = GPT2Dataset(tokenized_dataset)
#     train_data = []
#     for data, target in dataset:
#         train_data.append((data, target))
#         if len(train_data) == sample_size:
#             break
#     return tokenizer, train_data

def load_train_data(sample_size, max_length):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_docs = []
    for file in os.listdir('filtered_docs'):
        f = open("filtered_docs/" + file, "r", encoding='utf-8')
        document = f.read()
        document = tokenizer(document, truncation=True, max_length=max_length, padding="max_length")
        tokenized_docs.append(document)

    dataset = GPT2Dataset(tokenized_docs)
    train_data = []
    for data, target in dataset:
        train_data.append((data, target))
        if len(train_data) == sample_size:
            break
    return tokenizer, train_data


def load_test_data(sample_size):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    f = open('data/query_data.txt', 'r').readlines()
    queries = []

    for i in range(0, len(f), 2):
        query = f[i] + f[i+1][:-1]
        tokenized_query = tokenizer(query, truncation=True, max_length=16, padding="max_length")
        queries.append(tokenized_query)

    dataset = GPT2Dataset(queries)
    test_data = []
    for data, target in dataset:
        test_data.append((data, target))
        if len(test_data) == sample_size:
            break
    return test_data


def decode(token_ids):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer.decode(token_ids)
