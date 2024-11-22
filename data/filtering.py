import math
from collections import defaultdict

import torch
import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer

from dataloader import GPT2Dataset


def tf_idf(query_tokens, document_tokens):
    score = 0
    for tok in query_tokens:
        if tok in document_tokens:
            count = document_tokens.count(tok)
            tf = count*2.5/(count + 1.5*(0.25+0.75*(len(document_tokens)/avg)))
            idf = math.log(1 + (100000 - idf_scores[tok] + 0.5)/(0.5 + idf_scores[tok]))
            score += tf*idf
    return score


def calc_idf(query_tokens):
    idf_scores = defaultdict(int)
    sum = 0
    for i, (data, target) in enumerate(tqdm.tqdm(dataset)):
        document_tokens = [data[0].item()] + target.tolist()
        if 50256 in document_tokens:
            pos = document_tokens.index(50256)
            document_tokens = document_tokens[:pos]
        sum += len(document_tokens)
        for tok in set(query_tokens):
            if tok in document_tokens:
                idf_scores[tok] += 1
        if i == 100000:
            break
    avg = sum/100000
    return avg, idf_scores


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")


raw_dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, streaming=True)['train']
tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
dataset = GPT2Dataset(tokenized_dataset)

# f = open('query_data.txt', 'r').readlines()
# query = f[0] + f[1][:-1]
# tokenized_query = tokenizer(query, truncation=True, max_length=17, padding="max_length")['input_ids']
# tokenized_query = [tokenized_query[3]] + [tokenized_query[6]] + [tokenized_query[-1]]

# avg, idf_scores = calc_idf(tokenized_query)

# all_scores = []
all_scores = torch.load('scores.pt')
all_scores.sort(reverse=True)
idxs = [i for score, i in all_scores[:100] if score > 0]

for i, (data, target) in tqdm.tqdm(enumerate(dataset)):
    if i in idxs:
        f = open('filtered_docs/document_{}.txt'.format(i), 'w', encoding='utf-8')
        document_tokens = [data[0].item()] + target.tolist()
        if 50256 in document_tokens:
            pos = document_tokens.index(50256)
            document_tokens = document_tokens[:pos]
        document = tokenizer.decode(document_tokens)
        f.write(document)
        f.close()
    if i == 100000:
        break

    # document_tokens = [data[0].item()] + target.tolist()
    # if 50256 in document_tokens:
    #     pos = document_tokens.index(50256)
    #     document_tokens = document_tokens[:pos]
    # score = tf_idf(tokenized_query, document_tokens)
    # all_scores.append((score, i))
    # if i == 100000:
    #     break

# torch.save(all_scores, 'scores.pt')
