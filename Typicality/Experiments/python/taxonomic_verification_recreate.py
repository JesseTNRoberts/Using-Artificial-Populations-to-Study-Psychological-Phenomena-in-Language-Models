import os
import argparse
import csv
from tqdm import tqdm

import random

import torch
from torch.utils.data import DataLoader

from minicons import scorer


def shuffle_sentence(sentence, word):
    '''
        returns the shuffled form of a sentence while preserving the 
        multi-word expression order for the focus word.
    '''
    sentence = sentence.replace(".", "")
    if len(word.split()) > 1:
        sentence = sentence.replace(word, "@".join(word.split())).split()
    else:
        sentence = sentence.split()
    random.shuffle(sentence)
        
    return " ".join(sentence).replace("@", " ").capitalize() + "."

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
parser.add_argument("--model", default = 'distilbert-base', type = str)
parser.add_argument("--batchsize", default = 10, type = int)
parser.add_argument("--device", default = 'cpu', type = str)
parser.add_argument("--lmtype", default = 'masked', choices = ['mlm', 'masked', 'causal', 'incremental'], type = str)
args = parser.parse_args()

inpath = args.dataset
model_name = args.model
batch_size = args.batchsize
device = args.device
lm_type = args.lmtype

# make results dir: ../data/typicality/results/(dataset)/model_name.csv
components = inpath.split("/")
data_dir = "/".join(components[0:-1])
dataset_name = components[-1].split(".")[0]
results_dir = f"{data_dir}/results/recreation/{dataset_name}"

dataset = []
with open(args.dataset, "r") as f:
    reader = csv.DictReader(f)
    column_names = reader.fieldnames
    for row in reader:
        dataset.append(list(row.values()))

if lm_type == "masked" or lm_type == "mlm":
    transformer = scorer.MaskedLMScorer(model_name, device)
elif lm_type == "incremental" or lm_type == "causal":
    transformer = scorer.IncrementalLMScorer(model_name, device)

if "/" in model_name:
    model_name = model_name.replace("/", "_")

num_params = [sum(p.numel() for p in transformer.model.parameters())] * len(dataset)

stimuli_loader = DataLoader(dataset, batch_size = batch_size, num_workers=0)

results = []
control_results = []
conclusion_only = []

for batch in tqdm(stimuli_loader):
    premise = list(batch[0])
    conclusion = list(batch[1])
    print(conclusion[0])
    priming_scores = transformer.conditional_score(premise, conclusion, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))
    priming_scores = [score for score in priming_scores]
    results.extend(priming_scores)

dataset = list(zip(*dataset))
dataset.append(results)


dataset.append(num_params)
dataset.append([model_name] * len(results))

column_names += ["score (sum, mean, [list)"]
column_names += ["params", "model"]

with open(results_dir + f"/{model_name}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(column_names)
    writer.writerows(list(zip(*dataset)))
