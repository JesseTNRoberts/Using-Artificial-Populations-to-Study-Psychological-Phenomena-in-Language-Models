import os
import argparse
import csv
from tqdm import tqdm

import random

import torch
from torch.utils.data import DataLoader

from minicons import scorer
import PopulationLM as pop

committee_size = 50

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
results_dir = f"results/{dataset_name}"

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


# convert the internal model to use MC Dropout
pop.DropoutUtils.convert_dropouts(transformer.model)
pop.DropoutUtils.activate_mc_dropout(transformer.model, activate=True, random=0.1)

results = []
control_results = []
conclusion_only = []

# create a lambda function alias for the method that performs classifications
call_me = lambda prefixes, queries: transformer.conditional_score(prefixes, queries, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))

for batch in tqdm(stimuli_loader):
    premise = list(batch[0])
    priming = list(batch[1])
    preamble = [s1 + s2 for s1, s2 in zip(priming, premise)]
    print(preamble)
    conclusion = list(batch[2])

    # create the population identities
    population = pop.generate_dropout_population(transformer.model, lambda: call_me(preamble, conclusion), committee_size=committee_size)


    outs = [item for item in pop.call_function_with_population(transformer.model, population, lambda: call_me(preamble, conclusion))]
    transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]
    priming_scores = [score for score in transposed_outs]
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
