import argparse
import csv
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from minicons import scorer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type = str)
parser.add_argument("--dataset_name", type = str)
parser.add_argument("--model", default = 'distilbert-base', type = str)
parser.add_argument("--batchsize", default = 10, type = int)
parser.add_argument("--num_batches", default = -1, type = int)
parser.add_argument("--device", default = 'cpu', type = str)
parser.add_argument("--lmtype", default = 'masked', choices = ['mlm', 'masked', 'causal', 'incremental'], type = str)
args = parser.parse_args()

inpath = args.dataset_path
ds_name = args.dataset_name
model_name = args.model
batch_size = args.batchsize
num_batches = args.num_batches
device = args.device
lm_type = args.lmtype

components = inpath.split("/")
data_dir = "/".join(components[0:-1])
dataset_name = components[-1].split(".")[0]
results_dir = f"../data/results/{dataset_name}_recreate"

dataset = []
with open(inpath, "r") as f:
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


column_names += ["tx results", "ty results","px-tx results", "px-ty results", "py-tx results", "py-ty results", "ds_name"]
with open(results_dir + f"/{model_name}.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)

# create a lambda function alias for the method that performs classifications
call_me = lambda p1, q1: transformer.conditional_score(p1, q1, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))


if num_batches < 0:
    num_batches = len(stimuli_loader)
for batch in tqdm(stimuli_loader):
    out_dataset = [[], [], [], []]
    priming_scores = []
    for i in range(4):
        out_dataset[i].extend(batch[i])
        
    results = {'pn-tx': [], 'pn-ty': [], 'px-tx': [], 'px-ty': [], 'py-tx': [], 'py-ty': []}
    for primer, target in tqdm(itertools.product(('pn', 'px', 'py'), ('tx', 'ty')), leave=False):
        if primer == 'pn':
            p_list = ('',) * batch_size
        elif primer == 'px':
            p_list = batch[0]
        elif primer == 'py':
            p_list = batch[1]
        t_list = batch[2 if target == 'tx' else 3]
        # create the population identities
        #population = pop.generate_dropout_population(transformer.model, lambda: call_me(p_list, t_list), committee_size=committee_size)
        outs = [call_me(p_list, t_list)]
        transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]

        priming_scores = [score for score in transposed_outs]

        results[primer+'-'+target].extend(priming_scores)

    out_dataset.append(results['pn-tx'])
    out_dataset.append(results['pn-ty'])
    out_dataset.append(results['px-tx'])
    out_dataset.append(results['px-ty'])
    out_dataset.append(results['py-tx'])
    out_dataset.append(results['py-ty'])
    out_dataset.append([ds_name] * len(results['pn-tx']))
    with open(results_dir + f"/{model_name}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list(zip(*out_dataset)))


print(results_dir + f"/{model_name}.csv")