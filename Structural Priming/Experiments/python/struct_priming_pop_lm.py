import argparse
import csv
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from minicons import scorer
import PopulationLM as pop

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
parser.add_argument("--model", default = 'distilbert-base', type = str)
parser.add_argument("--batchsize", default = 10, type = int)
parser.add_argument("--num_batches", default = -1, type = int)
parser.add_argument("--committee_size", default = 50, type = int)
parser.add_argument("--device", default = 'cpu', type = str)
parser.add_argument("--lmtype", default = 'masked', choices = ['mlm', 'masked', 'causal', 'incremental'], type = str)
args = parser.parse_args()

inpath = args.dataset
model_name = args.model
batch_size = args.batchsize
num_batches = args.num_batches
committee_size = args.committee_size
device = args.device
lm_type = args.lmtype

components = inpath.split("/")
data_dir = "/".join(components[0:-1])
dataset_name = components[-1].split(".")[0]
results_dir = f"../data/results/{dataset_name}_popLM"

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


column_names += ["tx results", "ty results","px-tx results", "px-ty results", "py-tx results", "py-ty results"]
column_names += ["params", "model"]
with open(results_dir + f"/{model_name}.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)

# create a lambda function alias for the method that performs classifications
call_me = lambda p1, q1: transformer.conditional_score(p1, q1, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))

results = {'pn-tx': [], 'pn-ty': [], 'px-tx': [], 'px-ty': [], 'py-tx': [], 'py-ty': []}
if num_batches < 0:
    num_batches = len(stimuli_loader)
for i, batch in tqdm(zip(range(num_batches), stimuli_loader)):
    dataset = [[], [], [], []]
    priming_scores = []
    for i in range(4):
        dataset[i].extend(batch[i])

    for primer, target in tqdm(itertools.product(('pn', 'px', 'py'), ('tx', 'ty')), leave=False):
        if primer == 'pn':
            p_list = ('',) * batch_size
        elif primer == 'px':
            p_list = batch[0]
        elif primer == 'py':
            p_list = batch[1]
        t_list = batch[2 if target == 'tx' else 3]
        # create the population identities
        population = pop.generate_dropout_population(transformer.model, lambda: call_me(p_list, t_list), committee_size=committee_size)
        outs = [item for item in pop.call_function_with_population(transformer.model, population, lambda: call_me(p_list, t_list))]
        transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]

        priming_scores = [score for score in transposed_outs]

        results[primer+'-'+target].extend(priming_scores)

    dataset.append(results['pn-tx'])
    dataset.append(results['pn-ty'])
    dataset.append(results['px-tx'])
    dataset.append(results['px-ty'])
    dataset.append(results['py-tx'])
    dataset.append(results['py-ty'])
    dataset.append(num_params)
    dataset.append([model_name] * len(results["px-tx"]))
    with open(results_dir + f"/{model_name}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list(zip(*dataset)))


print(results_dir + f"/{model_name}.csv")