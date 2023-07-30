import os
import argparse
import csv
from tqdm import tqdm

import random


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
parser.add_argument("--out_file", type = str)
parser.add_argument("--num_samples", default = 500, type = int)
args = parser.parse_args()

dataset = []
with open(args.dataset, "r") as f:
    reader = csv.DictReader(f)
    column_names = reader.fieldnames
    for row in reader:
        dataset.append(list(row.values()))



with open(args.out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)
    writer.writerows(random.sample(dataset, args.num_samples))