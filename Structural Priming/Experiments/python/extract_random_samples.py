import os
import argparse
import csv
from tqdm import tqdm

import random


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
parser.add_argument("--out_file", type = str)
parser.add_argument("--shuffled", type = bool, default= True)
args = parser.parse_args()

dataset = []
with open(args.dataset, "r") as f:
    reader = csv.DictReader(f)
    column_names = reader.fieldnames
    for row in reader:
        dataset.append(list(row.values()))

seen = set()
repeat_rows = []
selected_rows = []
# input dataset repeats each target 10 times. This extracts just one example pairing from each target group
for i in range(0, len(dataset), 10): 
    row = dataset[i]
    # move extra repeats to the end for ease of exclusion if desired
    if row[2] in seen:
        repeat_rows.append(row)
    else:
        seen.add(row[2])
        selected_rows.append(row)

if args.shuffled:
    random.shuffle(selected_rows)

selected_rows += repeat_rows


with open(args.out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)
    writer.writerows(selected_rows)