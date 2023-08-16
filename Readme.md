## Replication Repository - Using Artificial Populations to Study Psychological Phenomena in Language Models

This repository holds the code necessary to recreate the experiments for "Using Artificial Populations to Study Psychological Phenomena in Language Models"

### Dependencies

The code dependencies and the hardware used are listed below.

#### PopulationLM

In Using Artificial Populations to Study Psychological Phenomena in Language Models, we propose an extension to mc dropout called stratified MC dropout. These experiments utilize that software. The code for it can be found here: https://github.com/JesseTNRoberts/PopulationLM

#### Minicons

We use the minicons software package as a convenient tool to gather probability data for sequences from language models. However, we found it necessary to modify the software. To replicate these experiments, install the version here: https://github.com/JesseTNRoberts/minicons_modded

For reference, the original can be found here: https://github.com/kanishkamisra/minicons

#### Other Utilities

- huggingface-0.0.1
- pandas-1.5.3
- numpy-1.23.5
- scipy-1.10.1
- researchpy-0.3.5
- plotly-5.15.0
- seaborn-0.12.2
- matplotlib-3.7.1
- tqdm-4.65.0
- torch-1.13.1
- huggingface-hub-0.16.4
- transformers-4.31.0

### Computer Architecture

All data analysis was run in google colab with the following specs:
- CPU - 2x AMD EPYC 7B12
- GPU - None
- Memory - 13 GB
- OS - Ubuntu-22.04
- Python 3.10.12


The experiments were conducted on a machine with the following specs:
- CPU - Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, 2592 Mhz, 6 Core(s), 12 Logical Processor(s)
- GPU - NVIDIA GeForce RTX 2060
- Memory - 16GB (physical)
- OS - Windows 10 Version 10.0.19045 Build 19045
- python 3.9.5


