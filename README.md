# CMA-distributivity
Code for [Testing Pre-trained Language Models' Understanding of Distributivty via Causal Mediation Analysis](arxiv)

## Installation
This repo is built based on Python >=3.8.

The dependencies can be installed by running `pip install -r requirement.txt`.

## Data
We release the control and intervention sets of the dataset DistNLI, which can be found at [data/DistNLI/DistNLI/control.tsv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/control.tsv) and [data/DistNLI/intervention.tsv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/intervention.tsv) respectively. Both sets do not come with golden labels due to the complication and controversy of distributivity.

We also release the list of distributive predicates [dist_pred_list.csv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/dist_pred_list.csv), list of ambiguous predicates [amb_pred_list.csv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/amb_pred_list.csv), and list of grammar [grammar.cfg](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/grammar.cfg) used to generate our dataset under `data/DistNLI/`, as well as scripts to create these examples [run_cma.py](https://github.com/aponimma/CMA-distributivity/blob/main/src/run_cma.py) under `src`. Note that in the distributive predicate list, the first line should be predicates that do not differentiate singular and plural subjects, and the second line should be predicates that do so. You can customize and create new datasets by running `src/data_generation.sh`.

## CMA Experiment
We release the code for the CMA experiment under `src`. To obtain total effect and natural indirect effect, you can run `run_cma.sh`, in which `--model_name` is the model you want to use and `--result_dir` is the directory you save the results. If the results are saved in the directory, you can simply load them using the same arguments without running the experiment again.
