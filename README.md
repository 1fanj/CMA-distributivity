# CMA-distributivity
Code for [Testing Pre-trained Language Models' Understanding of Distributivty via Causal Mediation Analysis](https://arxiv.org/abs/2209.04761)

## Installation
This repo is built based on Python>=3.8.

The dependencies can be installed by running `pip install -r requirements.txt`.

## Data Generation
We release the control and intervention sets of the dataset DistNLI, which can be found at [data/DistNLI/DistNLI/control.tsv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/control.tsv) and [data/DistNLI/intervention.tsv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/intervention.tsv) respectively. Both sets do not come with golden labels due to the complication and controversy of distributivity.

We also release the list of distributive predicates [dist_pred_list.csv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/dist_pred_list.csv), the full list of predicates [full_pred_list.csv](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/full_pred_list.csv), and list of grammar [grammar.cfg](https://github.com/aponimma/CMA-distributivity/blob/main/data/DistNLI/grammar.cfg) used to generate our dataset under `data/DistNLI/`, as well as scripts to create these examples [data_generation.py](https://github.com/aponimma/CMA-distributivity/blob/main/src/data_generation.py) under `src`. Note that in the distributive predicate list, the first line should be predicates that do not differentiate singular and plural subjects, and the second line should be predicates that do so. You can customize and create new datasets by running `src/data_generation.sh`.

## Pre-examination
In all [results](https://github.com/aponimma/CMA-distributivity/blob/main/results/), `contradiction` is labeled as `0`, `neutral` is labeled as `1`, and `entailment` is labeled as `2`. They follow the format of `predicted_label, true_label, [prob_of_label_0, prob_of_label_1, prob_of_label_2]`. 

To replicate the evaluation results, you may run: 
```
python src/evaluate_nli.py --model_name [model_name] \
                        --data_path [data_path] \
                        --dataset_type [dataset_type]
```
where `[model_name]` can be any huggingface model, `[data_path]` is the path to the dataset, and `[dataset_type]` can be either `ConjNLI`, `HANS`, or `distnli`. Sample bash script can be accessed at [evaluate_nli.sh](https://github.com/aponimma/CMA-distributivity/blob/main/src/evaluate_nli.sh). 


## CMA Experiment
We release the code for the CMA experiment under `src`. To obtain total effect and natural indirect effect, you can run `run_cma.sh`, in which `--model_name` is the model you want to use and `--result_dir` is the directory you save the results. If the results are saved in the directory, you can simply load them using the same arguments without running the experiment again.
