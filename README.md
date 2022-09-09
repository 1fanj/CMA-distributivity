# CMA-distributivity
Code for [Testing Pre-trained Language Models' Understanding of Distributivty via Causal Mediation Analysis](arxiv)

## Installation
This repo is built based on Python (version).

The dependencies can be installed through `requirement.txt`

## Data
We release the control and intervention sets of the dataset DistNLI, which can be found under `data\DistNLI` folder. Both sets do not come with golden labels due to the complication and controversy of distributivity.

We also release the grammar (`data\DistNLI`), the predicates list (`data\DistNLI`), and scripts (`src`) of creating these examples. You can customize and create new datasets by running the bash script.

## CMA Experiment
We release the code for the CMA experiment, which can be found under `src` folder. To obtain total effect and natural indirect effect, run `run_cma.sh` with `--model_name` being the model you want to use and `--result_dir` being the directory you save the results. If the results are saved in the directory, you can simply load them using the same arguments without running the experiment again.
