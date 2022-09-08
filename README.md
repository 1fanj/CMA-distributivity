# CMA-distributivity
Code for [Testing Pre-trained Language Models' Understanding of Distributivty via Causal Mediation Analysis](arxiv)

## Installation
This repo is built based on Python (version).

The dependencies can be installed through `requirement.txt`

## Data
We release the control and intervention sets of the dataset DistNLI, which can be found under `data\DistNLI` folder. Both sets do not come with golden labels due to the complication and controversy of distributivity.

We also release the grammar (`data\DistNLI`), the predicates list (`data\DistNLI`), and scripts (`src`) of creating these examples. You can customize and create new datasets by running the bash script.
