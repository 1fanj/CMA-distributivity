#!/bin/sh

python3 run_cma.py --model_name roberta-large-mnli \
                        --control_path ../data/DistNLI/control.tsv \
                        --intervention_path ../data/DistNLI/intervention.tsv \
                        --default_label entailment \
                        --num_neuron_batch 2 \
                        --topk_neurons 0.01 \
                        --result_dir ../results/CMA