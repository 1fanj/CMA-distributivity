#!/bin/sh


python src/evaluate_nli.py --model_name roberta-large-mnli \
                        --data_path data/Distnli/control.tsv \
                        --dataset_type distnli