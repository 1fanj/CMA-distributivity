#!/bin/sh

python3 data_generation_v2.py --cfg_path ../data/grammar.cfg --dist_path ../data/dist_pred_list.csv --ratio 2 --control_output_path ../output/control.tsv --intervention_output_path ../output/intervention.tsv
