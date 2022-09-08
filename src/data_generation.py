#!/usr/bin/python

import argparse
import csv
import random
from typing import *
from nltk import CFG
from nltk.data import load
from nltk.parse.generate import generate

def split_premise(sentence: List[str]) -> List[str]:
    """
    args:
        sentence: premise w/ the format of 'A and B pred'
    returns:
        a list formatted in premise, hypo1, premise, hypo2
    """
    if 'were' in sentence:
        a = [sentence[0]] + ['was', sentence[4]]
        b = [sentence[2]] + ['was', sentence[4]]
    else:
        a = [sentence[0]] + sentence[3:]
        b = sentence[2:]

    return [a, b]

def split_data(sentences: List[List[str]], filepath: str, ratio: int) -> Tuple[List[List[str]], List[List[str]]]:
    """
    args:
        - sentences: list of generated sentences w/o punctuations
        - filepath of distributive predicates
        - ratio of sentences in control to intervention
    returns:
        two lists of generated sentences in control & intervention group

    split data into two groups
    """
    #set up
    control = []
    intervention = []
    hypo = []
    prem = []
    ind = 0
    random.seed(3)

    #read in dist pred data
    with open(filepath) as f:
        r = list(csv.reader(f))
        pred = r[0]
        diff = r[1]

    #separate hypothesis from premise
    for i in sentences:
        if 'and' in i and i[0] != i[2]:
            prem.append(i)
        if 'and' not in i:
            hypo.append(i)

    #randomly pick sentences from prem for control
    for j in pred:
        temp = []
        for k in prem:
            if j in k:
                temp.append(k)
        y = random.choice(temp)
        #pair up hyp1 and hyp2
        conj_a, conj_b = split_premise(y)
        control.extend([[y, conj_a], [y, conj_b], [y, conj_a], [y, conj_b]])

    for b in diff:
        tem = []
        for c in prem:
            if b in c:
                tem.append(c)
        if len(tem) != 0:
            z = random.choice(tem)
            for p in diff:
                if b != p and b.split()[0] == p.split()[0] and b.split()[-1] == p.split()[-1]:
                    new_z = z[:-1] + [p]
                    break
            sg_a, sg_b = split_premise(new_z)
            control.extend([[z, sg_a], [z, sg_b], [z, sg_a], [z, sg_b]])

    #randomly pick intervention sentences based on control
    for m in control:
        l = []
        if ind % 4 == 0:
            for n in prem:
                if m[0][0] == n[0] and m[0][2] == n[2] and n[-1] not in pred and n[-1] not in diff:
                    l.append(n)
            if len(l) >= ratio:
                l = random.sample(l, ratio)
            for x in l:
                #pair up hyp1 and hyp2
                hyp_a, hyp_b = split_premise(x)
                intervention.extend([[x, hyp_a], [x, hyp_b]])
        ind += 1

    return control, intervention

def to_tsv(filepath: str, strings: List[str], encoding: str='utf-8') -> None:
    """
    read in sentences and output in the format of 'premise \t hypothesis'
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        #output = csv.writer(f, delimiter='\t')
        for i in range(len(strings)):
            st = ' '.join(strings[i][0]) + '.\t' + ' '.join(strings[i][1]) + '.'
            f.write(st + '\n')
            #output.writerow(strings[i])


def main(args: argparse.Namespace) -> None:
    print("loading components...")
    g = load(args.cfg_path)
    print("generating sentence...")
    sentences = list(generate(g))

    print("spliting data...")
    control, intervention = split_data(sentences, args.dist_path, args.ratio)
    print("writing in output files...")
    to_tsv(args.control_output_path, control)
    to_tsv(args.intervention_output_path, intervention)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', help="path to cfg grammar file")
    parser.add_argument('--dist_path', help="path to distributive pred file")
    parser.add_argument('--ratio', help="generation ratio of control to intervention")
    parser.add_argument('--control_output_path', help="path to output control file")
    parser.add_argument('--intervention_output_path', help="path to output intervention file")
    args = parser.parse_args()

    main(args)
