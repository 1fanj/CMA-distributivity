
from os import walk
from transformers import AutoConfig

def label_mapping(model, mode='lb2id'):
    if 'bert-base-uncased-mnli' in model:
        if mode == 'lb2id':
            return {
                'entailment': 1, 
                'neutral': 0, 
                'contradiction': 2,
            }
        elif mode == 'id2lb':
            return {
                1: 'entailment', 
                0: 'neutral', 
                2: 'contradiction',
            }
    config = AutoConfig.from_pretrained(model)
    if mode == 'lb2id':
        for k in config.label2id:
            config.label2id[k.lower()] = config.label2id.pop(k)
        assert 'entailment' in config.label2id, config.label2id
        assert 'neutral' in config.label2id, config.label2id
        assert 'contradiction' in config.label2id, config.label2id
        mapping_dict = config.label2id
    elif mode == 'id2lb':
        for k in config.id2label:
            config.id2label[k] = config.id2label[k].lower()
        assert 'entailment' in config.id2label.values(), config.id2label
        assert 'neutral' in config.id2label.values(), config.id2label
        assert 'contradiction' in config.id2label.values(), config.id2label
        mapping_dict = config.id2label
    return mapping_dict


def read_conjnli_data(path, model_name):
    label2idx = label_mapping(model_name)
    conjnli_data = []
    with open(path, 'r') as f:
        is_first_line = True
        for line in f.readlines():
            if is_first_line:
                is_first_line = False
                continue
            premise, hypothesis, label = line.strip().split('\t')
            label = label2idx[label]
            conjnli_data.append([premise, hypothesis, label])
    return conjnli_data


def capitalize_string(str):
    return str[0].upper() + str[1:]


def read_distnli_data(path, model_name, return_label=True):
    label2idx = label_mapping(model_name)
    nli_data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            premise, hypothesis = line.strip().split('\t')
            if return_label:
                label = None
                if 'control' in path:
                    label = 'entailment'
                    label = label2idx[label]
                elif 'intervention' in path:
                    label = -1
                else:
                    raise Exception('Error encountered when generating labels. ')
                nli_data.append([capitalize_string(premise), capitalize_string(hypothesis), label])
            else:
                nli_data.append([capitalize_string(premise), capitalize_string(hypothesis)])
    return nli_data


def read_hans_data(path, model_name):
    label2idx = label_mapping(model_name)
    # data_path = 'data/HANS/heuristics_evaluation_set.txt'
    # save_path = 'data/HANS/formatted_heuristics_evaluation_set.txt'
    data = []
    with open(path, 'r') as f:
        header = f.readline().strip().split('\t')
        while True:
            line = f.readline()
            if not line.strip():
                break
            row = line.strip().split('\t')
            label = row[header.index('gold_label')]
            label = label2idx[label] if label == 'entailment' else -1
            data.append((row[header.index('sentence1')], row[header.index('sentence2')], label))
    assert len(data) != 0, 'Fail to read data from file'
    return data


def model_mapping(model_name):
    model2full = {
        'bart-large-mnli': 'facebook/bart-large-mnli',
        'deberta-v2-xlarge-mnli': 'microsoft/deberta-v2-xlarge-mnli', 
        'distilbert-base-uncased-mnli': 'typeform/distilbert-base-uncased-mnli', 
        'nli-distilroberta-base': 'cross-encoder/nli-distilroberta-base', 
        'roberta-large-mnli': 'roberta-large-mnli', 
        'xlm-roberta-large-xnli': 'joeddav/xlm-roberta-large-xnli',
        'distilbart-mnli-12-1': 'valhalla/distilbart-mnli-12-1',
        'bert-base-uncased-mnli': 'ishan/bert-base-uncased-mnli', 
        'deberta-base-mnli': 'microsoft/deberta-base-mnli', 
        'deberta-large-mnli': 'microsoft/deberta-large-mnli',
        'deberta-xlarge-mnli': 'microsoft/deberta-xlarge-mnli', 
    }
    return model2full[model_name] if model_name in model2full else None


def _get_acc_from_file(result_dir):
    acc_list = {}
    _, _, pred_files = next(walk(result_dir))
    for pred_file in pred_files:
        with open(result_dir + pred_file, 'r') as f:
            acc_score = float(f.readline().strip().split(': ')[1])
            acc_list[pred_file.split('_')[0].replace('.txt', '')] = acc_score
    return acc_list


if __name__ == '__main__':
    model_name = "roberta-large-mnli"
    data_path = "data/distnli/control_2_v3.tsv"
    data = read_distnli_data(data_path, model_name)
    print()