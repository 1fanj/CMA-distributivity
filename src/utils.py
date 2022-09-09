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
                elif 'treatment' in path:
                    label = -1
                else:
                    raise Exception('Error encountered when generating labels. ')
                nli_data.append([capitalize_string(premise), capitalize_string(hypothesis), label])
            else:
                nli_data.append([capitalize_string(premise), capitalize_string(hypothesis)])
    return nli_data


def read_hans_data(path, model_name):
    label2idx = label_mapping(model_name)
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
