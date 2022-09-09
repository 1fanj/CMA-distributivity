import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import argparse
import os
from utils import read_conjnli_data, read_distnli_data, read_hans_data, label_mapping


def compute_acc(preds, labels, dataset_type, model_name, suffix):
    assert len(preds) == len(labels), 'Predictions size not matching with labels size'
    if dataset_type == 'ConjNLI' or (dataset_type == 'distnli' and 'control' in suffix):
        return (preds == labels).mean()
    elif dataset_type == 'HANS' or (dataset_type == 'distnli' and 'treatment' in suffix):
        idx2label = label_mapping(model_name, mode='id2lb')
        f = np.vectorize(lambda x: -1 if idx2label[x] != 'entailment' else x)
        preds = f(preds)
        if dataset_type == 'HANS':
            return ((labels[np.where(labels==-1)] == preds[np.where(labels==-1)]).mean(), (labels[np.where(labels!=-1)] == preds[np.where(labels!=-1)]).mean())
        else:
            return (preds == labels).mean()

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def evaluate(data, model_name, device, dataset_type, suffix):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model_name_to_save = model_name.split("/")[-1]
    save_dir = f'./results/{dataset_type}'
    CHECK_OUTPUT_DIR = os.path.isdir(save_dir)
    if not CHECK_OUTPUT_DIR:
        os.makedirs(save_dir)
    output_path = f'./results/{dataset_type}/{model_name_to_save}_{dataset_type}{suffix}.txt'

    model.to(device)
    model.eval()
    has_result = False

    with torch.no_grad():
        for premise, hypothesis, label in tqdm(data, desc=f'Evaluating {model_name} on dataset...'):
            inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt').to(device)
            output = model(inputs)
            logits = output.logits
            if not has_result:
                all_logits = logits.detach().cpu().numpy()
                all_labels = np.array([label])
                has_result = True
            else:
                all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(all_labels, np.array([label]), axis=0)
        all_preds = np.argmax(all_logits, axis=1)
        acc = compute_acc(all_preds, all_labels, dataset_type, model_name, suffix)
    
    with open(output_path, 'w') as f:
        f.write(f'Accuracy: {acc}\n')
        for pred, label, logits in zip(all_preds, all_labels, all_logits):
            f.write(f'{pred}\t{label}\t{softmax(logits)}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, 
                        help='The model name if it can be loaded directly through huggingface, otherwise the path to the model. ')
    parser.add_argument('--data_path', default=None, type=str, required=True, 
                        help='The path to the NLI data for evaluation including the file name. ')
    parser.add_argument('--dataset_type', default='ConjNLI', type=str, required=True, 
                        help='Dataset format. Options are ConjNLI, distnli. ')
    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    dataset_type = args.dataset_type

    device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = None
    if dataset_type == 'ConjNLI':
        data = read_conjnli_data(data_path, model_name)
    elif dataset_type == 'distnli':
        data = read_distnli_data(data_path, model_name)
        suffix = f"_{data_path.split('/')[-1].split('_')[0]}"
    elif dataset_type == 'HANS':
        data = read_hans_data(data_path, model_name)
    evaluate(data, model_name, device, dataset_type, suffix)


if __name__ == '__main__':
    main()