import os
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from cma import CMA
from utils import read_distnli_data


def run_cma(args):
    model_name = args.model_name
    control_path = args.control_path
    intervention_path = args.intervention_path
    default_label = args.default_label
    num_neuron_batch = args.num_neuron_batch
    topk_neurons = args.topk_neurons
    result_dir = args.result_dir
    result_file = os.path.join(result_dir, f'{model_name}.npz')
    TEs, NIEs, topk_NIEs = None, None, None
    
    # Read data
    control_data = read_distnli_data(control_path, model_name, return_label=False)
    intervention_data = read_distnli_data(intervention_path, model_name, return_label=False)
    assert len(control_data) == len(intervention_data), 'Unmatched control and intervention data. '

    # Initialize CMA
    with torch.no_grad():
        torch.cuda.empty_cache()
    cma = CMA(model_name, default_label, num_neuron_batch)
    encoded_data = cma.encode_data(control_data, intervention_data)
    
    # Read results if exists
    if os.path.exists(result_file):
        npzfile = np.load(result_file)
        TEs = npzfile["TEs"]
        if "NIEs" in npzfile:
            NIEs = npzfile["NIEs"]
        topk_NIEs = npzfile["topk_NIEs"]
    
    # Calculate TE
    if TEs is None:
        TEs = cma.calculate_TE(encoded_data)
    
    # Perform t-test
    TE_mean = np.mean(TEs)
    TE_var = np.std(TEs)
    ttest = stats.ttest_1samp(TEs, 0.0, alternative="greater")
    t_stat = ttest.statistic
    p_value = ttest.pvalue
    print("Total Effect | Mean: {} SD: {} T: {} P-value {}".format(TE_mean, TE_var, t_stat, p_value))
    
    # Calculate NIE
    if p_value < 0.05:
        if not os.path.exists(result_file):
            NIEs = cma.calculate_NIE(encoded_data)
            topk_NIEs = cma.calculate_topk_NIE(encoded_data, NIEs, topk_neurons)
        avg_NIE = np.mean(NIEs, axis=0)
        avg_topk_NIE = np.mean(topk_NIEs, axis=0)
    
    # Plot results
    sns.set(rc = {'figure.figsize':(6, 4)})
    sns.set_style("white")
    ax0 = sns.histplot(TEs, color="#68a8ad", kde=True)
    ax0.set(xlabel="Total Effect")
    plt.show()
    
    if p_value < 0.05:
        sns.set(rc = {'figure.figsize':(12, 4)})
        sns.set_style("whitegrid")
        df = pd.DataFrame(avg_NIE).stack().rename_axis(['Layer', 'Neuron']).reset_index(name='ANIE')
        ax1 = sns.regplot(x = "Neuron", y = "ANIE", data = df, fit_reg = False, color="#68a8ad", scatter_kws={'s':10})
        ax1.set(xlim=(-20, avg_NIE.shape[1] + 20))
        ax1.set(ylabel="Average Natural Indirect Effect")
        plt.show()
        
    if p_value < 0.05:
        sns.set(rc = {'figure.figsize':(2, 6)})
        ax2 = sns.heatmap(avg_topk_NIE, cmap=sns.color_palette("mako", as_cmap=True), center=0, vmin=0, vmax=1, cbar_kws = dict(aspect=30, pad=0.1))
        ax2.invert_yaxis()
        plt.show()
    
    # Save results
    if not os.path.exists(result_file):
        np.savez(result_file, TEs=TEs, NIEs=NIEs, topk_NIEs=topk_NIEs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, 
                        help='Model name on huggingface model hub. ')
    parser.add_argument('--control_path', default=None, type=str, required=True, 
                        help='The path to the control tsv file. ')
    parser.add_argument('--intervention_path', default=None, type=str, required=True, 
                        help='The path to the intervention tsv file. ')
    parser.add_argument('--default_label', default=None, type=str, required=True, 
                        help='Default NLI label in CMA. ')
    parser.add_argument('--num_neuron_batch', default=None, type=int, required=True)
    parser.add_argument('--topk_neurons', default=None, type=float, required=True, 
                        help='Calculate the NIE of top k neurons, ranging from 0 to 1. ')
    parser.add_argument('--result_dir', default=None, type=str, required=False, 
                        help='The directory to save the results. ')
    args = parser.parse_args()
    run_cma(args)
