import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from itertools import repeat
from collections import OrderedDict
from glob import glob
import os
import shutil


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

# %%
# create dataframe of sample & tile level summarise file
# move images to folder corresponding to their labels
# --src_dir : ./sortedCOAD
# --dst_dir : ./data/CRC
def summarizeDF(src_dir, dst_dir, label_path, sample_df_path, tile_df_path):
    sample_dic = {} # Barcode : [AS_Label, Tile_Num, Set, MSI_status]
    tile_dic = {} # path : [Barcode, AS_Label]

    labels_df = pd.read_excel(label_path, header=1, index_col='Sample')
    tumor_type = ['COAD', 'READ']
    mask_tumor = labels_df.Type.isin(tumor_type)
    labels_df = labels_df[mask_tumor]

    for sub_dir in glob(os.path.join(src_dir, '*/')):
        if 'TRAIN' in sub_dir:
            set_tag = 'Train'
        else:
            set_tag = 'Test'
        if 'MSS' in sub_dir:
            msi_status = 'MSS'
        else:
            msi_status = 'MSIMUT'

        tiles = glob(os.path.join(sub_dir, '*.png'))
        for tile in tiles:
            fname = os.path.basename(tile)
            barcode = fname[17 : 32]
            if barcode in labels_df.index:
                AS = labels_df.loc[barcode, 'AneuploidyScore(AS)']
                _dst_dir = os.path.join(dst_dir, set_tag, str(AS))
                dst_path = os.path.join(_dst_dir, fname)
                if barcode in sample_dic:
                    sample_dic[barcode][1] += 1
                else:
                    sample_dic[barcode] = [AS, 1, set_tag, msi_status]
                    if not os.path.exists(_dst_dir):
                        os.makedirs(_dst_dir)
                shutil.move(tile, _dst_dir)
                tile_dic[dst_path] = [barcode, AS]

    shutil.rmtree(src_dir)

    # create csv file
    sample_df = pd.DataFrame(
        [{'Barcode' : key,
          'AS_Label' : value[0],
          'Tile_Num' : value[1],
          'Set' : value[2],
          'MSI_status' : value[3]}
         for key, value in sample_dic.items()]
    )
    sample_df.to_csv(sample_df_path, index = False)

    tile_df = pd.DataFrame(
        [{'Path' : key,
          'Barcode' : value[0],
          'AS_Label' : value[1]}
         for key, value in tile_dic.items()]
    )
    tile_df.to_csv(tile_df_path, index = False)

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize='large')
    plt.yticks(tick_marks, class_names, fontsize='large')

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize='large')

    plt.ylabel('True label', fontsize='x-large')
    plt.xlabel('Predicted label', fontsize='x-large')

    figure.tight_layout()

    return figure

def plot_roc(fpr, tpr, roc_auc, opt_idx, thresh_opt):
    figure = plt.figure()
    lw = 2
    # plot roc
    plt.plot(
        fpr,
        tpr,
        color="black",
        lw=lw,
        label="ROC curve (area = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")

    # plot optimal threshold
    plt.plot(fpr[opt_idx], tpr[opt_idx],
             marker="o", markersize=10, markerfacecolor="red")
    plt.annotate("optimal threshold : %0.4f" % thresh_opt,
                 (fpr[opt_idx], tpr[opt_idx]),
                 (fpr[opt_idx], tpr[opt_idx] - 0.15))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")

    figure.tight_layout()

    return figure

def scatterplot_scores(score_name, scores, labels, thresh_opt=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    scatter = ax.scatter(scores, np.zeros_like(scores), c=labels)
    if thresh_opt:
        ax.plot(thresh_opt, ax.get_ylim()[0]-0.006, 'v', c='red', zorder=10, clip_on=False)
        ax.annotate(f'optimal threshold :\n{thresh_opt:.4f}', (thresh_opt, ax.get_ylim()[0]+0.014),
                c='red', ha='center')

    ax.set_title(score_name)
    legend = ax.legend(*scatter.legend_elements(), loc='lower right', title='Classes')
    ax.add_artist(legend)

    ax.set_xticks(np.arange(-0.1, 1.1, 0.1))
    ax.set_ylim(-0.06, 0.06)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    xticks[-1].set_visible(False)

    fig.tight_layout()
    return fig

def histplot_results(tile_results, sample_results, log_dir):
    tileprob_per_AS(tile_results, log_dir)
    tileprob_per_sample(tile_results, log_dir)
    tileprob_per_pred(tile_results, sample_results, log_dir)

# plot and save Tile_Prob histogram
def tileprob_per_AS(tile_results, log_dir):
    # hue by AS
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(15, 8)
    sns.histplot(data=tile_results, x='Tile_Prob', hue='AS_Label', kde=True, ax=ax1)
    ax1.set_title('Distribution of Tile-level Probability', fontsize='xx-large')
    ax1.set_xlabel('Tile-level Probability', fontsize='xx-large')
    ax1.set_ylabel('Count', fontsize='xx-large')
    ax1.tick_params(axis='both', which='major', labelsize='xx-large')
    ax1.set(xlim=(0.0, 1.0))
    fig1.tight_layout()
    fig1.savefig(log_dir / 'Tile_Prob Histogram.png')

    # 1 subplot per AS
    AS_list = tile_results.AS_Label.unique()
    AS_list.sort()
    rows = len(AS_list) // 3 if (len(AS_list) % 3 == 0) else len(AS_list) // 3 + 1
    cols = 3
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    for i, AS in enumerate(AS_list):
        data = tile_results.loc[tile_results['AS_Label'] == AS]
        ax = axes2[i // 3][i % 3]
        sns.histplot(data=data, x='Tile_Prob', kde=True, ax=ax)
        ax.set_title(f'AS : {AS}', fontsize='x-large')
        ax.set_xlabel('Tile-level Probability', fontsize='large')
        ax.set_ylabel('Count', fontsize='large')
        ax.tick_params(axis='both', which='major', labelsize='medium')
        ax.set(xlim=(0.0, 1.0))
    if (len(AS_list) % 3 != 0):
        for col in range(len(AS_list) % 3, 3):
            fig2.delaxes(axes2[rows - 1][col])
    fig2.tight_layout()
    fig2.savefig(log_dir / 'Tile_Prob Histogram per AS.png')

# plot and save Tile_Prob histogram per sample
def tileprob_per_sample(tile_results, log_dir):
    # 1 subplot per AS, hue by sample
    AS_list = tile_results.AS_Label.unique()
    AS_list.sort()
    rows = len(AS_list) // 3 if (len(AS_list) % 3 == 0) else len(AS_list) // 3 + 1
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    for i, AS in enumerate(AS_list):
        data = tile_results.loc[tile_results['AS_Label'] == AS]
        ax = axes[i // 3][i % 3]
        sns.histplot(data=data, x='Tile_Prob', kde=True, hue='Barcode', ax=ax)
        ax.set_title(f'AS : {AS}', fontsize='x-large')
        ax.set_xlabel('Tile-level Probability', fontsize='large')
        ax.set_ylabel('Count', fontsize='large')
        ax.tick_params(axis='both', which='major', labelsize='medium')
        ax.set(xlim=(0.0, 1.0))
    if (len(AS_list) % 3 != 0):
        for col in range(len(AS_list) % 3, 3):
            fig.delaxes(axes[rows - 1][col])
    fig.tight_layout()
    fig.savefig(log_dir / 'Tile_Prob Histogram per Sample.png')

# plot and save Tile_Prob histogram of correctly / incorrectly predicted samples
def tileprob_per_pred(tile_results, sample_results, log_dir):
    # 2 subplots per AS - correctly / incorrectly predicted samples
    # hue by sample
    merged_df = pd.merge(tile_results, sample_results[['Barcode', 'Pred']],  how='left', left_on=['Barcode'], right_on = ['Barcode'])
    AS_list = merged_df.AS_Label.unique()
    AS_list.sort()
    fig, axes = plt.subplots(len(AS_list), 2, figsize=(15, len(AS_list) * 4))
    for i, AS in enumerate(AS_list):
        corrects = merged_df.loc[(merged_df['AS_Label'] == AS) & (merged_df['Pred'] == merged_df['Tile_Label'])]
        ax1 = axes[i][0]
        sns.histplot(data=corrects, x='Tile_Prob', kde=True, hue='Barcode', ax=ax1)
        ax1.set_title(f'Correct Samples (AS : {AS})', fontsize='x-large')
        ax1.set_xlabel('Tile-level Probability', fontsize='large')
        ax1.set_ylabel('Count', fontsize='large')
        ax1.tick_params(axis='both', which='major', labelsize='medium')
        ax1.set(xlim=(0.0, 1.0))

        incorrects = merged_df.loc[(merged_df['AS_Label'] == AS) & (merged_df['Pred'] != merged_df['Tile_Label'])]
        ax2 = axes[i][1]
        sns.histplot(data=incorrects, x='Tile_Prob', kde=True, hue='Barcode', ax=ax2)
        ax2.set_title(f'Incorrect Samples (AS : {AS})', fontsize='x-large')
        ax2.set_xlabel('Tile-level Probability', fontsize='large')
        ax2.set_ylabel('Count', fontsize='large')
        ax2.tick_params(axis='both', which='major', labelsize='medium')   
        ax2.set(xlim=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(log_dir / 'Tile_Prob Histogram per Prediction.png')
