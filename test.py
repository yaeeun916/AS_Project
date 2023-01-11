import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import plot_confusion_matrix, plot_roc, scatterplot_scores


def main(config):
    logger = config.get_logger('test')
    log_dir = config.log_dir

    # setup data_loader instances
    data_info_dir = Path(config.config['data']['data_info_dir']) / config.config['data']['type']
    sample_df_path = data_info_dir / config.config['data']['sample_df_fname']
    tile_df_path = data_info_dir / config.config['data']['tile_df_fname']
    sample_df = pd.read_csv(sample_df_path)
    tile_df = pd.read_csv(tile_df_path)

    data_loader = getattr(module_data, config['data_loader']['type'])(
        sample_df_path=sample_df_path,
        tile_df_path=tile_df_path,
        AS_low_thresh=config['data_loader']['args']['AS_low_thresh'],
        AS_high_thresh=config['data_loader']['args']['AS_high_thresh'],
        MSIstat=config['data_loader']['args']['MSIstat'],
        tilenum_thresh=config['data_loader']['args']['tilenum_thresh'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        transform=False,
        num_workers=2
    )

    # log filtered barcodes
    barcodes = data_loader.barcode_list[0]
    logger.info('    {:15s}: {}'.format("test_barcodes", barcodes))

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    tile_paths = []
    tile_preds = []
    tile_probs = []
    tile_labels = []

    with torch.no_grad():
        for i, (data, target, path) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.unsqueeze(1).float()

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            prob = torch.sigmoid(output)
            pred = torch.round(prob)
            tile_paths.extend(path)
            tile_probs.append(prob.detach().cpu().numpy())
            tile_preds.append(pred.detach().cpu().numpy())
            tile_labels.append(target.detach().cpu().numpy())

    tile_probs = np.concatenate(tile_probs)
    tile_preds = np.concatenate(tile_preds)
    tile_labels = np.concatenate(tile_labels)

    # 1. tile-level evaluation
    logger.info('========= tile-level evaluation =========')
    n_samples = len(data_loader.sampler)
    tile_log = {'loss': total_loss / n_samples}
    tile_log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    tile_log.update({'auroc': roc_auc_score(tile_labels, tile_probs)})
    for metric, value in tile_log.items():
        logger.info('{}: {:.4f}'.format(metric, value))
 
    cm = confusion_matrix(tile_labels, tile_preds, labels=[0, 1]) # labels are set for binary classification
    cm_fig = plot_confusion_matrix(cm, [0, 1])
    cm_fig.savefig(log_dir / 'confusion_matrix(tile).png')
    
    # save results
    tile_result = {
        'Tile_Path' : tile_paths,
        'Tile_Pred' : tile_preds.squeeze(),
        'Tile_Prob' : tile_probs.squeeze(),
        'Tile_Label' : tile_labels.squeeze()
    }
    tile_result = pd.DataFrame(tile_result)
    tile_result = pd.merge(tile_result, tile_df, left_on='Tile_Path', right_on='Path', how='left')
    tile_result.to_csv(log_dir / 'tile_data.csv', index=False)


    # 2. sample-level evaluation
    logger.info('========= sample-level evaluation =========')
    votes = []
    sample_labels = []
    img_per_sample = []

    for barcode in barcodes:
        mask = [barcode in path for path in tile_paths]

        pred_0 = np.sum(tile_preds[mask] == 0)
        pred_1 = np.sum(tile_preds[mask] == 1)
        label = tile_labels[mask][0]

        sample_labels.append(label)
        votes.append(pred_1 / (pred_0 + pred_1))
        img_per_sample.append(np.sum(mask))

    # auroc
    auroc = roc_auc_score(sample_labels, votes)
    sample_log = {'auroc': auroc}

    # optimal threshold (obtain by optimizing Youden's J statistics)
    fpr, tpr, thresholds = roc_curve(sample_labels, votes)
    youdenJ = tpr - fpr
    index = np.argmax(youdenJ)
    thresh_opt = round(thresholds[index], ndigits=4)
    sample_log.update({'thresh_opt': thresh_opt})

    # roc curve
    roc_fig = plot_roc(fpr, tpr, auroc, index, thresh_opt)
    roc_fig.savefig(log_dir / 'roc_curve(sample).png')

    # scatter plot of votes
    scatter_fig = scatterplot_scores('votes', votes, sample_labels, thresh_opt)
    scatter_fig.savefig(log_dir / 'votes.png')

    # accuracy, precision, recall, confusion matrix at optimal threshold
    sample_preds = [1 if vote >= thresh_opt else 0 for vote in votes]
    cm = confusion_matrix(sample_labels, sample_preds, labels=[0, 1]) # labels are set for binary classification
    cm_fig = plot_confusion_matrix(cm, [0, 1])
    cm_fig.savefig(log_dir / 'confusion_matrix(sample).png')
    accuracy = accuracy_score(sample_labels, sample_preds)
    precision = precision_score(sample_labels, sample_preds)
    recall = recall_score(sample_labels, sample_preds)
    f1 = f1_score(sample_labels, sample_preds)
    sample_log.update({
        'accuracy(thresh_opt)': accuracy,
        'precision(thresh_opt)': precision,
        'recall(thresh_opt)': recall,
        'f1(thresh_opt)': f1
    })
    for metric, value in sample_log.items():
        logger.info('{}: {:.4f}'.format(metric, value))

    # save results
    sample_result = {
        'Barcode' : barcodes,
        'Tile_Num' : img_per_sample,
        'Votes' : votes,
        'Pred' : sample_preds,
        'True_Label' : sample_labels
    }
    sample_result = pd.DataFrame(sample_result)
    sample_result = pd.merge(sample_result, sample_df[['Barcode', 'AS_Label', 'Type', 'MSI_status']], left_on='Barcode', right_on='Barcode', how='left')
    sample_result.to_csv(log_dir / 'sample_data.csv', index=False)

    # log each sample
    for index, row in sample_result.iterrows():
        logger.info('barcode : {};\t # tiles : {:5d};\t votes : {:.4f};\t label : {:2d};\t AS : {:2d};'.format(row['Barcode'], row['Tile_Num'], row['Votes'], row['True_Label'], row['AS_Label']))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
