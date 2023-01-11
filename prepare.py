import argparse
import collections
import torch
import numpy as np
from pathlib import Path
import zipfile
import os
from parse_config import ConfigParser
from utils import summarizeDF

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    data_dir = Path(config.config['data']['data_dir']) / config.config['data']['type']
    data_info_dir = Path(config.config['data']['data_info_dir']) / config.config['data']['type']
    os.makedirs(data_info_dir, exist_ok=True)
    sample_df_path = data_info_dir / config.config['data']['sample_df_fname']
    tile_df_path = data_info_dir / config.config['data']['tile_df_fname']

    # unzip tile data under src_dir
    data_zip_path = 'sortedCOAD.zip'
    src_dir = 'sortedCOAD/'
    os.makedirs(src_dir, exist_ok=True)
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall('./')
    os.remove(data_zip_path)

    # AS label data
    label_path = data_info_dir / 'mmc2.xlsx'

    summarizeDF(src_dir, data_dir, label_path, sample_df_path, tile_df_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--dtype'], type=str, target='data;type')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)