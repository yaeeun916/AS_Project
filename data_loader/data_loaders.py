from torchvision import datasets, transforms
from torchvision.io import read_image
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# perform transform only for train, val set
# shuffle only train set
class CrcTileDataset(Dataset):
    def __init__(self, tile_paths, labels, transform):
        self.tile_paths = tile_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        img_path = self.tile_paths[idx]
        image = read_image(img_path)
        image = image / 255
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

class CrcTileDataLoader(DataLoader):
    def __init__(self, sample_df_path, tile_df_path,
                 AS_low_thresh, AS_high_thresh, MSIstat, tilenum_thresh,
                 batch_size, shuffle=True, validation_split=0.0, num_workers=1, collate_fn=default_collate,
                 training=True, transform=True):
        self.sample_df = pd.read_csv(sample_df_path)
        self.tile_df = pd.read_csv(tile_df_path)
        self.AS_low_thresh = AS_low_thresh
        self.AS_high_thresh = AS_high_thresh
        self.MSIstat = MSIstat
        self.tilenum_thresh = tilenum_thresh
        self.transform = transform

        # [[test barcodes]] for test loader
        # [[train barcodes], [val barcodes]] for train loader
        self.barcode_list = []

        # dataset
        barcodes, sample_labels = self._filter_samples(
            self.sample_df, training,
            self.AS_low_thresh, self.AS_high_thresh,
            self.MSIstat, self.tilenum_thresh)
        self.dataset = self._dataset_from_barcodes(barcodes) # self.tile_df is updated to filtered tiles

        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(self.dataset)

        # split sampler
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, barcodes, sample_labels)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': training
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    # split tiles on sample level
    def _split_sampler(self, split, barcodes, sample_labels):
        if split == 0.0:
            self.barcode_list.append(barcodes)
            return None, None

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."

        # sample-level train-val stratified split
        train_barcodes, val_barcodes = train_test_split(barcodes, test_size=split, stratify=sample_labels, random_state=42)
        self.barcode_list.append(train_barcodes)
        self.barcode_list.append(val_barcodes)
        train_idx = self.tile_df.index[self.tile_df['Barcode'].isin(train_barcodes)].tolist()
        val_idx = self.tile_df.index[self.tile_df['Barcode'].isin(val_barcodes)].tolist()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    # train loader and val loader : same init_args(including dataset), different sampler
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    # sample-level
    def _filter_samples(self, sample_df, training, AS_low_thresh, AS_high_thresh, MSIstat, tilenum_thresh):
        Set = 'Train' if training else 'Test'
        mask_set = (sample_df.Set == Set)
        # include AS 0
        # mask_AS = (sample_df.AS_Label <= AS_low_thresh) | (sample_df.AS_Label >= AS_high_thresh)
        # exclude AS 0
        mask_AS = ((sample_df.AS_Label <= AS_low_thresh) | (sample_df.AS_Label >= AS_high_thresh)) & (sample_df.AS_Label != 0)
        MSIstat = MSIstat.split("+")
        mask_MSS = (sample_df.MSI_status.isin(MSIstat))
        mask_tile = (sample_df.Tile_Num >= tilenum_thresh)

        # filter samples
        # use .copy() prevents SettingWithCopyWarning
        filtered_df = sample_df[mask_set & mask_AS & mask_MSS & mask_tile].copy()
        # add column 'Label'
        filtered_df['Label'] = filtered_df.apply(lambda row: 0 if row['AS_Label'] <= AS_low_thresh else 1, axis=1)

        barcodes = filtered_df['Barcode'].tolist()
        labels = filtered_df['Label'].tolist()

        return barcodes, labels

    # tile-level
    # update self.tile_df using filtered barcodes
    def _dataset_from_barcodes(self, barcodes):
        self.tile_df = self.tile_df.loc[self.tile_df['Barcode'].isin(barcodes)]
        self.tile_df['Label'] = self.tile_df.apply(lambda row: 0 if row['AS_Label'] <= self.AS_low_thresh else 1, axis=1)
        tile_paths = self.tile_df['Path'].tolist()
        labels = self.tile_df['Label'].tolist()
        self.tile_df.reset_index(drop=True, inplace=True) # so that attribute index == numeric row indices (needed for sampler)

        trsfm = None
        if (self.transform):
            trsfm = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]) # empirically, other augmentation don’t improve performance (J.N.Kather)

        return CrcTileDataset(tile_paths, labels, trsfm)
