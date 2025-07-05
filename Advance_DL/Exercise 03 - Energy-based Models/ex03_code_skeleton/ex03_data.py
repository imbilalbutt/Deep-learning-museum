import numpy as np
import torch
import torch.utils.data as data

from torchvision import transforms
from pathlib import Path
from typing import Union, Dict, List, Tuple


class TransformTensorDataset(data.Dataset):
    """
    TensorDataset with support of transforms.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        try:
            y = self.tensors[1][index]
        except IndexError:
            y = -1

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def get_datasets(data_dir: Union[str, Path]) -> Dict[str, TransformTensorDataset]:
    with open(Path(data_dir).joinpath('train.npy'), 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)

    with open(Path(data_dir).joinpath('val.npy'), 'rb') as f:
        X_val = np.load(f)
        y_val = np.load(f)

    with open(Path(data_dir).joinpath('test.npy'), 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)

    with open(Path(data_dir).joinpath('ood_ta.npy'), 'rb') as f:
        ood_ta = np.load(f)

    with open(Path(data_dir).joinpath('ood_tb.npy'), 'rb') as f:
        ood_tb = np.load(f)

    # Train / Validation / Test splits
    X_train_t = torch.from_numpy(X_train).to(torch.float32).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train).to(torch.int64)
    X_val_t = torch.from_numpy(X_val).to(torch.float32).unsqueeze(1)
    y_val_t = torch.from_numpy(y_val).to(torch.int64)
    X_test_t = torch.from_numpy(X_test).to(torch.float32).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).to(torch.int64)

    # Two types of OOD datasets
    X_ood_ta = torch.from_numpy(ood_ta).to(torch.float32).unsqueeze(1)
    X_ood_tb = torch.from_numpy(ood_tb).to(torch.float32).unsqueeze(1)

    std_transform = transforms.Compose([transforms.Lambda(minmax),
                                        transforms.Resize(size=56,antialias=True),
                                        transforms.CenterCrop(size=56),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])

    # Create the train, val, test datasets.
    train_set = TransformTensorDataset(X_train_t, y_train_t, transform=std_transform)
    val_set = TransformTensorDataset(X_val_t, y_val_t, transform=std_transform)
    test_set = TransformTensorDataset(X_test_t, y_test_t, transform=std_transform)

    # Create the OOD datasets
    ood_ta_set = TransformTensorDataset(X_ood_ta, transform=std_transform)
    ood_tb_set = TransformTensorDataset(X_ood_tb, transform=std_transform)

    return {'train': train_set,
            'val': val_set,
            'test': test_set,
            'ood_ta': ood_ta_set,
            'ood_tb': ood_tb_set}
