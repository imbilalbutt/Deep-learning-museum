from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor, Normalize
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data:pd.DataFrame, mode:str, apply_transforms=True) -> None:
        super().__init__()
        if mode not in ["val", "train"]:
            raise ValueError("mode = %s is not supported".format(mode))
        self.data = data
        self.mode = mode
        self.apply_transforms = apply_transforms

    def __len__(self):
        return len(self.data.index)

    def _transform(self, trnsfrm_list):
        return tv.transforms.Compose(trnsfrm_list)

    def __getitem__(self, index):
        filename, crack, inactive = self.data.iloc[index]
        img = imread(Path(filename), as_gray=True)
        img = gray2rgb(img)
        transform = self._transform([ToPILImage(), ToTensor(), Normalize(train_mean, train_std)])
        img = transform(img)
        return (img, torch.tensor([crack, inactive], dtype=torch.float32))

if __name__=="__main__":
    data = pd.read_csv("data.csv", sep=';')
    print(data.head(1))
    dataset = ChallengeDataset(data, "val")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
