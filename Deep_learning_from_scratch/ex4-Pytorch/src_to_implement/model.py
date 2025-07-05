from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Sigmoid
from torch.nn.modules.flatten import Flatten
import torch


class ResNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3, 2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(in_features=512, out_features=2),
            Sigmoid()
        )

    def forward(self, input):
        output = self.resnet(input)
        return output


class ResBlock(torch.nn.Module):
    def __init__(self, in_chan, out_chan, stride) -> None:
        super(ResBlock, self).__init__()
        self.resblock = Sequential(
            Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(num_features=out_chan),
            ReLU(),
            Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=out_chan),
            ReLU()
        )
        self.shortcut = Sequential(
            Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=stride),
            BatchNorm2d(num_features=out_chan)
        )

    def forward(self, input):
        output = self.resblock(input) + self.shortcut(input)
        return output


if __name__ == "__main__":
    model = ResNet()
    pred = model(torch.rand((40, 3, 300, 300)))
    print(pred)