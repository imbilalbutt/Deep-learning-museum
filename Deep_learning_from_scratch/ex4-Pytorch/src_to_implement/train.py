import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
dataset = pd.read_csv("data.csv", sep=";")
train_data, val_data = train_test_split(dataset, train_size=0.9)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
batch_size = 20
train_data = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=batch_size)
val_data = t.utils.data.DataLoader(ChallengeDataset(val_data, 'val'), batch_size=batch_size)


# create an instance of our ResNet model
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
crit = t.nn.BCELoss()
optim = t.optim.SGD(params=resnet.parameters() , lr=0.0001, momentum=0.9)
trainer = Trainer(model=resnet,
                crit=crit,
                optim=optim,
                train_dl=train_data,
                val_test_dl=val_data,
                cuda=False,
                early_stopping_patience=-1)

# go, go, go... call fit on trainer
res = trainer.fit(100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')