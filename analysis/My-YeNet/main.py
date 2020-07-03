import argparse
import os

import matplotlib.pyplot as plt
import numpy
import torchvision
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter as DataLoaderIter
from torch.utils.data.dataset import Dataset

from YeNet import YeNet
from utils import GenerateDatasets

cover_path = '/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/sample'
stego_path = '/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/sample'


# cover_dataset = MakeDataset(dspath=cover_path, is_cover=True)
# stego_dataset = MakeDataset(dspath=stego_path, is_cover=False)

# train_loader = (ds, batch_size=8, shuffle=True)
# train_iter = iter(train_loader)
ds = GenerateDatasets(
    boss_path='/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01',
    bows_path='/home/emadhelmi/myworks/stego/datasets/BOWS2OrigEp3'
)
ds.create_ds()
exit()
net = YeNet()


def train(epoch):
    print("Epoch number %d" % (epoch + 1))
    net.train()
    for idx, data in enumerate(train_loader):
        images, labels = data
        outputs = net(images)


for epoch in range(1):
    train(epoch)
