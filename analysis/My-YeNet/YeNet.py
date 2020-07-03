import torch
import torch.nn as nn
import torch.nn.functional as F


class YeNet(nn.Module):
    def __init__(self, threshold=3):
        super(YeNet, self).__init__()

        self.TLU = nn.Hardtanh(-threshold, threshold, True)

        self.conv1 = nn.Conv2d(1, 30, 5, 1)
        self.conv2 = nn.Conv2d(30, 30, 3, 1)
        self.conv3 = nn.Conv2d(30, 30, 3, 1)

        self.conv4 = nn.Conv2d(30, 30, 3, 1)
        self.pool4 = nn.AvgPool2d(2, 2)

        self.conv5 = nn.Conv2d(30, 32, 5, 1)
        self.pool5 = nn.AvgPool2d(3, 2)

        self.conv6 = nn.Conv2d(32, 32, 5, 1)
        self.pool6 = nn.AvgPool2d(3, 2)

        self.conv7 = nn.Conv2d(32, 32, 5, 1)
        self.pool7 = nn.AvgPool2d(3, 2)

        self.conv8 = nn.Conv2d(32, 16, 3, 1)

        self.conv9 = nn.Conv2d(16, 16, 3, 3)

        self.l10 = nn.Linear(16 * 3 * 3, 2)

        # TODO: define reset_parameters

    def forward(self, inp):
        inp = inp.float()
        # inp = torch.from_numpy(inp)
        # print(type(inp))
        print("New input arrived", inp.shape, self.conv1)
        # TODO: should apply preprocessing
        # TODO: should apply TLU
        # TODO: should apply normalization

        print(inp.shape)
        inp = self.conv1(inp)
        print(inp.shape)
        inp = self.conv2(inp)
        inp = self.conv3(inp)
        inp = self.conv4(inp)
        inp = self.pool4(inp)
        inp = self.conv5(inp)
        inp = self.pool5(inp)
        inp = self.conv6(inp)
        inp = self.pool6(inp)
        inp = self.conv7(inp)
        inp = self.pool7(inp)
        inp = self.conv8(inp)
        inp = self.conv9(inp)

        inp = inp.view(inp.size(0), -1)
        inp = self.l10(inp)

        return inp
