from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import utils
import YeNet

parser = argparse.ArgumentParser(
    description='PyTorch implementation of YeNet')

parser.add_argument('train_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training cover images')
parser.add_argument('train_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training stego images or beta maps')
parser.add_argument('--valid-cover-dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation cover images')
parser.add_argument('--valid-stego-dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation stego images or beta maps')
parser.add_argument('--make-validations', action='store_true', default=False,
                    help='make validation set from the train dir for both stego and cover images')
parser.add_argument('--validation-ratio', type=float, default=0.3,
                    help='validation to train ratio')

parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')

parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=4e-1, metavar='LR',
                    help='learning rate (default: 4e-1)')

parser.add_argument('--use-batch-norm', action='store_true', default=True,
                    help='use batch normalization after each activation,' +
                    ' also disable pair constraint (default: False)')
parser.add_argument('--embed-otf', action='store_true', default=False,
                    help='use beta maps and embed on the fly instead' +
                    ' of use stego images (default: False)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0,
                    help='index of gpu used (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')

# TODO: use a format to store logs (tensorboard ?)
# parser.add_argument('--log-path', type=str, default='logs/training.log',
#                     metavar='PATH', help='path to generated log file')
args = parser.parse_args()
arch = 'YeNet_with_bn' if args.use_batch_norm else 'YeNet'
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda()
else:
    args.gpu = None
    criterion = nn.CrossEntropyLoss().cuda()

kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {
    'num_workers': 1, 'pin_memory': False}

train_transform = transforms.Compose([
    utils.RandomRot(),
    utils.RandomFlip(),
    utils.ToTensor()
])

valid_transform = transforms.Compose([
    utils.ToTensor()
])

if args.make_validations:
    print("Make validations...")
    if os.path.isdir("dataset"):
        shutil.rmtree("dataset")
    files_list = os.listdir(args.train_cover_dir)
    random.shuffle(files_list)
    validation, train = files_list[:int(len(
        files_list) * args.validation_ratio)], files_list[int(len(files_list) * args.validation_ratio):]
    valid_cover_dir, valid_stego_dir = "dataset/validation/cover", "dataset/validation/stego"
    train_cover_dir, train_stego_dir = "dataset/train/cover", "dataset/train/stego"
    print("Copy to local train dir")
    os.makedirs(train_cover_dir)
    os.makedirs(train_stego_dir)
    for indx, file in enumerate(train):
        shutil.copyfile(
            os.path.join(args.train_cover_dir, file),
            os.path.join(train_cover_dir, file)
        )
        shutil.copyfile(
            os.path.join(args.train_stego_dir, file),
            os.path.join(train_stego_dir, file)
        )
        print("%d/%d (%.2f%%)" %
              (indx + 1, len(train), ((indx + 1) * 100 / len(train))), end="\r")
    print()

    print("Copy to local validation dir")
    os.makedirs(valid_cover_dir)
    os.makedirs(valid_stego_dir)
    for indx, file in enumerate(validation):
        shutil.copyfile(
            os.path.join(args.train_cover_dir, file),
            os.path.join(valid_cover_dir, file)
        )
        shutil.copyfile(
            os.path.join(args.train_stego_dir, file),
            os.path.join(valid_stego_dir, file)
        )
        print("%d/%d (%.2f%%)" %
              (indx + 1, len(validation), ((indx + 1) * 100 / len(validation))), end="\r")

    args.train_cover_dir, args.train_stego_dir = train_cover_dir, train_stego_dir
    args.valid_cover_dir, args.valid_stego_dir = valid_cover_dir, valid_stego_dir
    print()


print("Generate loaders...")
train_loader = utils.DataLoaderStego(args.train_cover_dir, args.train_stego_dir,
                                     embedding_otf=args.embed_otf, shuffle=True,
                                     pair_constraint=True,
                                     batch_size=args.batch_size,
                                     transform=train_transform,
                                     num_workers=kwargs['num_workers'],
                                     pin_memory=kwargs['pin_memory'])

valid_loader = utils.DataLoaderStego(args.valid_cover_dir, args.valid_stego_dir,
                                     embedding_otf=False, shuffle=False,
                                     pair_constraint=True,
                                     batch_size=args.test_batch_size,
                                     transform=valid_transform,
                                     num_workers=kwargs['num_workers'],
                                     pin_memory=kwargs['pin_memory']
                                     )

print('train_loader have {} iterations, valid_loader have {} iterations'.format(
    len(train_loader), len(valid_loader)))

net = YeNet.YeNet(with_bn=args.use_batch_norm)
if args.cuda:
    net.cuda()
optimizer = optim.Adadelta(net.parameters(), lr=args.lr, rho=0.95, eps=1e-8,
                           weight_decay=5e-4)
_time = time.time()


def get_criterion(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    result = criterion(outputs.squeeze(), labels.squeeze())
    return result


def train(epoch):
    net.train()
    running_loss = 0.
    running_accuracy = 0.
    for batch_idx, data in enumerate(train_loader):
        print("Batch index %d" % (batch_idx + 1), end="\r")
        images, labels = Variable(
            data['images']), Variable(data['labels'])
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        accuracy = YeNet.accuracy(outputs, labels).item()
        running_accuracy += accuracy
        loss = get_criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval
            print(('Train epoch: {} [{}/{}]\tAccuracy: ' +
                   '{:.2f}%\tLoss: {:.6f}').format(
                  epoch, batch_idx + 1, len(train_loader),
                  100 * running_accuracy, running_loss))
            running_loss = 0.
            running_accuracy = 0.
            net.train()


def valid():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.
    correct = 0
    for batch_idx, data in enumerate(valid_loader):
        # break
        print("Batch index %d" % (batch_idx + 1), end="\r")
        images, labels = Variable(
            data['images']), Variable(data['labels'])
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        valid_loss += get_criterion(outputs, labels).item()
        valid_accuracy += YeNet.accuracy(outputs, labels).item()
        # if (batch_idx + 1) % args.log_interval == 0:
        #     valid_accuracy /= args.log_interval
        #     valid_loss /= args.log_interval
        #     print(('Train epoch: {} [{}/{}]\tAccuracy: ' +
        #            '{:.2f}%\tLoss: {:.6f}').format(
        #           epoch, batch_idx + 1, len(valid_loader),
        #           100 * valid_accuracy, valid_loss))
        #     valid_loss = 0.
        #     valid_accuracy = 0.
        #     net.train()
    valid_loss /= len(valid_loader)
    valid_accuracy /= len(valid_loader)
    print('\nTest set: Loss: {:.4f}, Accuracy: {:.2f}%)\n'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy


def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')
        print("This is the best result so far!")


best_accuracy = 0.
for epoch in range(1, args.epochs + 1):
    print("Epoch:", epoch)
    train(epoch)
    print("Time:", time.time() - _time)
    _, accuracy = valid()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        is_best = True
    else:
        is_best = False
    print("Time:", time.time() - _time)
    save_checkpoint({
        'epoch': epoch,
        'arch': arch,
        'state_dict': net.state_dict(),
        'best_prec1': accuracy,
        'optimizer': optimizer.state_dict(),
    }, is_best)
