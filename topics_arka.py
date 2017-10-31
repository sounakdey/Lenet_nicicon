from __future__ import print_function
import argparse
import cv2
import sys
import os
#import numpy as np
#from torchsample.transforms import RandomAffine, ToTensor, TypeCast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from ArkaDataset import ArkaDataset
import utils


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Topic Example')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--resume', default='./checkpoint/',    # ./checkpoint/
                    help='path to latest checkpoint')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

dset_train = ArkaDataset(set_type='Train',
                     transform=transforms.Compose([transforms.ToTensor()]))
dset_test = ArkaDataset(set_type='Test',
                    transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 300)
        self.classifier = nn.Linear(600, 40)

    def forward(self, img_features, text_features):
        img_feats = self.fc1(img_features.float())
        tot_feats = torch.cat((img_feats, text_features.float()), 1)
        x = F.relu(self.classifier(tot_feats))
        return F.log_softmax(x)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)      # Adagrad optimizer

def train(epoch):
    model.train()
    for batch_idx, (img_features, text_features, labels) in enumerate(train_loader):

        if args.cuda:
            img_feats, text_feats, labels = img_features.cuda(), text_features.cuda(), labels.cuda()
       # else: 
       #     img_feats, text_feats, labels = img_features, text_features, labels  #arkaedit
        # data, target = Variable(data), Variable(target)
        img_feats, text_feats, labels = Variable(img_feats), Variable(text_feats), Variable(labels)
        optimizer.zero_grad()
        output = model(img_feats, text_feats)
        loss = nn.CrossEntropyLoss()(output, labels)                                  # Cross entropy Loss
        # loss = nn.NLLLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img_feats), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (img_features, text_features, labels) in enumerate(test_loader):
        if args.cuda:
            img_feats, text_feats, labels = img_features.cuda(), text_features.cuda(), labels.cuda()
        img_feats, text_feats, labels = Variable(img_feats, volatile=True), Variable(text_feats, volatile=True), Variable(labels)
        output = model(img_feats, text_feats)
        loss = nn.CrossEntropyLoss()(output, labels)  # Cross entropy Loss
        # output = model(data)
        #test_loss += nn.CrossEntropyLoss()(output, target).data[0]                     # Cross Entropy Loss
        test_loss += nn.CrossEntropyLoss()(output, labels).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(labels.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                           'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume)


for epoch in range(1, args.epochs + 1):
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {}; accuracy {})".format(best_model_file, checkpoint['epoch'],
                                                                             best_acc1))

    train(epoch)
    test(epoch)

