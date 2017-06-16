'''
Description :LeNet for NicIcon data-set
Author : Anjan Dutta and Sounak Dey
Dataroot Organisation: nicicon_WI
                       |
                       -- Train (files : 4_w033-03309-7_033_Casualty ; where 4 is the label, then writer and so on)
                       |
                       -- Test  (same as above)
Test Acurracy : 93% (Writer Independent)
'''
from __future__ import print_function
import argparse
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from Nicicon import NicIcon

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NicIcon Example')
parser.add_argument('--dataroot', type=str, default='/home/sounak/Documents/Datasets/nicicon_WI/',
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=154, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
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
base_path = args.dataroot

dset_train = NicIcon(base_path, train=True, img_ext='.pbm',
                     transform=transforms.Compose([transforms.ToTensor()]))
dset_test = NicIcon(base_path, img_ext='.pbm',
                    transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        lenet_mnist = torch.load('./lenet_minst_model.pt')
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1.weight.data = lenet_mnist.items()[0][1]
        self.conv1.bias.data = lenet_mnist.items()[1][1]
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2.weight.data = lenet_mnist.items()[2][1]
        self.conv2.bias.data = lenet_mnist.items()[3][1]
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(9680, 4096)
        self.fc2 = nn.Linear(4096, 14)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 9680)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# model = torch.load('./lenet_minst_model.pt')
# print (model.items()[0:3])
# sys.exit()
model = Net()
if args.cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)         # SGD optimizer
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)      # Adagrad optimizer

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        # loss = nn.CrossEntropyLoss()(output, target)                                  # Cross entropy Loss
        loss = nn.NLLLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #test_loss += nn.CrossEntropyLoss()(output, target).data[0]                     # Cross Entropy Loss
        test_loss += nn.NLLLoss()(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
