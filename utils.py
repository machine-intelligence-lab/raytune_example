import argparse
import torchvision
import torch.optim
import torch.nn as nn
import os
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as TF

class Net(nn.Module):
    def __init__(self, dim1, dim2):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(28*28, dim1)
        self.layer2 = nn.Linear(dim1, dim2)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(dim2, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        return self.classifier(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--num_trials", type=int, default=5) #The number of trials for raytune
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--dim1", type=int, default=128)
    parser.add_argument("--dim2", type=int, default=128)
    parser.add_argument("--batchsize", type=int, default=256)
    args = parser.parse_args()

    args.dataroot = os.path.abspath(".data")

    return args

def get_model(args):
    model = Net(args.dim1, args.dim2)
    return model

def get_optimizer(args, model):
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.wd)

    return optimizer
    

def get_loader(args):
    transform = TF.Compose([
        TF.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(args.dataroot, True, TF.ToTensor(), download= True)
    validset = torchvision.datasets.MNIST(args.dataroot, False, TF.ToTensor(), download= True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batchsize, shuffle = True, num_workers = 2, pin_memory = True)
    validloader = torch.utils.data.DataLoader(validset, batch_size = args.batchsize, shuffle = False, num_workers = 2, pin_memory = True)

    return trainloader, validloader
