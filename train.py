from ray import tune
import torch.nn as nn
from  utils import *

def main(args):
    model = get_model(args).cuda()
    optimizer = get_optimizer(args, model)
    train_loader, valid_loader = get_loader(args)
    criterion = nn.CrossEntropyLoss()

    for it in range(args.max_epoch):
        trainloss = train_epoch(args, model, optimizer, criterion, train_loader)
        valloss, valacc = evaluate(args, model, criterion, valid_loader)
        print("{}/{} TrainLoss: {:.4f}, ValidLoss: {:.4f}, ValidAccuracy: {:.4f}".format(it, args.max_epoch, trainloss, valloss, valacc))

            

def train_epoch(args, model, optimizer, criterion, train_loader):
    model.train()
    loss_sum = 0
    for img, label in train_loader:
        img, label = img.cuda(), label.cuda()
        output = model(img)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(train_loader)
    
def evaluate(args, model, criterion, valid_loader):
    model.eval()
    loss_sum = 0
    acc = 0
    samples = 0
    with torch.no_grad():
        for img, label in valid_loader:
            img, label = img.cuda(), label.cuda()
            output = model(img)
            loss = criterion(output, label)

            _, pred = torch.max( output, dim = 1 )
            acc += (pred == label).sum().item()
            loss_sum += loss.item()
            samples += label.size(0)
    
    acc /= samples
    return loss_sum / len(valid_loader), acc

if __name__=="__main__":
    args = parse_args()
    print(args)
    main(args)
