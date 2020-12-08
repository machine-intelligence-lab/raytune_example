import torch.nn as nn
from utils import *

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import numpy as np

    

#raytune Trainable function interface
#First argument:
#  -'config' : python dictionary contating {key: hyper parameter name, value: hyper parameter value suggested by raytune}
#Second argument (optional):
#  -'checkpoing_dir' : related to a checkpoint save path
def main(config, checkpoint_dir = None, args = None):
    #Adopt hparam values suggested by raytune
    args.lr = config['lr']
    args.dim1 = config['dim1']
    args.dim2 = config['dim2']
    args.wd = config['wd']
    args.momentum = config['momentum']


    model = get_model(args).cuda()
    optimizer = get_optimizer(args, model)
    train_loader, valid_loader = get_loader(args)
    criterion = nn.CrossEntropyLoss()

    for it in range(args.max_epoch):
        trainloss = train_epoch(args, model, optimizer, criterion, train_loader)
        valloss, valacc = evaluate(args, model, criterion, valid_loader)

        tune.report(train_loss = trainloss, val_loss = valloss, val_accuracy = valacc ) #report metrics to tune

            

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
    
    #Define hyper parameter serach space
    config = {
        'lr' : tune.qloguniform(1e-4, 0.1, 1e-4), #loguniform, range:[1e-4,0.1], quantization level:1e-4
        'momentum' : tune.quniform(0.7, 0.95, 0.05), #quniform, range:[0.6, 0.99], quantization level:0.05
        'wd' : tune.choice([1e-1, 1e-2, 1e-3, 1e-4]),
        'dim1' : tune.choice([64, 128, 256, 512, 1024]),
        'dim2' : tune.sample_from(lambda spec: spec.config.dim1 // (2 ** np.random.randint(0, 3)))
    }

    #(Optional) Apply ASHAScheduler, recommended starting point
    asha_scheduler = ASHAScheduler(
        metric = 'val_accuracy', #A metric which we want to optimize with respect to
        mode = 'max', #'max' or 'min'
        grace_period = 5, #Perform trials at least this amount of time units
        max_t = args.max_epoch #Maximum time units
            )

    #(Optional), you can specify metrics that will be shown in the raytune progress table
    reporter = CLIReporter( metric_columns = ["train_loss", "val_loss", "val_accuracy", "training_iteration"] )

    #Tune launch
    experiment = tune.run(
        partial(main, args=args), #Trainable function
        #name = 'raytune_example', #(Optional) Experiment Name
        config = config, #Hyper parameter search space
        local_dir = '.ray_result', #raytune result save directory
        resources_per_trial = {'cpu': 2, 'gpu': 0.25}, #CPU&GPU resources that each trial can use
        num_samples = args.num_trials, #How many trials will be perforemd
        scheduler = asha_scheduler, #raytune scheduler
        fail_fast = True, #If at least one trial reports error, immediately quit every trial
        log_to_file = True,
        progress_reporter = reporter)

    best_trial = experiment.get_best_trial('val_accuracy', 'max', 'last')
    print("best trial config:", best_trial.config)
    print("best trial final validation accuracy:", best_trial.last_result['val_accuracy'])
    

    
    #main(args)
