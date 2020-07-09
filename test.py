import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import os
import argparse

from models import *
from utils.datasets import * 
from utils.utils import * 

def test(model=None, testloader=None, batch_size=64):
    # Data
    if testloader is None:
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        testloader = cifar10_test_dataset(batch_size=batch_size, nw=nw)

    # Model
    if model is None:
        device = select_device(opt.device, batch_size=batch_size)
        checkpoint = opt.checkpoint
        model_name = opt.model_name

        assert os.path.isfile(checkpoint), 'Error: no %s file found!'%checkpoint

        cfg = None
        if not(opt.cfg == ''):
            cfg = torch.load(opt.cfg)['cfg']

        if checkpoint.endswith('.pth'): # only load weights from file
            model = globals()[model_name](cfg)
            model.load_state_dict(torch.load(checkpoint)['model'])
        else:
            assert False, "checkpoint file must end with '.pth', so '%s' does not meet the requirements!!!"%checkpoint

        model = model.to(device)
    else:
        device = next(model.parameters()).device  # get model device

    # Loss
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        print(('\n' + '%10s' * 4) % ('test_loss', 'correct', 'total', 'Acc'))
        pbar = tqdm(enumerate(testloader), total=len(testloader))  # progress bar
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            s = ('%10.3g' * 4) % (test_loss/(batch_idx+1), correct, total, 100.*correct/total)
            pbar.set_description(s)

    return test_loss, correct, total



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--model-name', type=str, default='ResNet18', help='the function to generate model')
    parser.add_argument('--cfg', type=str, default='', help='the config file to create model')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint/last.pth', help='checkpoint file path')

    parser.add_argument('--batch-size', type=int, default=128, help='size of each image batch')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')

    opt = parser.parse_args()
    print(opt)

    test(batch_size=opt.batch_size)