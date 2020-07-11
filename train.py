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
from models.resnet import *
from utils.datasets import * 
from utils.utils import * 
from utils.prune_utils import *
import test

def train():
    start_epoch = 0
    epochs = opt.epochs
    batch_size = opt.batch_size

    ckpt_dir = opt.checkpoint_dir
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")  

    device = select_device(opt.device, batch_size=opt.batch_size)

    # Data
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = cifar10_train_dataset(batch_size=batch_size,nw=nw)
    testloader = cifar10_test_dataset(batch_size=batch_size, nw=nw)
    classs = cifar10_class()

    # Model    
    model_name = opt.model_name
    checkpoint = opt.checkpoint

    cfg = None
    if not(opt.cfg == ''):
        cfg = torch.load(opt.cfg)['cfg']

    if checkpoint == '':
        model = globals()[model_name](cfg)
        init_params(model)
    else:
        if checkpoint.endswith('.pth'): # only load weights from file
            model = globals()[model_name](cfg)
            model.load_state_dict(torch.load(checkpoint)['model'])
        else:
            assert False, "checkpoint file must end with '.pth', so '%s' does not meet the requirements!!!"%checkpoint

        if opt.from_scratch:
            init_params(model)


    model = model.to(device)

    # from torchsummary import summary
    # summary(model, input_size=(3, 32, 32))

    best_acc = -1.0
    
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(ckpt_dir), 'Error: no %s directory found!'%ckpt_dir
        checkpoint = torch.load(ckpt_last)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    
    #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
    if 'VGG' in model_name:
        bn_list = []
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                bn_list.append(m)
        if opt.sr:
            print('sparse training!!!')  
    else:
        if opt.prune == 0:
            module_list, _ = model.get_module_list()
            resnet_cfg = ResNetCFG()
            prune_idx = resnet_cfg.get_prune_idx(model)

            bn_list = []
            for idx,m in enumerate(module_list):
                if idx in prune_idx:
                    bn_list.append(m['bn'])
            if opt.sr:
                print('normal sparse training!!!')
        elif opt.prune == 1:
            bn_list = []
            for m in model.modules():
                if isinstance(m,nn.BatchNorm2d):
                    bn_list.append(m)
            if opt.sr:
                print('shortcut sparse training!!!')                
    
    if tb_writer:
        for idx in range(len(bn_list)):
            bn_weights = bn_list[idx].weight.data.abs().clone().cpu()
            tb_writer.add_histogram('before_train_perlayer_bn_weights/hist', bn_weights.numpy(), idx, bins='doane')
    #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
    
    def adjust_learning_rate(optimizer, gamma, epoch):
        # 学习率衰减
        step_index = 0
        if epoch > opt.epochs * 0.5:
            # 在进行总epochs的50%时，进行以gamma的学习率衰减
            step_index = 1
        if epoch > opt.epochs * 0.7:
            # 在进行总epochs的70%时，进行以gamma^2的学习率衰减
            step_index = 2
        if epoch > opt.epochs * 0.9:
            # 在进行总epochs的90%时，进行以gamma^3的学习率衰减
            step_index = 3

        lr = opt.lr * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    for epoch in range(start_epoch, epochs):
        #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
        sr_flag = get_sr_flag(epoch, opt.sr)
        #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
        
        lr = adjust_learning_rate(optimizer, 0.1, epoch)
        print("lr:",lr)

        # train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        print(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'Loss', 'correct', 'total', 'Acc'))
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))  # progress bar
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            BNOptimizer.updateBN(sr_flag, bn_list, opt.s, epoch)
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 4) % ('%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, train_loss/(batch_idx+1), correct, total, 100.*correct/total)
            pbar.set_description(s)

        # test
        test_loss, test_correct, test_total = test.test(model=model, testloader=testloader)
        acc = 100.*test_correct/test_total

        # Save checkpoint
        final_epoch = epoch == epochs-1
        save = (not opt.nosave) or final_epoch
        if save:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            if not os.path.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)

            # Save last
            torch.save(state, ckpt_last)

            # Save best
            if acc > best_acc and not final_epoch:
                torch.save(state, ckpt_best)
                best_acc = acc

            del state

        # log
        if tb_writer:
            results = [train_loss, 100.*correct/total, lr, test_loss, acc]
            tags = ['train/Loss', 'train/Acc', 'train/lr', 'val/Loss', 'val/Acc']
            for x, tag in zip(results, tags):
                tb_writer.add_scalar(tag, x, epoch)
                
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            bn_weights = gather_bn_weights(bn_list).cpu()
            tb_writer.add_histogram('bn_weights/hist', bn_weights.numpy(), epoch, bins='doane')
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
        
        # end epoch
    # end training
    
    #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
    if tb_writer:
        for idx in range(len(bn_list)):
            bn_weights = bn_list[idx].weight.data.abs().clone().cpu()
            tb_writer.add_histogram('after_train_perlayer_bn_weights/hist', bn_weights.numpy(), idx, bins='doane')
    #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, default='ResNet18', help='the function to generate model')
    parser.add_argument('--cfg', type=str, default='', help='the config file to create model')
    parser.add_argument('--checkpoint', '-ckpt', default='', help='checkpoint file path')
    parser.add_argument('--checkpoint-dir', '-ckpt-dir', default='checkpoint', help='checkpoint directory')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', '-bs',type=int, default=128)
    parser.add_argument('--logdir', '-log', default='runs', help='tensorboard logdir')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)') # not support multi-GPU
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning_rate')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    
    parser.add_argument('--prune', type=int, default=0, help='0:nomal prune 1:other prune ')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.001, help='scale sparse rate')
    parser.add_argument('--from-scratch', action='store_true', help='training form scratch, not using pretrained weights')

    opt = parser.parse_args()
    print(opt)

    
    try: 
        from torch.utils.tensorboard import SummaryWriter
        print('Start Tensorboard with "tensorboard --logdir=%s, view at http://localhost:6006/'%opt.logdir)
        tb_writer = SummaryWriter(opt.logdir)
    except:
        tb_writer = None

    train()
