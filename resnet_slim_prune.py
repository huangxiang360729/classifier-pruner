import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import os
import argparse
import numpy as np
from copy import deepcopy

from models import *
from models.resnet import *
from utils.datasets import * 
from utils.utils import * 
import test

from terminaltables import AsciiTable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='ResNet18', help='model name')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='ResNet18/checkpoint/last.pth', help='checkpoint path')
    parser.add_argument('--saved-dir', type=str, default='pruned_model', help='pruned_model saved directory')
    parser.add_argument('--percent', type=float, default=0.5, help='channel prune percent')
    parser.add_argument('--layer-keep', type=float, default=0.01, help='channel keep percent per layer')
    parser.add_argument('--img-size', type=int, default=32, help='inference size (pixels)')

    # opt = parser.parse_args(args=[])
    opt = parser.parse_args()
    
    print(opt)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get original model
    checkpoint = opt.checkpoint
    model_name = opt.model_name
    assert os.path.isfile(checkpoint), 'Error: no %s file found!'%checkpoint
    if checkpoint.endswith('.pth'): # only load weights from file
        model = globals()[opt.model_name]()
        model.load_state_dict(torch.load(checkpoint)['model'])
    else: # load network architecture and weights from file 
        model = torch.load(checkpoint)

    resnet_cfg = ResNetCFG()
    model_cfg = resnet_cfg.get_cfg(model)
        
    if isinstance(model, ResNet): # convert ResNet model to PrunableResNet model
        prunable_model = globals()[opt.model_name](model_cfg)
        
        for src,dest in zip(model.parameters(),prunable_model.parameters()):
            dest.data.copy_(src.data)
            
        # parameters not include BN's running_mean and running_var, so it needs a special treatment
        for src,dest in zip(model.modules(),prunable_model.modules()):
            if isinstance(src,nn.BatchNorm2d):
                dest.running_mean.data.copy_(src.running_mean.data)
                dest.running_var.data.copy_(src.running_var.data)

        model = prunable_model

    model = model.to(device)
    
    eval_model = lambda model:test.test(model=model)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model:")
    origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)
    
    
    #######################################################################################################
    # 伪剪枝
    
    masked_model = deepcopy(model)
    
    module_list, shortcutCB_list = masked_model.get_module_list()
        
    # get the prune_idx which means the module idx in module_list that can be pruned 
    prune_idx = resnet_cfg.get_prune_idx(masked_model)
    
    # the number of all conv's filters
    total = 0 
    for m in masked_model.modules():
        if isinstance(m,nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    
    bn = torch.zeros(total)
    index = 0
    for i,m in enumerate(module_list):
        size = m['bn'].weight.data.shape[0]
        bn[index:(index+size)] = m['bn'].weight.data.abs().clone()
        index += size

                
    for m in shortcutCB_list:
        size = m['bn'].weight.data.shape[0]
        bn[index:(index+size)] = m['bn'].weight.data.abs().clone()
        index += size

    y, i = torch.sort(bn)
    thre_index = int(total * opt.percent)
    thre = y[thre_index] 
    
    total_pruned =0
    
    #+++++++++++++++++++++++++求cfg构成的module_list中的<Conv,BN>掩码+++++++++++++++++++++++++#
    cfg_pruned = []
    cfg_remain = []
    cfg_mask = []
    for i,m in enumerate(module_list): # 伪剪枝，利用掩码将gamma系数较小的置0
        weight_copy = m['bn'].weight.data.clone()
        channels = weight_copy.shape[0] # 当前卷积模块的输出通道数目
        
        # 当前卷积模块最少要保留的输出通道数目
        min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 1 
        if i in prune_idx:
            mask = weight_copy.abs().ge(thre)
            
            if int(torch.sum(mask)) < min_channel_num: # 当剪枝后输出通道数目小于当前卷积模块最少要保留的输出通道数目
                _, sorted_index_weights = torch.sort(weight_copy,descending=True)
                mask[sorted_index_weights[:min_channel_num]]=1. 
            
            tmp = mask.float().to(device)
            m['bn'].weight.data.mul_(tmp)
            m['bn'].bias.data.mul_(tmp)
            
            cfg_remain.append(int(torch.sum(mask))) # 后面需要通过cfg_remain(剩余的filter数目)来构造pruned_model
            cfg_pruned.append(len(mask) - cfg_remain[-1])
            cfg_mask.append(mask.clone()) # 后面需要通过cfg_mask来把masked_model中的一些参数复制到pruned_modelz中
            
            total_pruned += cfg_pruned[-1]
            
            print('prune_idx: {:5d} \t total: {:5d} \t remain: {:5d} \t pruned: {:5d}'.
                format(i, len(mask), cfg_remain[-1], cfg_pruned[-1]))
        else:
            mask = [1.]*weight_copy.shape[0]
            
            cfg_pruned.append(0)
            cfg_remain.append(len(mask))
            cfg_mask.append(mask)
    #+++++++++++++++++++++++++求cfg构成的module_list中的<Conv,BN>掩码 end+++++++++++++++++++++#        
    
    #+++++++++++++++++++++++++求dependency中每一个依赖关系的掩码+++++++++++++++++++++++++#
    dp_pruned = []
    dp_remain = []
    dp_mask = []
    # 表示依赖关系，每一个元素代表一个依赖关系，每一个依赖关系是由若干个有依赖关系的<Conv,BN>构成的
    # 有依赖关系的所有<Conv,BN>必须是同样的剩余filter数目，且掩码也要相同
    dp = masked_model.dependency 
    
    for i in range(len(dp)):
        merge_masks = []
        for j in range(len(dp[i])):
            weight_copy = dp[i][j]['bn'].weight.data.clone()
            mask = weight_copy.abs().ge(thre)
            merge_masks.append(mask.clone().unsqueeze(0))
        
        merge_masks = torch.cat(merge_masks, 0)
        """
        重点：
            * 剪枝掩码对1求并集，即但凡有一个卷积模块中的某个位置的输出通道不被剪枝，则所有卷积模块在该位置的输出通道都不被剪枝
            * 即剪枝掩码对0求交集,即只有所有卷积模块在某个位置的输出通道被剪枝，则所有卷积模块在该位置的输出通道才能被剪枝
        """
        merge_masks = (torch.sum(merge_masks, dim=0) > 0)
        
        merge_masks_tolist = merge_masks.cpu().numpy().tolist()
        remain = int(torch.sum(merge_masks))
        mask_total = len(merge_masks_tolist)
        pruned = mask_total - remain # 
        cb_num = len(dp[i]) # 当前的依赖关系中，有多少个<Conv,BN>
        
        for j in range(len(dp[i])):
            dp[i][j]['bn'].weight.data.mul_(merge_masks.float())
            dp[i][j]['bn'].bias.data.mul_(merge_masks.float())
            
            for k,m in enumerate(module_list): # 当该<Conv,BN>既在依赖关系中也在cfg构成的module_list中时，需要重写cfg_mask
                if id(m['bn']) == id(dp[i][j]['bn']):
                    cfg_remain[k] = remain
                    cfg_mask[k] = deepcopy(merge_masks_tolist)
        
        dp_remain.append(remain*cb_num)
        dp_pruned.append(pruned*cb_num)
        dp_mask.append(deepcopy(merge_masks_tolist))

        total_pruned += dp_pruned[-1]

        print('dependency_idx: {:5d} \t total: {:5d} \t remain: {:5d} \t pruned: {:5d}'.
            format(i, merge_masks.shape[0]*len(dp[i]), dp_remain[-1], dp_pruned[-1]))
    #+++++++++++++++++++++++++求dependency中每一个依赖关系的掩码 end+++++++++++++++++++++#
    
    real_ratio = total_pruned/total
    print("%s:%g"%('real_ratio',real_ratio))
    
    print("\nlet's test the masked_model:")
    masked_model_metric = eval_model(masked_model) # 测试masked_model
    masked_nparameters = obtain_num_parameters(masked_model)
    
    #####################################################################################################
    # 真正的剪枝
    
    pruned_model_cfg = resnet_cfg.convert_list_to_cfg(cfg_remain) # 剪枝后模型的cfg
    print(pruned_model_cfg)
    pruned_model = globals()[opt.model_name](pruned_model_cfg) # 通过cfg生成剪枝后的模型
    
    pruned_model = pruned_model.to(device)
    
    pruned_module_list, pruned_shortcutCB_list = pruned_model.get_module_list() # 剪枝后的模型的module_list
    
    # 从masked_model中拷贝参数到pruned_model
    for i,(m,p_m) in enumerate(zip(module_list,pruned_module_list)):
        conv = m['conv']
        bn = m['bn']
        p_conv = p_m['conv']
        p_bn = p_m['bn']
        
        in_channel_idx = [] # 当前层剩余的输入通道，即剩余的深度idx,即掩码为1的深度idx
        if i == 0: # 第0个卷积前面没有层了，需要特殊考虑
            for j in range(conv.weight.data.shape[1]):
                in_channel_idx.append(j)
        else:
            for j in range(len(cfg_mask[i-1])):
                if cfg_mask[i-1][j]:
                    in_channel_idx.append(j)

        out_channel_idx = [] # 当前层剩余的输出通道，即剩余的filter的idx,即掩码为1的filter的idx
        for j in range(len(cfg_mask[i])):
            if cfg_mask[i][j]:
                out_channel_idx.append(j)

        # 对BN层的参数赋值
        p_bn.weight.data.copy_(bn.weight.data[out_channel_idx])
        p_bn.bias.data.copy_(bn.bias.data[out_channel_idx])
        p_bn.running_mean.data.copy_(bn.running_mean.data[out_channel_idx])
        p_bn.running_var.data.copy_(bn.running_var.data[out_channel_idx])

        # 对卷积层的参数赋值
        tmp = conv.weight.data[:,in_channel_idx,:,:]
        p_conv.weight.data.copy_(tmp[out_channel_idx,:,:,:])

    init_idx =  1
    for i,(m,p_m) in enumerate(zip(shortcutCB_list,pruned_shortcutCB_list)):
        conv = m['conv']
        bn = m['bn']
        p_conv = p_m['conv']
        p_bn = p_m['bn']
        
        # 对于ResNet18， ResNet34来说，idx为0的依赖关系中不包含shortcutCB
        # 对于ResNet50， ResNet101, ResNet152来说，idx为0的依赖关系是整个model的第0个卷积层
        idx = init_idx + i 

        in_channel_idx = [] # 当前层剩余的输入通道，即剩余的深度idx,即掩码为1的深度idx
        for j in range(len(dp_mask[idx-1])):
            if dp_mask[idx-1][j]:
                in_channel_idx.append(j)

        out_channel_idx = [] # 当前层剩余的输出通道，即剩余的filter的idx,即掩码为1的filter的idx
        for j in range(len(dp_mask[idx])):
            if dp_mask[idx][j]:
                out_channel_idx.append(j)

        # 对BN层的参数赋值
        p_bn.weight.data.copy_(bn.weight.data[out_channel_idx])
        p_bn.bias.data.copy_(bn.bias.data[out_channel_idx])
        p_bn.running_mean.data.copy_(bn.running_mean.data[out_channel_idx])
        p_bn.running_var.data.copy_(bn.running_var.data[out_channel_idx])

        # 对卷积层的参数赋值
        tmp = conv.weight.data[:,in_channel_idx,:,:]
        p_conv.weight.data.copy_(tmp[out_channel_idx,:,:,:])
        
    # 全连接层的参数赋值
    collum_idx = [] # 上一个卷积层剩余的filter数目等于当前全连接层权重矩阵的列数
    for j in range(len(dp_mask[-1])):
        if dp_mask[-1][j]:
            collum_idx.append(j)
    pruned_model.linear.weight.data.copy_(masked_model.linear.weight.data[:,collum_idx])
    pruned_model.linear.bias.data.copy_(masked_model.linear.bias.data)
    
    print("\nlet's test the pruned_model:")
    pruned_model_metric = eval_model(pruned_model) # 测试pruned_model
    pruned_nparameters = obtain_num_parameters(pruned_model)

    #####################################################################################################
    
    random_input = torch.rand((1, 3, opt.img_size, opt.img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time
    
    print('\ntesting avg forward time...')
    origin_forward_time = obtain_avg_forward_time(random_input, model)
    masked_forward_time = obtain_avg_forward_time(random_input, masked_model)
    pruned_forward_time = obtain_avg_forward_time(random_input, pruned_model)
    
    origin_acc = 100.*origin_model_metric[1]/origin_model_metric[2]
    masked_acc = 100.*masked_model_metric[1]/masked_model_metric[2]
    pruned_acc = 100.*pruned_model_metric[1]/pruned_model_metric[2]
    
    metric_table = [
        ["Metric",     "origin",                     "masked",                     "pruned"],
        ["Acc",        f"{origin_acc:.6f}",          f"{masked_acc:.6f}",          f"{pruned_acc:.6f}"],
        ["Parameters", f"{origin_nparameters}",      f"{masked_nparameters}",      f"{pruned_nparameters}"],
        ["Inference",  f"{origin_forward_time:.4f}", f"{masked_forward_time:.4f}", f"{pruned_forward_time:.4f}"]
    ]
    
    print(AsciiTable(metric_table).table)
    

    #####################################################################################################
    
    if not os.path.isdir(opt.saved_dir):
        os.mkdir(opt.saved_dir)
    cfg_saved_path = "pruned_percent-%g_%s.cfg"%(opt.percent,opt.model_name)
    weights_saved_path = "pruned_percent-%g_%s.pth"%(opt.percent,opt.model_name)

    cfg_saved_path = os.path.join(opt.saved_dir, cfg_saved_path)
    weights_saved_path = os.path.join(opt.saved_dir, weights_saved_path)

    cfg_state = {
        'cfg': pruned_model.cfg
    }

    weights_state = {
        'model': pruned_model.state_dict()
    }

    torch.save(cfg_state, cfg_saved_path)
    torch.save(weights_state, weights_saved_path)

    print(f"pruned_model's cfg has been saved: {cfg_saved_path}")
    print(f"pruned_model's weights has been saved: {weights_saved_path}")