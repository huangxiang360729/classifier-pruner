'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class PrunableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PrunableBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes[1])
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PrunableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PrunableBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        
        self.conv3 = nn.Conv2d(planes[1], planes[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes[2])

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes[2]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes[2])
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class PrunableResNet(nn.Module):
    def __init__(self, block, num_blocks, cfg=None, num_classes=10):
        super(PrunableResNet, self).__init__()
        assert not(cfg is None)
        self.cfg = cfg
            
        self.in_planes = cfg[0][0]

        self.conv1 = nn.Conv2d(3, cfg[0][0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0][0])
        
        start = 1
        end = start + num_blocks[0]
        self.layer1 = self._make_layer(block, cfg[start:end], num_blocks[0], stride=1)
        
        start = end
        end = start + num_blocks[1]
        self.layer2 = self._make_layer(block, cfg[start:end], num_blocks[1], stride=2)
        
        start = end
        end = start + num_blocks[2]
        self.layer3 = self._make_layer(block, cfg[start:end], num_blocks[2], stride=2)
        
        start = end
        end = start + num_blocks[3]
        self.layer4 = self._make_layer(block, cfg[start:end], num_blocks[3], stride=2)
        
        self.linear = nn.Linear(cfg[-1][-1], num_classes)
        
        

    def _make_layer(self, block, blocks_planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for i,stride in enumerate(strides):
            # print(blocks_planes[i])
            b = block(self.in_planes, blocks_planes[i],stride)
            layers.append(b)
            self.in_planes = blocks_planes[i][-1]
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_module_list(self): # get the module_list that only contain CONV and BN, but not contain CONV and BN which is in shortcut
        self.module_list = []
        self.shortcutCB_list = []
        CB = []
        for name,module in self.named_modules():
            if 'conv' in name:
                CB.append(module)
            elif isinstance(module, nn.Conv2d): # shortcut-CONV
                CB.append(module)
                
            if 'bn' in name: 
                CB.append(module)
                self.module_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
            elif isinstance(module, nn.BatchNorm2d): # shortcut-BN
                CB.append(module)
                self.shortcutCB_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
                
        return self.module_list, self.shortcutCB_list
        

class ResNetCFG:
    def __init__(self):
        self.cfg18 = [[64], [64, 64], [64, 64], [128, 128], [128, 128], [256, 256], [256, 256], [512, 512], [512, 512]]
        self.cfg34 = [[64], [64, 64], [64, 64], [64, 64], [128, 128], [128, 128], [128, 128], [128, 128], [256, 256], \
                        [256, 256], [256, 256], [256, 256], [256, 256], [256, 256], [512, 512], [512, 512], [512, 512]]
        self.cfg50 = [[64], [64, 64, 256], [64, 64, 256], [64, 64, 256], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                      [128, 128, 512], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                      [256, 256, 1024], [256, 256, 1024], [512, 512, 2048], [512, 512, 2048], [512, 512, 2048]] 
        self.cfg101 = [[64], [64, 64, 256], [64, 64, 256], [64, 64, 256], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                       [128, 128, 512], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [512, 512, 2048], \
                       [512, 512, 2048], [512, 512, 2048]]
        self.cfg152 = [[64], [64, 64, 256], [64, 64, 256], [64, 64, 256], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                       [128, 128, 512], [128, 128, 512], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [512, 512, 2048], [512, 512, 2048], [512, 512, 2048]]
        
        self.prune_idx18 = [1,3,5,7,9,11,13,15]
        self.prune_idx34 = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]

        # self.prune_idx50 = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44,46,47]
        self.prune_idx50 = []
        for idx in range(50 - 1): # 不算全连接层，有49个卷积层(也不包括shortcut支路上的卷积层)，能被3整除的代表第0个卷积层或block中的最后一个卷积层
            if idx % 3 != 0:
                self.prune_idx50.append(idx)

        self.prune_idx101 = []
        for idx in range(101 - 1): # 不算全连接层，有100个卷积层(也不包括shortcut支路上的卷积层)，能被3整除的代表第0个卷积层或block中的最后一个卷积层
            if idx % 3 != 0:
                self.prune_idx101.append(idx)

        self.prune_idx152 = []
        for idx in range(152 - 1): # 不算全连接层，有151个卷积层(也不包括shortcut支路上的卷积层)，能被3整除的代表第0个卷积层或block中的最后一个卷积层
            if idx % 3 != 0:
                self.prune_idx152.append(idx)

    def get_cfg(self, raw_model): # raw_model is generated by class ResNet
        layers = 1 # '1' is linear layer
        for name,params in raw_model.named_parameters():
            if 'conv' in name:
                layers = layers + 1
        
        if layers in [18, 34]:
            block_convs = 2
        else:
            block_convs = 3
        
        first_conv_flag = True
        cfg = []
        block = []
        count = 0
        for name,params in raw_model.named_parameters():
            if 'conv' in name:
                if first_conv_flag: # first conv's output channels
                    block = [params.shape[0]] 
                    cfg.append(deepcopy(block))
                    block.clear()
                    first_conv_flag = False
                else: # all conv's output channels in a block, but shortcut's conv is not included  
                    block.append(params.shape[0])
                    count = count + 1
                    if count == block_convs:
                        cfg.append(deepcopy(block))
                        block.clear()
                        count = 0
        return cfg
    
    def get_prune_idx(self, model):
        layers = 1 # '1' is linear layer
        for name,params in model.named_parameters():
            if 'conv' in name:
                layers = layers + 1
        
        if layers == 18:
            return self.prune_idx18
        
        if layers == 34:
            return self.prune_idx34
            
        if layers == 50:
            return self.prune_idx50
            
        if layers == 101:
            return self.prune_idx101
            
        if layers == 152:
            return self.prune_idx152
        
        assert False,"Error: model's layers=%g is illegal!!!"%layers
    
    def convert_list_to_cfg(self, li):
        layers = len(li) + 1 # '1' is linear layer
        if layers in [18, 34]:
            block_convs = 2
        else:
            block_convs = 3
            
        first_conv_flag = True
        cfg = []
        block = []
        count = 0
        for i in range(len(li)):
            if first_conv_flag: # first conv's output channels
                block = [li[i]] 
                cfg.append(deepcopy(block))
                block.clear()
                first_conv_flag = False
                
            else: # all conv's output channels in a block, but shortcut's conv is not included  
                block.append(li[i])
                count = count + 1
                if count == block_convs:
                    cfg.append(deepcopy(block))
                    block.clear()
                    count = 0
        return cfg
        
def ResNet18(cfg=None):
    if cfg is None:
        # return ResNet(BasicBlock, [2, 2, 2, 2])
        return PrunableResNet(PrunableBasicBlock, [2, 2, 2, 2], cfg=ResNetCFG().cfg18)
    else:
        return PrunableResNet(PrunableBasicBlock, [2, 2, 2, 2], cfg=cfg)

def ResNet34(cfg=None):
    if cfg is None:
        # return ResNet(BasicBlock, [3, 4, 6, 3])
        return PrunableResNet(PrunableBasicBlock, [3, 4, 6, 3], cfg=ResNetCFG().cfg34)
    else:
        return PrunableResNet(PrunableBasicBlock, [3, 4, 6, 3], cfg=cfg)
    
def ResNet50(cfg=None):
    if cfg is None:
        # return ResNet(Bottleneck, [3, 4, 6, 3])
        return PrunableResNet(PrunableBottleneck, [3, 4, 6, 3], cfg=ResNetCFG().cfg50)
    else:
        return PrunableResNet(PrunableBottleneck, [3, 4, 6, 3], cfg=cfg)
    
def ResNet101(cfg=None):
    if cfg is None:
        # return ResNet(Bottleneck, [3, 4, 23, 3])
        return PrunableResNet(PrunableBottleneck, [3, 4, 23, 3], cfg=ResNetCFG().cfg101)
    else:
        return PrunableResNet(PrunableBottleneck, [3, 4, 23, 3], cfg=cfg)

def ResNet152(cfg=None):
    if cfg is None:
        # return ResNet(Bottleneck, [3, 8, 36, 3])
        return PrunableResNet(PrunableBottleneck, [3, 8, 36, 3], cfg=ResNetCFG().cfg152)
    else:
        return PrunableResNet(PrunableBottleneck, [3, 8, 36, 3], cfg=cfg)

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
