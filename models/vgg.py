'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class PrunableVGG(nn.Module):
    def __init__(self, cfg):
        super(PrunableVGG, self).__init__()
        self.cfg = cfg
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-2], 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    # module_list: 包含所有的<Conv, BN>
    def get_module_list(self): # get the module_list that only contain CONV and BN
        self.module_list = []
        CB = []
        for name,module in self.named_modules():
            if isinstance(module, nn.Conv2d): 
                CB.append(module)
                
            if isinstance(module, nn.BatchNorm2d):
                CB.append(module)
                self.module_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
                
        return self.module_list
    
class VGGCFG:
    # 将一个1D的list变成带'M'的cfg
    def convert_list_to_cfg(self, model, li):
        raw_cfg = model.cfg
        
        cfg = []
        idx = 0
        for i in range(len(raw_cfg)):
            if raw_cfg[i] == 'M':
                cfg.append('M')
            else:
                cfg.append(li[idx])
                idx += 1
        return cfg

def VGG11(cfg_=None):
    if cfg_ is None:
        return PrunableVGG(cfg['VGG11'])
    else:
        return PrunableVGG(cfg_)
    
def VGG13(cfg_=None):
    if cfg_ is None:
        return PrunableVGG(cfg['VGG13'])
    else:
        return PrunableVGG(cfg_)
    
def VGG16(cfg_=None):
    if cfg_ is None:
        return PrunableVGG(cfg['VGG16'])
    else:
        return PrunableVGG(cfg_)
    
def VGG19(cfg_=None):
    if cfg_ is None:
        return PrunableVGG(cfg['VGG19'])
    else:
        return PrunableVGG(cfg_)

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
