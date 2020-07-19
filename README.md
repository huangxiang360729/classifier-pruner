# classifier-pruner
Using Network-Slimming to prune classifier.

[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017)

# Git update log
| timestamp | description |  related file |  
| :---: | :-----:| :-----: |
|2020-07-16| debug vgg_prune.py, add p_conv's bias parameters copy | vgg_prune.py | 
|2020-07-19| upload vgg_sensitity.ipynb (analyze each layer's sensitivity for prune) | vgg_sensitity.ipynb |
|2020-07-19| upload vgg_l1-norm_ns_random.ipynb (analyze each layer's sensitivity for l1-norm algorithm, network slimming algorithm and random prune algorithm) | vgg_l1-norm_ns_random.ipynb |

## Dependencies
- pytorch v1.5.1
- torchvision v0.6.1
- terminaltables v3.1.0
- tensorboardX v2.0  
- tqdm v4.46.1
- torchsummary v1.5.1

## Limitation
### Models that support pruning：
- ResNet(18, 36, 50, 101, 152) 
- VGG(11, 13, 16, 19)

### Supported datasets:
- CIFAR10  
Of course, it is easy to extend this code to other classification datasets.  

### Prune mode:
- VGG
  + vgg_prune.py
    > There is no dependency between VGG's Conv, so all Convs can be pruned.  

- ResNet
  + resnet_prune.py(prune = 0): 
    > It can only prune the first Conv's filters in every BasicBlock or First two Conv's filters in every Bottleneck.   
    > When dealing with Sparse-regularization（sp）, prune shoud set to 0， that is "--prune 0"  
  + resnet_slim_prune.py(prune = 1):   
    > It can prune all Conv's filters.  
    > When multiple Convs depend on each other (these Convs need to have the same number of filters), we can only prune the filters that each Conv thinks can be prune. That is, when seeking the pruning mask, you need to find the union of '1' (or we can also say the intersection of '0'), where '1' indicates reservation and '0' indicates pruning.  
    > When dealing with Sparse-regularization（sp）, prune shoud set to 1， that is "--prune 1"  

Next time, I will take more models into account when pruning.

## ResNet's cfg file
To prune ResNet Convenience, I have change the function to create ResNet model.

I define a cfg and use it to create ResNet pruned model.

cfg  is a 2D integer list:
  - row index means the block idx in ResNet
  - collum index means the Conv layer idx in the block (the first Conv, BasicBlock or Bottleneck)
  - each element value is the number of Conv's output chananel (or we can say the Conv's filter number)

When prune is complete, I generate a cfg file to save the model's architecture by using 'torch.save()' function.

## Usage
### baseline
training from scratch to get the baseline of Acc, this step will get the baseline's weights.
```shell
python train.py --model-name [eg. ResNet18] \
  -ckpt-dir [the directory to save checkpoint] \
  -log [the directory to save tensorboard output]  \
  --lr [learning rate] --epochs [epochs]
```

### Sparse-regularization（sp）
training with Sparse-regularization from baseline's weights, this step will get the weights which with sparse gamma.
```shell
python train.py --model-name [eg. ResNet18] \
  -ckpt [the file path of baseline's checkpoint] \
  -ckpt-dir [the directory to save checkpoint] \
  -log [the directory to save tensorboard output] \
  --lr [learning rate] \
  --epochs [epochs] \
  -sr \
  --s 0.001 \
  --prune 0
```
| param | description |  
| :---: | :-----:| 
|    sr | the switch to open sparse-regularization |  
|     s | sparsity factor(in [paper](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html), it is lambda) |  
| prune | sparse mode, used to control which BN layer's gamma need to be sparse and only works under ResNet(VGG does not need to set this parameter) |  

### prune
prune the filters with small corresponding gamma.
```shell
python resnet_prune.py --model-name [eg. ResNet18] \
  -ckpt [the file path of sp's checkpoint] \
  -saved-dir [the directory to save pruned model's cfg and weights] \
  -percent [the prune ratio]
```
After prune, it will create pruned cfg and weights that inherit the some weights of the original model.

percent = (the filters pruned) / (the filters that can be pruned),  "the filters that can be pruned" means all filters in a layer which idx is in prune_idx.

### finetune 
finetune pruned model to recover the Acc.
```shell
python train.py --model-name [eg. ResNet18] \
  --cfg [the config file of pruned model] \
  -ckpt [the file path of pruned_model's weights] \
  -ckpt-dir [the directory to save finetune model's cfg and weights] \
  -log [the directory to save tensorboard output] \
  --lr [learning rate] \
  --epochs [epochs]
```

### from-scratch 
train pruned model from scratch.
```shell
python train.py --model-name [eg. ResNet18] \
  --cfg [the config file of pruned model] \
  -ckpt [the file path of pruned_model's weights] \
  -ckpt-dir [the directory to save finetune model's cfg and weights]  \
  -log [the directory to save tensorboard output] \
  --lr [learning rate] \
  --epochs [epochs] \
  --from-scratch
```

### run
I also write a shell script to run baseline -> sp -> prune -> finetune
```shell
sh run_resnet18_prune.sh
```
