# classifier-pruner
Using Network-Slimming to prune classifier.

Network-Slimming come form [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).

## Now, it only support prune ResNet(18 36 50 101 152) on CIFAR10
Of course, it is easy to extend this code to other classification datasets.  

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
|     s | sparsity factor(in paper, it is lambda) |  
| prune | sparse mode, used to control which BN layer's gamma need to be sparse |  

### prune
prune the filters with small corresponding gamma.
```shell
python prune.py --model-name [eg. ResNet18] \
  -ckpt [the file path of sp's checkpoint] \
  -saved-dir [the directory to save pruned model's cfg and weights] \
  -percent [the prune ratio]
```
After prune, it will create pruned cfg and weights that inherit the some weights of the original model.

percent = (the filters pruned) / (the filters that can be pruned),  "the filters that can be pruned" means all filters in a layer which idx is in prune_idx.

### finetune 
finetune pruned model to recover the Acc.
```shell
python prune.py --model-name [eg. ResNet18] \
  --cfg [the config file of pruned model] \
  -ckpt [the file path of pruned_model's weights] \
  -saved-dir [the directory to save finetune model's cfg and weights] \
  -log [the directory to save tensorboard output] \
  --lr [learning rate] \
  --epochs [epochs]
```

### from-scratch 
train pruned model from scratch.
```shell
python prune.py --model-name [eg. ResNet18] \
  --cfg [the config file of pruned model] \
  -ckpt [the file path of pruned_model's weights] \
  -saved-dir [the directory to save finetune model's cfg and weights]  \
  -log [the directory to save tensorboard output] \
  --lr [learning rate] \
  --epochs [epochs] \
  --from-scratch
```
