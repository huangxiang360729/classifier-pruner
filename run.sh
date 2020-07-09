#!/bin/bash

device=0
prefix=exp_ResNet152
model=ResNet152
epochs=100
batch_size=64


############################################################
mkdir -p ${prefix}/baseline

echo "################ start baseline"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    --model-name ${model} \
    -ckpt-dir ${prefix}/baseline/checkpoint \
    -log ${prefix}/baseline/runs \
    --epochs ${epochs} \
    --batch-size ${batch_size}
echo "################ end baseline"


############################################################
mkdir -p ${prefix}/sp

echo "################ start sp"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    -sr \
    --model-name ${model} \
    -ckpt-dir ${prefix}/sp/checkpoint \
    -ckpt ${prefix}/baseline/checkpoint/last.pth \
    -log ${prefix}/sp/runs \
    --prune 0 \
    --epochs ${epochs} \
    --batch-size ${batch_size}
echo "################ end sp"

for percent in 0.9
do
    ############################################################
    mkdir -p ${prefix}/percent-${percent}

    echo "################ start prune"
    CUDA_VISIBLE_DEVICES=${device} python prune.py \
        --model-name ${model} \
        -ckpt ${prefix}/sp/checkpoint/last.pth \
        --saved-dir ${prefix}/percent-${percent}/pruned_model \
        --percent ${percent}
    echo "################ end prune"

    ###########################################################
    mkdir -p ${prefix}/percent-${percent}/finetune_lr-0.1

    CUDA_VISIBLE_DEVICES=${device} python train.py \
        --model-name ${model} \
        -ckpt-dir ${prefix}/percent-${percent}/finetune_lr-0.1/checkpoint \
        --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.cfg \
        -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.pth \
        -log ${prefix}/percent-${percent}/finetune_lr-0.1/runs \
        --lr 0.1 \
        --epochs ${epochs} \
        --batch-size ${batch_size}


    # ############################################################
    # mkdir -p ${prefix}/percent-${percent}/finetune_lr-0.01

    # CUDA_VISIBLE_DEVICES=${device} python train.py \
    #     --model-name ${model} \
    #     -ckpt-dir ${prefix}/percent-${percent}/finetune_lr-0.01/checkpoint \
    #     --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.cfg \
    #     -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.pth \
    #     -log ${prefix}/percent-${percent}/finetune_lr-0.01/runs \
    #     --lr 0.01 \
    #     --epochs ${epochs} \
    #     --batch-size ${batch_size}

    # ###########################################################
    # mkdir -p ${prefix}/percent-${percent}/from_scratch

    # CUDA_VISIBLE_DEVICES=${device} python train.py \
    #     --model-name ${model} \
    #     -ckpt-dir ${prefix}/percent-${percent}/from_scratch/checkpoint \
    #     --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.cfg \
    #     -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.pth \
    #     -log ${prefix}/percent-${percent}/from_scratch/runs \
    #     --from-scratch \
    #     --epochs ${epochs} \
    #     --batch-size ${batch_size}
done

echo "################ finish!!!"
