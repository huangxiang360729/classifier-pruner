#!/bin/bash

device=0
prefix=exp_VGG16
model=VGG16
epochs=100
batch_size=128
s=0.001
prune_mode=vgg_prune.py
prune=0
baseline_lr=0.1
sp_lr=0.01

echo "device=${device}"
echo "prefix=${prefix}"
echo "model=${model}"
echo "epochs=${epochs}"
echo "batch_size=${batch_size}"
echo "s=${s}"
echo "prune=${prune}"
echo "baseline_lr=${baseline_lr}"
echo "sp_lr=${sp_lr}"

############################################################
mkdir -p ${prefix}/baseline

echo "################ start baseline"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    --model-name ${model} \
    -ckpt-dir ${prefix}/baseline/checkpoint \
    -log ${prefix}/baseline/runs \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --lr ${baseline_lr}
echo "################ end baseline"


############################################################
mkdir -p ${prefix}/sp

echo "################ start sp"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    -sr \
    --s ${s} \
    --model-name ${model} \
    -ckpt-dir ${prefix}/sp/checkpoint \
    -ckpt ${prefix}/baseline/checkpoint/last.pth \
    -log ${prefix}/sp/runs \
    --prune ${prune} \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --lr ${sp_lr}
echo "################ end sp"

for percent in 0.5
do
    ############################################################
    mkdir -p ${prefix}/percent-${percent}

    echo "################ start prune"
    CUDA_VISIBLE_DEVICES=${device} python ${prune_mode} \
        --model-name ${model} \
        -ckpt ${prefix}/sp/checkpoint/last.pth \
        --saved-dir ${prefix}/percent-${percent}/pruned_model \
        --percent ${percent}
    echo "################ end prune"

    # ###########################################################
    # mkdir -p ${prefix}/percent-${percent}/finetune_lr-0.1

    # CUDA_VISIBLE_DEVICES=${device} python train.py \
    #     --model-name ${model} \
    #     -ckpt-dir ${prefix}/percent-${percent}/finetune_lr-0.1/checkpoint \
    #     --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.cfg \
    #     -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.pth \
    #     -log ${prefix}/percent-${percent}/finetune_lr-0.1/runs \
    #     --lr 0.1 \
    #     --epochs ${epochs} \
    #     --batch-size ${batch_size}

    ############################################################
    mkdir -p ${prefix}/percent-${percent}/finetune_lr-0.01

    CUDA_VISIBLE_DEVICES=${device} python train.py \
        --model-name ${model} \
        -ckpt-dir ${prefix}/percent-${percent}/finetune_lr-0.01/checkpoint \
        --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.cfg \
        -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${model}.pth \
        -log ${prefix}/percent-${percent}/finetune_lr-0.01/runs \
        --lr 0.01 \
        --epochs ${epochs} \
        --batch-size ${batch_size}

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