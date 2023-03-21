#!/bin/bash

TASK=$1
DEVICE=cuda:0

NAME=Exp91.DistgEPIT_wider.B4.L5e-5.E100.P128_32mp.S6.EMA0.999.FTT
MODEL=DistgEPIT_wider
SOURCE=vanilla
MODEL_PATH=checkpoints/Exp91.DistgEPIT_wider.B4.L5e-5.E100.P128_32mp.S6.EMA0.999.FTT.pth
PATCH=32
STRIDE=16

# NAME=DistgSSR_32x8_6x6x128_finetune
# MODEL=DistgSSR
# SOURCE=wzq
# MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x128_finetune.pth
# PATCH=84
# STRIDE=42

if [[ "$TASK" = "train" ]]; then
    python lfsr.py \
      --name $NAME \
      --device $DEVICE \
      --task train \
      --dataset LFSR.ALL \
      --train-batchsize 4 \
      --model $MODEL \
      --train-epoch 100 \
      --train-lr 2e-4 \
      --scale 4 \
      --ema-decay 0.999

elif [[ "$TASK" = "val" ]]; then
    python lfsr.py \
      --name $NAME.VAL \
      --device $DEVICE \
      --task val_all \
      --model $MODEL \
      --model-source $SOURCE \
      --model-path $MODEL_PATH \
      --test-patch $PATCH \
      --test-stride $STRIDE \
      --scale 4

elif [[ "$TASK" = "val_tta" ]]; then
    python lfsr.py \
      --name $NAME.VAL.TTA \
      --device $DEVICE \
      --task val_all \
      --model $MODEL \
      --model-source $SOURCE \
      --model-path $MODEL_PATH \
      --test-patch $PATCH \
      --test-stride $STRIDE \
      --tta \
      --scale 4

elif [[ "$TASK" = "test_val" ]]; then
    python lfsr.py \
      --name $NAME.NTIRE.VAL \
      --device $DEVICE \
      --task test \
      --dataset LFSR.NTIRE.VAL \
      --model $MODEL \
      --model-source $SOURCE \
      --model-path $MODEL_PATH \
      --test-patch $PATCH \
      --test-stride $STRIDE \
      --tta \
      --scale 4

elif [[ "$TASK" = "test" ]]; then
    python lfsr.py \
      --name $NAME.NTIRE.TEST \
      --device $DEVICE \
      --task test \
      --dataset LFSR.NTIRE.TEST \
      --model $MODEL \
      --model-source $SOURCE \
      --model-path $MODEL_PATH \
      --test-patch $PATCH \
      --test-stride $STRIDE \
      --tta \
      --scale 4
fi