#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=Net \
python -u ../train_s4GAN.py  \
--dataset pascal_voc  \
--checkpoint-dir ../checkpoints/voc_semi_0_125 \
--labeled-ratio 0.125 \
--ignore-label 255 \
--num-classes 21 \
--split-id ../splits/voc/split_0.pkl
2>&1 | tee ../log/voc/train_semi_gan.log
