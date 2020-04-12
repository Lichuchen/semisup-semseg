#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=Net \
python -u ../evaluate.py \
--dataset pascal_voc  \
--num-classes 21 \
--restore-from /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/checkpoints/voc_semi_0_125/VOC_10000.pth
