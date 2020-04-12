#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=Net \
python -u ../train_full.py  \
--dataset CMR  \
--checkpoint-dir ../checkpoints/sunny_fully_1_4 \
--labeled-ratio 0.25 \
--ignore-label 255 \
--num-classes 2 \
--data-dir /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/sunnybrook \
--data-list /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/sunnybrook/train/train.txt \
--input-size '128,128' \
--batch-size 4 \
--save-pred-every 500 \
--num-steps 10000 \
--split-id /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/checkpoints/sunny_semi_0_25/train_voc_split.pkl \
2>&1 | tee ../log/sunny/train_fully_0.25.log
