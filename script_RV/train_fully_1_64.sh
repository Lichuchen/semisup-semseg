#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=Net \
python -u ../train_full.py  \
--dataset CMR  \
--checkpoint-dir ../checkpoints/RV2/RV_fully_1_64 \
--ignore-label 255 \
--num-classes 2 \
--labeled-ratio 0.015625 \
--data-dir /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/final/train/ocontour \
--data-list /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/final/train/ocontour/train.txt \
--input-size '216,216' \
--batch-size 4 \
--save-pred-every 500 \
--num-steps 10000 \
--split-id /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/checkpoints/RV2/RV_semi_1_64/train_voc_split.pkl \
2>&1 | tee ../log/RV/traino_fully_0.25.log
