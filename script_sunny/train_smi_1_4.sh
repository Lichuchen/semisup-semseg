#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=net \
python -u ../train_s4GAN.py  \
--dataset CMR  \
--checkpoint-dir ../checkpoints/RV/sunny_semi_0_25 \
--labeled-ratio 0.25 \
--ignore-label 255 \
--num-classes 2 \
--data-dir /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/RV \
--data-list /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/RV/train/train.txt \
--input-size '216,216' \
--batch-size 4 \
--save-pred-every 500 \
--num-steps 10000 \
2>&1 | tee ../log/RV/train_semi_gan_0.25.log
