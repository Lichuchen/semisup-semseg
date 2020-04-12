#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=5Net \
python -u ../train_s4GAN.py  \
--dataset RV  \
--checkpoint-dir ../checkpoints/RV2/RV_semi_0_75 \
--labeled-ratio 0.75 \
--ignore-label 255 \
--num-classes 2 \
--data-dir /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/final/train/ocontour \
--data-list /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/final/train/ocontour/train.txt \
--input-size '216,216' \
--batch-size 4 \
--save-pred-every 500 \
--num-steps 12000 \
2>&1 | tee ../log/RV/traino_semi_gan_0.75.log
