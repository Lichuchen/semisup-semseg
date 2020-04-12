#!/usr/bin/env bash

# mkdir checkpoints/tracklet-fconly

GPU_NUM=$(($1<8?$1:8))
dcl_fix=25

srun \
  --mpi=pmi2  --gres=gpu:${GPU_NUM}  -n$1 --ntasks-per-node=${GPU_NUM} --cpus-per-task=2 --partition=VI_OP_1080TI --job-name=Net \
python -u ../evaluate.py \
--dataset RV  \
--num-classes 2 \
--data-dir /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/final/test2/icontour \
--data-list /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/data/cmr/final/test2/icontour/test.txt \
--restore-from /mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/checkpoints/RV_other_i/RV_fully_1_20/VOC_500.pth \
