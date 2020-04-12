import torch
import os
import torch.backends.cudnn as cudnn
os.environ['CUDA_VISIBLE_DEVICES']='0'

path1 = '/mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/checkpoints/RV2/RV_semi_0_125/VOC_7000.pth'
path2 = '/mnt/lustre/lichuchen/lily/few-shot/semisup-semseg/checkpoints/try/VOC_7000.pth'

a = torch.load(path1,map_location='cpu')
b = torch.load(path2,map_location='cpu')

print(a.keys())

print(b.keys())
