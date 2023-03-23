# EDSR,MDSR (CVPR 2017)

- LR image: 48x48 size
- augmentation: random horizontal flip, 90 rotation
- substracting the mean RGB value
- Optimizer: Adam, b1=0.9, b2=0.999, e=10e-8
- Scheduler: Multistep (milestone=[180,360,540,720], gamma=0.5)
- loss: L1
- batch size: 64
- train epoch: 800

## Experiment result

|  | PSNR(x2,x3,x4) | SSIM(x2,x3,x4) |
| --- | --- | --- |
| DIV2K val | 32.02 |0.9132  |
| Set5 |  |  |
| Set14 |  |  |
| B100 |  |  |
| Urban 100 |  |  |