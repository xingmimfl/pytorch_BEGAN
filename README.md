BEGAN
===============

Code accompanying the paper ["BEGAN"](https://arxiv.org/abs/1703.10717), with MaxPool2d as subsampling.

![image](https://github.com/xingmimfl/pytorch_BEGAN/blob/master/imgs/fake_samples_72500.png)
![image](https://github.com/xingmimfl/pytorch_BEGAN/blob/master/imgs/fake_samples_202500_64x64.png)

### Progress

- [x] 32x32
- [x] skip connection
- [x] 64x64,128x128
- [] Training G and D independently
- [] refinement

### Train

default gamma=0.5

```bash
"""
32x32: imageSize=32, size=3
64x64: imageSize=64, size=4
128x128: imageSize=128, size=5
"""
python main.py --dataset CelebA --dataroot [CelebA-train-folder] --imageSize 64 --size 4 --gamma 0.5 --cuda
```
