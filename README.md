BEGAN
===============

Code accompanying the paper ["BEGAN"](https://arxiv.org/abs/1703.10717), with MaxPool2d as subsampling.

![gensample](imgs/fake_samples_72500.png,"32x32 samples")

### Progress

- [x] 32x32
- [x] skip connection
- [] 64x64,128x128
- [] Training G and D independently
- [] refinement

### Train

default gamma=0.5

```bash
python main.py --dataset CelebA --dataroot [CelebA-train-folder] --cuda
```
