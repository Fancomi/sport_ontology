"""
infer
"""
import sys
import torch
import time

dev = int(sys.argv[1])
# 设置默认的 CUDA 设备
torch.cuda.set_device(dev)

# 创建一个张量并将其放满 GPU
# x = torch.tensor(range(20000000)).cuda()
x = torch.tensor(range(2000)).cuda()
tmp = []
for i in range(100):
    tmp.append(torch.tensor(range(10000)).cuda())

# 无限循环，占用 GPU 资源
while True:
    # 在 GPU 上执行一些操作
    x = x * 2
    x = torch.sin(x)

    # 可以添加一些延迟，以控制循环的速度
    # time.sleep(0.002)
