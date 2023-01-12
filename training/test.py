# Test results of training
import ultralytics
ultralytics.checks()

import torch
print(torch.cuda.is_available())
torch.zeros(1).cuda()

# py -m pip install nvidia-pyindex
# IS NEEDED