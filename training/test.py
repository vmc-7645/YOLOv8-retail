# Run to see if environment is set up correctly

import ultralytics
import torch
ultralytics.checks()
print(torch.cuda.is_available())
print("Torch for CUDA version (not CUDA): "+str(torch.version.cuda))
torch.zeros(1).cuda()
torch.cuda.empty_cache()
