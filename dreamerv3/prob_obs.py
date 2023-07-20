
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

a = np.zeros((3, 64, 64))
print(a)

print(torch.distributions.Independent(torch.Tensor(a), len((64, 64, 3,))))
