
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import time

obs_shape = (3, 64, 64,)
x_mu = np.zeros(obs_shape)
x_hat = np.ones(obs_shape)

print(torch.distributions.Independent(torch.distributions.Normal(torch.Tensor(x_mu), 1), len(obs_shape)).log_prob(torch.Tensor(x_hat)))
print(torch.sum(torch.distributions.Independent(torch.distributions.Normal(torch.Tensor(x_mu), 1), len(obs_shape)).base_dist.log_prob(torch.Tensor(x_hat))))

# print(np.sum((x_hat - x_mu)**2))

for i in np.arange(0.1, 10.0, 0.1):
    print(i)
    print(torch.sum(torch.distributions.Independent(torch.distributions.Normal(torch.Tensor(x_mu), 1), len(obs_shape)).base_dist.log_prob(torch.Tensor(x_hat * i))))
    time.sleep(1)
