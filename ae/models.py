import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
Encoder: z_t ~ q_phi(z_t | h_t, x_t)
"""
# TODO: stride to prevent downsizing from convs
class Encoder(nn.Module):
    def __init__(self, in_shape, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(322624, 128)
        self.fc2 = nn.Linear(128, latent_dims)

    def forward(self, in_img):
        x = F.relu(self.conv1(in_img))
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        print("enc", x.shape)
        return x

"""
Decoder: x_hat_t ~ p_phi(x_hat_t | h_t, z_t)
"""
class Decoder(nn.Module):
    def __init__(self, latent_dims, out_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 128)
        self.fc2 = nn.Linear(128, 322624)
        self.unflat = nn.Unflatten(1, (64, 71, 71))
        self.convT1 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3))
        self.convT2 = nn.ConvTranspose2d(32, 3, kernel_size=(3, 3))

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.unflat(z)
        z = F.relu(self.convT1(z))
        z = F.relu(self.convT2(z))
        return z

