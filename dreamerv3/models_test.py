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
        print("dec", z.shape)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.unflat(z)
        z = F.relu(self.convT1(z))
        z = F.relu(self.convT2(z))
        return z

"""
Sequence Model: h_t = f_phi(h_t-1, z_t-1, a_t-1)
"""
class SequencePredictor(nn.Module):
    def __init__(self, recurrent_dim, latent_dim, action_dim):
        super(SequencePredictor, self).__init__()
        self.gru_cell = nn.GRUCell(latent_dim + action_dim, recurrent_dim)

    def forward(self, prev_recurrent, prev_latent, prev_action):
        # inp = torch.cat((prev_latent, prev_action), 1)
        inp = torch.cat((prev_latent, prev_action), 0)
        # inp = inp.squeeze()
        # print(inp.shape, prev_recurrent.shape, prev_latent.shape)
        curr_recurrent = self.gru_cell(inp, prev_recurrent)
        return curr_recurrent

"""
Dynamics Predictor: z_hat_t ~ p_phi(z_hat_t | h_t)
"""
class DynamicsPredictor(nn.Module):
    def __init__(self, latent_dim, recurrent_dim):
        super(DynamicsPredictor, self).__init__()
        self.fc1 = nn.Linear(recurrent_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, recurrent):
        x = F.relu(self.fc1(recurrent))
        x = F.relu(self.fc2(x))
        return torch.distributions.Normal(x, 1)

'''
"""
Reward & Continue Predictor: r_hat_t, c_hat_t ~ p_phi(r_hat_t & c_hat_t | h_t, z_t)
"""
class RewConPredictor(nn.Module):
    def __init__(self, input_size=1, recurrent_size=50, out_size=1):
        super().__init__()
        # GRUUUUUUU Cell
        self.rnn = nn.RNN(input_size, recurrent_size)
        self.recurrent_size = hidden_size
        self.linear = nn.Linear(recurrent_size, out_size)
        self.recurrent = torch.zeros(1, 1, hidden_size)

    def forward(self, seq):
        rnn_out, self.recurrent = self.rnn(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(rnn_out.view(len(seq), -1))

        return pred[-1]
'''
