import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Seq(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        # self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, seq):
        # print(seq.view(len(seq), 1, -1), self.hidden, (seq.view(len(seq), 1, -1)).shape)
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        # print(self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]

class Encoder(nn.Module):
    def __init__(self, in_shape, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), bias=False)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64*(in_shape[0]**2), 128)
        self.fc2 = nn.Linear(128, latent_dims)

    def forward(self, in_img):
        x = F.relu(self.conv1(in_img))
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims, out_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 128)
        self.fc2 = nn.Linear(128, 64*(out_shape[0]**2))
        self.convT1 = nn.ConvTranspose2d(64, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 3, (3, 3))

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.convT1(z))
        z = F.relu(self.conv2(z))
        return z
