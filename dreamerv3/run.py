import gymnasium as gym
import torch
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models import Encoder, Decoder, DynamicsPredictor, SequencePredictor

# env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.make("LunarLander-v2", render_mode="rgb_array")
state, info = env.reset()
scene = cv2.resize(env.render(), (75, 75))
batch_size = 128

recurrent_dim, latent_dim, action_dim = (1,), (15,), (1,)
display_shape = (400, 400, 3)
scene_shape = (75, 75, 3)
device = torch.device("cuda")
enc, dec = Encoder(scene_shape, latent_dim[0]).to(device), Decoder(recurrent_dim[0], latent_dim[0], scene_shape).to(device)
sequence_mdl, dynamics_mdl = SequencePredictor(recurrent_dim[0], latent_dim[0], action_dim[0]).to(device), DynamicsPredictor(latent_dim[0], recurrent_dim[0]).to(device)
opt_enc, opt_dec = optim.AdamW(enc.parameters(), lr=1e-3, amsgrad=True), optim.AdamW(dec.parameters(), lr=1e-3, amsgrad=True)


class Element:
    def __init__(self, scene, state, action, nxt_scene, nxt_state, reward, done):
        self.scene = scene
        self.state = state
        self.action = action
        self.nxt_scene = nxt_scene
        self.nxt_state = nxt_state
        self.reward = reward
        self.done = done

    def __str__(self, key):
        return '{}, {}'.format(self.scene, self.state, self.action, self.nxt_scene, self.nxt_state, self.reward, self.done)

# Element-wise data: scene, state, action, nxt_scene, nxt_state, rew, term
class ReplayBuffer:
    # capacity due to memory-constraints
    def __init__(self, capacity=1000000):
        self.buffer = []
        self.capacity = capacity

    def add(self, val):
        while len(self.buffer) > self.capacity:
            del self.buffer[0]
        self.buffer.append(val)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def last(self):
        return self.buffer[-1]

    def get(self):
        return self.buffer

replay_buffer = ReplayBuffer()

def get_action(state):
    # random policy
    return env.action_space.sample()

# Taken from pytorch dqn page
def plot_durations(eps_returns, show_result=False):
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
            from IPython import display
    plt.ion()


    plt.figure(1)
    durations_t = torch.tensor(eps_returns, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Epochs')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


while True:
    for i in range(batch_size * 100): # TODO: also divisible by batch number
        # agent policy that uses the state and info
        action = get_action(state)
        nxt_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if done:
            state, info = env.reset()

        nxt_scene = cv2.resize(env.render(), (75, 75))
        replay_buffer.add(Element(scene, state, action, nxt_scene, nxt_state, reward, done))
        cv2.imshow("Win", cv2.resize(replay_buffer.last().nxt_scene, display_shape[:2]))
        cv2.waitKey(1)

        state = nxt_state
        scene = nxt_scene

    """
    https://arxiv.org/pdf/2301.04104.pdf

    Steps:

    Recurrent State Space Model

    Sequence Model: h_t = f_phi(h_t-1, z_t-1, a_t-1)
    Encoder: z_t ~ q_phi(z_t | h_t, x_t)
    Dynamics Predictor: z_hat_t ~ p_phi(z_hat_t | h_t)
    Reward & Continue Predictor: r_hat_t, c_hat_t ~ p_phi(r_hat_t & c_hat_t | h_t, z_t)
    Decoder: x_hat_t ~ p_phi(x_hat_t | h_t, z_t)

    z_t has to be stochastic -- must be sampleable from a categorical distribution for loss function

    L_pred(phi) = -ln(p_phi(x_t | z_t, h_t)) - ln(p_phi(r_t | z_t, h_t)) - ln(p_phi(c_t | z_t, h_t))
    L_dyn(phi) = max(1, KL[sg(q_phi(z_t | h_t, x_t)) || p_phi(z_t | h_t)])
    L_rep(phi) = max(1, KL[q_phi(z_t | h_t, x_t) || sg(p_phi(z_t | h_t))])
    """

    '''
    # from: # https://github.com/pytorch/examples/blob/e11e0796fc02cc2cd5b6ec2ad7cea21f77e25402/word_language_model/main.py#L155

    def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)

    def init_hidden(self, bsz):
        # weight = next(self.parameters()).data
        # return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        return Variable(torch.zeros(recurrent_dim)).zero_()
    '''

    # TODO: make batches
    loss_pred = 0
    batch_losses = []
    for epoch in range(epochs:=15):
        h_t = torch.zeros(recurrent_dim).to(device)
        # h_t = torch.zeros(recurrent_dim, requires_grad=False).to(device)
        for data, i in zip(replay_buffer.get(), range(len(replay_buffer.get()))):
            if data.done:
                h_t = torch.zeros(recurrent_dim).to(device)
                # h_t = torch.zeros(recurrent_dim, requires_grad=False).to(device)
                # h_t = repackage_hidden(torch.zeros(recurrent_dim).to(device))
                # h_t = torch.autograd.Variable(h_t.data) # https://github.com/pytorch/examples/blob/e11e0796fc02cc2cd5b6ec2ad7cea21f77e25402/word_language_model/main.py#L155
            if i % batch_size == 0 and i != 0:
                print("Backpropagating @", i)
                opt_enc.zero_grad()
                opt_dec.zero_grad()
                loss_pred.backward(retain_graph=True) # TODO: why do I need this, I think this is wrong
                opt_enc.step()
                opt_dec.step()
                batch_losses.append(loss_pred.item())
                loss_pred = 0
                #plot_durations(batch_losses)
                print("Batch loss", batch_losses[-1])

            x_t = torch.Tensor(data.scene/255.).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            # a_t = torch.Tensor([data.action]).to(device).unsqueeze(1).unsqueeze(1)
            a_t = torch.Tensor([data.action]).to(device)
            # a_t = a_t.squeeze()
            # z_t = enc(x_t).sample().unsqueeze(2)
            # z_t = enc(x_t).sample().squeeze()
            z_t = enc(x_t).mean.squeeze()
            h_t_nxt = sequence_mdl(h_t, z_t, a_t)
            # print(h_t, h_t_nxt, h_t.shape, h_t_nxt.shape, z_t, z_t.shape)
            h_t = h_t_nxt
            x_t_hat = dec(h_t, z_t)
            loss_pred += -x_t_hat.log_prob(x_t).sum()

            cv2.imshow("Original", cv2.resize(np.moveaxis(x_t[0].cpu().detach().numpy(), 0, 2), display_shape[:2]))
            cv2.imshow("Reconstructed", cv2.resize((255 * np.moveaxis(torch.clip(x_t_hat.mean, 0, 1)[0].cpu().detach().numpy(), 0, 2)).astype("uint8"), display_shape[:2]))
            print(z_t)
            cv2.waitKey(1)
