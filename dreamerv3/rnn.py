import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# boiler plate rnn pytorch code from: https://www.kaggle.com/code/namanmanchanda/rnn-in-pytorch/notebook
x = torch.linspace(0, 799, 800)
y = torch.sin(x*2*3.1416/40)

plt.figure(figsize=(12, 4))
plt.xlim(-10, 801)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("sin")
plt.title("Sin Plot")
plt.plot(y.numpy())
# plt.show()

test_size = 40
train_set = y[:-test_size]
test_set = y[-test_size:]

plt.plot(train_set.numpy(), color='#8000ff')
plt.plot(range(760, 800), test_set.numpy())
# plt.show()

def input_data(seq, ws):
    out = []
    L = len(seq)

    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))

    return out

window_size = 40
train_data = input_data(train_set, window_size)
"""
print(len(train_data))
print(train_data[0])
print()
print(train_data[1])
"""

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=2, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        # self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, seq, hidden):
        # print(seq.view(len(seq), 1, -1), self.hidden, (seq.view(len(seq), 1, -1)).shape)
        # print(seq.unsqueeze(1).unsqueeze(1))
        # print(seq.view(len(seq), 1, -1))
        rnn_out, hidden = self.rnn(seq.unsqueeze(1).unsqueeze(1), hidden)
        # print(rnn_out)
        # print()
        # print()
        # print()
        # print(rnn_out.view(len(seq), -1))
        # print(rnn_out.squeeze())
        # print()
        # print()
        # print()
        # print(hidden)
        # print(self.hidden)
        # pred = self.linear(rnn_out.view(len(seq), -1))
        pred = self.linear(rnn_out.squeeze())
        return pred, hidden

torch.manual_seed(42)
model = RNN()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# print(model)

epochs = 10
future = 40

for i in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        pred = model(seq, recurrent:=torch.zeros(1, 1, model.hidden_size))
        y_pred = pred[0][-1]
        hiddens = pred[1]
        # print(len(hiddens))
        # print(len(pred[0]))
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    print(f"Epoch {i} Loss: {loss.item()}")

    preds = train_set[-window_size:].tolist()
    for f in range(future):
        seq = torch.FloatTensor(preds[-window_size:])
        with torch.no_grad():
            preds.append(model(seq, recurrent:=torch.zeros(1, 1, model.hidden_size))[0][-1].item())
    loss = criterion(torch.tensor(preds[-window_size:]), y[760:])
    print(f"Performance on test range: {loss}")

    plt.plot(y.numpy(), color='#8000ff')
    plt.plot(range(760, 800), preds[window_size:], color='#ff8000')
    plt.show()
