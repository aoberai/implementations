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
print(len(train_data))
print(train_data[0])
print()
print(train_data[1])

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = torch.zeros(1, 1, hidden_size)
        # self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, seq):
        rnn_out, self.hidden = self.rnn(seq.unsqueeze(1).unsqueeze(1), self.hidden)
        pred = self.linear(rnn_out.squeeze())
        return pred[-1]

torch.manual_seed(42)
model = RNN()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(model)

epochs = 10
future = 40

for i in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        # model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))
        model.hidden = torch.zeros(1, 1, model.hidden_size)
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    print(f"Epoch {i} Loss: {loss.item()}")
    
    # window_size = random.randint(20, 50)
    preds = train_set[-window_size:].tolist()
    for f in range(future):
        seq = torch.FloatTensor(preds[-window_size:])
        with torch.no_grad():
            # model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))
            model.hidden = torch.zeros(1, 1, model.hidden_size)
            preds.append(model(seq).item())

    loss = criterion(torch.tensor(preds[-window_size:]), y[760:])
    print(f"Performance on test range: {loss}")

    plt.plot(y.numpy(), color='#8000ff')
    plt.plot(range(760, 800), preds[window_size:], color='#ff8000')
    plt.show()
