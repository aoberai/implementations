import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BertTokenizerFast
import numpy as np
import os
import sys
import copy
# np.set_printoptions(threshold=sys.maxsize)
# torch.set_printoptions(profile="full")

"""
TODO:
test set
fixed inference
cross entropy loss, not mse 
"""


# config
embedding_dim = 540
num_head = 6
head_size = embedding_dim // num_head # concatenation of heads == embedding dim
block_ct = 6
batch_size = 512
context_length = 80 # max length example
lr = 1e-4

device = torch.device("cuda")

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        class Block(nn.Module):
            def __init__(self):
                super(Block, self).__init__()
                self.fcq = [nn.Linear(embedding_dim, head_size, bias=False, device=device) for i in range(num_head)]
                self.fck = [nn.Linear(embedding_dim, head_size, bias=False, device=device) for i in range(num_head)]
                self.fcv = [nn.Linear(embedding_dim, head_size, bias=False, device=device) for i in range(num_head)]

                self.ff1 = nn.Linear(embedding_dim, 6 * head_size)
                self.ff2 = nn.Linear(6 * head_size, embedding_dim)

                self.ln1 = nn.LayerNorm(embedding_dim)
                self.ln2 = nn.LayerNorm(embedding_dim)

            def forward(self, x):
                # self.to(device)
                attention_vals = []

                for j in range(num_head):
                    q, k, v = self.fcq[j](x), self.fck[j](x), self.fcv[j](x)  # (B, T, C -> head size) 
                    attention_score = nn.Softmax(dim=1)(torch.matmul(q, torch.transpose(k, 1, 2)) * head_size**-0.5) # (B, T, T)
                    # print(attention_score)
                    attention_val = torch.matmul(attention_score, v) # (B, T, C)
                    attention_vals.append(attention_val)

                combined_heads = torch.cat(attention_vals, dim=-1)

                x = combined_heads + x
                x = self.ln1(x)

                ff = self.ln2(self.ff2(F.relu(self.ff1(x))) + x) # feedforward part

                return ff

        self.embedder = nn.Embedding(len(tokenizer), embedding_dim)
        self.position_embedder = nn.Embedding(context_length, embedding_dim)
        self.blocks = [Block().to(device) for i in range(block_ct)]
        self.ffc = nn.Linear(context_length * embedding_dim, 1) # final fully connected
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x_batch):
        # embeddings
        # pad_mask = [None if word == 0 else [0] * embedding_dim for word in sent for sent in x_batch]
        embedding = self.embedder(x_batch) # (B, T, C) # TODO: turn all [PAD] to 0

        """
        # Turns [PAD] to 0 # TODO: this is too slow
        for i in range(len(x_batch)):
            for j in range(len(x_batch[0])):
                if x_batch[i][j] == 0:
                   embedding[i][j] = torch.Tensor([0] * embedding_dim).to(device)
        """

        pos_embedding = self.position_embedder(torch.arange(x_batch.shape[1], device=device))
        x = embedding + pos_embedding

        for i in range(block_ct):
            x = self.blocks[i].forward(x)

        x = self.flatten(x) # need to flatten token embedding dimension for whole sentence -> 1 polarity output
        polarity = self.ffc(x) # final fully connected

        return polarity

def save(amodel, aopt, model_path='attn_encoder.pt'):
    torch.save({
            'model_state_dict': amodel.state_dict(),
            'optimizer_state_dict': aopt.state_dict(),
            }, model_path)

def load(model_path='attn_encoder.pt'):
    try:
        checkpoint = torch.load(model_path)
        model = Transformer()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Found in checkpoint form w/ model and optimizer state')
            print('Previously trained model weights loaded...')
            return model, opt # TODO: return opt?
        except TypeError:
            print('Found in only static model state')
            return checkpoint
    except Exception as e:
        print(e, e.args)
        print('Model could not be loaded, does not exist')


# tokenization and ids

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
x_train, y_train = [], []
if "x.pt" not in os.listdir(".") or "y.pt" not in os.listdir("."):
    sst = load_dataset("sst2")
    x_batch_tmp, y_batch_tmp = [], []
    for i in range(1, len(sst["train"])):
        elem = sst["train"][i]
        sentence = elem["sentence"]
        sentence += "[PAD]" * (context_length - len(tokenizer.encode(sentence)))
        tokens = tokenizer.tokenize(sentence)
        encoded = tokenizer.encode(sentence)
        x_batch_tmp.append(encoded)
        y_batch_tmp.append([elem["label"]])
        # y_batch_tmp.append([1 if True in [True if word in sentence.split() else False for word in ["the", "and", "but", "or", "a", "is"]] else 0])
        print(sentence)
        print(y_batch_tmp[-1])
        decoded = tokenizer.decode(encoded)

        """
        print("Sentence:", sentence)
        print("Tokens:", tokens)
        print("Encoded:", encoded)
        """
        if (i % batch_size == 0 and i != 0):
            x_train.append(copy.copy(x_batch_tmp))
            y_train.append(copy.copy(y_batch_tmp))
            x_batch_tmp.clear()
            y_batch_tmp.clear()
        # if i > 1000:
        #     break

    x_train = torch.LongTensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    torch.save(x_train, "x.pt")
    torch.save(y_train, "y.pt")
else:
    x_train, y_train = torch.load("x.pt").to(device), torch.load("y.pt").to(device)

if "attn_encoder.pt" not in os.listdir("."):
    model = Transformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(28000):
        for batch_idx in range(len(x_train)):
            x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]

            # x_batch = torch.LongTensor(x_batch).to(device) # (B, T)
            # y_batch = torch.Tensor(y_batch).to(device)

            rand_order = torch.randperm(x_batch.size()[0])
            x_batch = x_batch[rand_order]
            y_batch = y_batch[rand_order]

            polarity = model.forward(x_batch)
            print(*[tokenizer.decode(x_batch[i]).replace("[PAD]", "") for i in range(-10, -1)], polarity[-10:-1], y_batch[-10:-1], "\n\n\n\n\n",  sep='\n')
            mae_loss = torch.nn.L1Loss()(polarity, y_batch).to(device)
            mse_loss = torch.nn.MSELoss()(polarity, y_batch).to(device)
            print("EPOCH:", epoch, "MAE:", mae_loss.item(), "MSE:", mse_loss.item())
            opt.zero_grad()
            mse_loss.backward()
            opt.step()

        save(model, opt)

else:
    print("Inference Time")
    model, opt = load("model.pt")
    model.to(device)
    while True:
        sentence = input("Sentence? : ")
        print()
        sentence += "[PAD]" * (context_length - len(tokenizer.encode(sentence)))
        print(sentence)
        print("Polarity:", model.forward(torch.LongTensor([tokenizer.encode(sentence)]).to(device)))
