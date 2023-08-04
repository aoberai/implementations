import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BertTokenizerFast

# config
embedding_dim = 512
head_size = 32
num_head = 6
block_ct = 6
batch_size = 128

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        class Block(nn.Module):
            def __init__(self):
                super(Block, self).__init__()
                self.fcq = [nn.Linear(embedding_dim, head_size, bias=False) for i in range(num_head)]
                self.fck = [nn.Linear(embedding_dim, head_size, bias=False) for i in range(num_head)]
                self.fcv = [nn.Linear(embedding_dim, head_size, bias=False) for i in range(num_head)]

                self.ff1 = nn.Linear(num_head * head_size, head_size)
                self.ff2 = nn.Linear(head_size, embedding_dim)

            def forward(self, x):
                attention_vals = []

                for j in range(num_head):
                    q, k, v = self.fcq[j](x), self.fck[j](x), self.fcv[j](x)  # (B, T, C -> head size) 
                    # print(fcq.shape, torch.transpose(fck, 1, 2).shape)
                    attention_score = nn.Softmax(dim=1)(torch.matmul(q, torch.transpose(k, 1, 2)) * head_size**-0.5) # (B, T, T)
                    attention_val = torch.matmul(attention_score, v) # (B, T, C)
                    attention_vals.append(attention_val)
                    # print(attention_val.shape)

                combined_heads = torch.cat(attention_vals, dim=-1)

                ff = self.ff2(F.relu(self.ff1(combined_heads))) # feedforward part

                return ff

        self.embedder = nn.Embedding(len(tokenizer), embedding_dim)
        self.blocks = [Block() for i in range(block_ct)]
        self.ffc = nn.Linear(context_length * embedding_dim, 1) # final fully connected

    def forward(self, x_batch, y_batch):
        # embeddings
        embedding = self.embedder(x_batch) # (B, T, C)

        # print(embedding)
        # print(embedding.shape)
        x = embedding
        for i in range(block_ct):
            x = self.blocks[i].forward(x)

        x = torch.flatten(x, start_dim=1) # need to flatten token embedding dimension for whole sentence -> 1 polarity output
        polarity = self.ffc(x) # final fully connected

        # print(polarity)
        # print(polarity.shape)
        # print(polarity.shape, y_batch.shape)

        return polarity

# tokenization and ids

device = torch.device("cuda")
context_length = 56
sst = load_dataset("sst2")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
x_batch, y_batch = [], [] # encodings, polarity  # TODO: convert to batches
for i in range(len(sst["train"])):
    elem = sst["train"][i]
    sentence = elem["sentence"]
    sentence += "[PAD]" * (context_length - len(tokenizer.encode(sentence)))
    tokens = tokenizer.tokenize(sentence)
    encoded = tokenizer.encode(sentence)
    x_batch.append(encoded)
    y_batch.append([elem["label"]])
    decoded = tokenizer.decode(encoded)

    """
    print("Sentence:", sentence)
    print("Tokens:", tokens)
    print("Encoded:", encoded)
    """

    if i >= batch_size: # batch_size
       break 

x_batch = torch.LongTensor(x_batch) # (B, T)
y_batch = torch.Tensor(y_batch)
# print("\n")
# print(x_batch)
model = Transformer()

for epoch in range(32): # not really an epoch 
    polarity = model.forward(x_batch, y_batch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss = torch.nn.MSELoss()(polarity, y_batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss)

