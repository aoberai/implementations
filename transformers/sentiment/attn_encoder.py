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

# tokenization and ids

# device = torch.device("cuda")
context_length = 56
sst = load_dataset("sst2")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
x_train, y_train = [], [] # encodings, polarity 
for i in range(len(sst["train"])):
    elem = sst["train"][i]
    sentence = elem["sentence"]
    sentence += "[PAD]" * (context_length - len(tokenizer.encode(sentence)))
    tokens = tokenizer.tokenize(sentence)
    encoded = tokenizer.encode(sentence)
    x_train.append(encoded)
    y_train.append(elem["label"])
    decoded = tokenizer.decode(encoded)

    """
    print("Sentence:", sentence)
    print("Tokens:", tokens)
    print("Encoded:", encoded)
    """

    if i >= 1000:
       break 

x_train = torch.LongTensor(x_train) # (B, T)
y_train = torch.Tensor(y_train)

# embeddings
embedding = nn.Embedding(len(tokenizer), embedding_dim)(x_train) # (B, T, C

print("\n")
print(x_train)
print(embedding)
print(embedding.shape)

class Block():
    def forward(self, x):
        attention_vals = []

        for j in range(num_head):
            fcq, fck, fcv = nn.Linear(embedding_dim, head_size, bias=False)(x), nn.Linear(embedding_dim, head_size, bias=False)(x), nn.Linear(embedding_dim, head_size, bias=False)(x)  # (B, T, C -> head size) 
            # print(fcq.shape, torch.transpose(fck, 1, 2).shape)

            attention_score = nn.Softmax(dim=1)(torch.matmul(fcq, torch.transpose(fck, 1, 2)) * head_size**-0.5) # (B, T, T)
            attention_val = torch.matmul(attention_score, fcv) # (B, T, C)
            attention_vals.append(attention_val)
            # print(attention_val.shape)

        combined_heads = torch.cat(attention_vals, dim=-1)
        ff1, ff2 = nn.Linear(num_head * head_size, head_size), nn.Linear(head_size, embedding_dim)

        ff = ff2(F.relu(ff1(combined_heads))) # feedforward part

        return ff

x = embedding
for i in range(block_ct):
    x = Block().forward(x)

x = torch.flatten(x, start_dim=1) # need to flatten token embedding dimension for whole sentence -> 1 polarity output
polarity = nn.Linear(context_length * embedding_dim, 1)(x) # final fully connected

print(polarity)
print(polarity.shape)
