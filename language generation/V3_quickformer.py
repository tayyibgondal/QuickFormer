import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparameters
from config import (batch_size, block_size, max_iters,
                    eval_interval, learning_rate, device,
                    eval_iters, n_embd, n_layer, dropout)
sqrt_d = torch.sqrt(torch.tensor(n_embd)).int().item()
n_head = sqrt_d // 5
# ------------

# Function to log hyperparameters
def log_hyperparameters(log_file):
    log_file.write("\n")
    log_file.write("Hyperparameters:\n")
    log_file.write(f"batch_size = {batch_size}\n")
    log_file.write(f"block_size = {block_size}\n")
    log_file.write(f"max_iters = {max_iters}\n")
    log_file.write(f"eval_interval = {eval_interval}\n")
    log_file.write(f"learning_rate = {learning_rate}\n")
    log_file.write(f"device = {device}\n")
    log_file.write(f"eval_iters = {eval_iters}\n")
    log_file.write(f"n_embd = {n_embd}\n")
    log_file.write(f"n_head = {n_head}\n")
    log_file.write(f"n_layer = {n_layer}\n")
    log_file.write(f"dropout = {dropout}\n")
    log_file.write("\n")

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
import string
import re

def preprocess_text(text):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Convert to lowercase
    text = text.lower()

    return text

text = preprocess_text(text)

# here are all the unique characters that occur in this text
words = sorted(list(set(text.split(' '))))
vocab_size = len(words)

# create a mapping from words to integers
stoi = { wd:i for i,wd in enumerate(words) }
itos = { i:wd for i,wd in enumerate(words) }

# Lambda function to encode a list of words to their corresponding numbers
encode = lambda word_list: [stoi[word] for word in word_list]

# Lambda function to decode a list of numbers to their corresponding words
decode = lambda number_list: " ".join([itos[number] for number in number_list])

data = torch.tensor(encode(text.split(" ")), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Feebler(nn.Module):
    ''' 
    input: B, T, C
    output: B, T, sqrt(C)
    '''
    def __init__(self, sqrt_d):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(sqrt_d, sqrt_d, block_size)
        )
        self.sqrt_d = sqrt_d

    def forward(self, data):
        # Data is of shape (b, n, d)
        data_reshaped = data.view(batch_size, n_embd, block_size)  # set up data for feebler
        data_reshaped = data.view(batch_size, self.sqrt_d, self.sqrt_d, block_size)  # reshape incoming data
        product = data_reshaped * self.weights  # multiply data with weights
        # perform columnwise sum inside each window
        updated_product = torch.sum(product, dim=2, keepdim=False)  # finally we have converted from dxn to sqrt(d)xn
        return updated_product.view(batch_size, block_size, self.sqrt_d)
    

class Booster(nn.Module):
    ''' 
    input: B, T, sqrt(C)
    output: B, T, C
    '''
    def __init__(self, sqrt_d):
        super(Booster, self).__init__()
        self.weights = nn.Parameter(
            torch.randn(sqrt_d, sqrt_d, block_size)
        )
        self.sqrt_d = sqrt_d

    def forward(self, attention_output):
        # attention_output is of shape (batch, n, sqrt_d)
        # set up data shape for the booster
        attention_output = attention_output.view(batch_size, self.sqrt_d, block_size)
        attention_output_reshaped = attention_output.view(batch_size, 1, -1) # flatten all rows into one row
        attention_output_reshaped = attention_output_reshaped.repeat(1, self.sqrt_d, 1)  # repeat each row sqrt_d times
        attention_output_reshaped = attention_output_reshaped.view(batch_size, self.sqrt_d, self.sqrt_d, block_size)
        # multiply
        revived_output = self.weights * attention_output_reshaped
        revived_output = revived_output.view(-1, block_size)
        return revived_output.view(batch_size, block_size, n_embd)

class QuickHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(sqrt_d, head_size, bias=False)
        self.query = nn.Linear(sqrt_d, head_size, bias=False)
        self.value = nn.Linear(sqrt_d, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is of shape (batch_size, n, sqrt_d)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        collective_k = k.sum(1, keepdim=True)
        # Broadcast explicitly
        collective_k_bc = collective_k.repeat(1, block_size, 1)
        # q multiply k
        qk = q * collective_k_bc
        attention_weights = torch.softmax(qk, dim=1)
        collective_v = v.sum(dim=1, keepdim=True)
        collective_v_bc = collective_v.repeat(1, block_size, 1)
        output = collective_v_bc * attention_weights
        return output
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([QuickHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(sqrt_d, sqrt_d) # global variable sqrt_d
        self.dropout = nn.Dropout(dropout)  # global variable dropout

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, sqrt_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sqrt_d, 4 * sqrt_d),
            nn.ReLU(),
            nn.Linear(4 * sqrt_d, sqrt_d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = sqrt_d // n_head
        self.feebler = Feebler(sqrt_d)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(sqrt_d)
        self.ln1 = nn.LayerNorm(sqrt_d)
        self.ln2 = nn.LayerNorm(sqrt_d)
        self.booster = Booster(sqrt_d)

    def forward(self, x):
        x = self.feebler(x)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = self.booster(x)
        return x
    
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print('training started, gpt in making... :)')

start_time = time.time()
# training loop
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print('training completed successfully')
duration = end_time - start_time

# Log the time taken to a file
log_file_path = "quickformer.txt"
with open(log_file_path, "a") as log_file:
    log_file.write(f"Training duration: {duration:.2f} seconds\n")
    log_file.write(str(sum(p.numel() for p in m.parameters())/1e6) + ' M parameters' + "\n")
    log_file.write(str(sum(p.numel() for p in m.parameters())/1e9) + ' B parameters')
    # Log hyperparameters
    log_hyperparameters(log_file)
    # generate from the model
    context = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
    log_file.write(f"\n=========================================================\n")
    log_file.write(decode(m.generate(context, max_new_tokens=512)[0].tolist()))
    log_file.write(f"\n=========================================================\n")

print(f"Training duration: {duration:.2f} seconds")


