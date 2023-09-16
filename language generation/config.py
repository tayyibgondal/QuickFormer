import torch

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-4  # for bigger model, keep learning rate bigger than 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 625 
n_layer = 16
dropout = 0.001

# For experiments, I'll get model comparisons by changing no. of parameters
# I'll compare: 
# 1) Training times
# 2) Perplexities
# 3) Inference times (Not implemented yet)

# Only implement quick attention keeping d dimension the same
# Only reduce d keeping old attention as the same

# EXPERIMENTS TILL NOW: (ON QUICKFORMER)
# Tested with 
# 2500(div 50 by 10: 5 heads), 2025(div 45 by 5: 9 heads),
# 1600(div 40 by 4: 10 heads), 1225(div 35 by 5: 7 heads),
# 900(div 30 by 6: 5 heads), 625(div 25 by 5: 5 heads)
# EXPERIMENTS TILL NOW: (ON OLD GPT)
# Tested with 
# 2500(div 50 by 10: 5 heads), 2025(div 45 by 5: 9 heads),
# 1600(div 40 by 4: 10 heads), 1225(div 35 by 5: 7 heads),
# 900(div 30 by 6: 5 heads), , 625(div 25 by 5: 5 heads)
# Noitce: training rate of 1e-4 drastically improved perplexity scores.