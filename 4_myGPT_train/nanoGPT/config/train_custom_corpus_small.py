# train a small model on our custom corpus
# designed for a quick test run on a GPU like RTX 3080 (10GB)

out_dir = 'out-custom-small'

dataset = 'custom_corpus'
meta_path = 'meta.json'

n_layer = 6
n_head = 8
n_embd = 512
vocab_size = 262144

# -- Optimizer configuration --
learning_rate = 6e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 6e-5
beta2 = 0.99

warmup_iters = 200

batch_size = 4
block_size = 512
gradient_accumulation_steps = 4

# -- Evaluation and logging --
eval_interval = 250
eval_iters = 200
log_interval = 10

# -- Checkpointing --
always_save_checkpoint = True

# -- Mixed precision and device --
device = 'cuda'
compile = False
