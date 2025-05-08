# train a medium-sized model on our custom corpus for a long run
# Designed for RTX 3080 Ti (12GB) with good VRAM usage and MFU

out_dir = 'out-custom-long'

# -- Data configuration --
dataset = 'custom_corpus'
meta_path = 'meta.json'

# -- Model configuration --
n_layer = 6
n_head = 8
n_embd = 512
vocab_size = 262144

dropout = 0.1

# -- Optimizer configuration --
learning_rate = 6e-4
max_iters = 500000

lr_decay_iters = 500000
min_lr = 6e-5

beta1 = 0.9
beta2 = 0.95

warmup_iters = 200

# -- Batch size and sequence length --
batch_size = 4
block_size = 512
gradient_accumulation_steps = 4

# -- Evaluation and logging --
eval_interval = 1000
eval_iters = 200

log_interval = 100 

wandb_log = False
wandb_project = 'custom-corpus'
wandb_run_name = 'long-run-1'

# -- Checkpointing --
always_save_checkpoint = False

# -- Mixed precision and device --
device = 'cuda'
compile = False