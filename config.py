epochs = 500
clamp = 2.0

# optimizer
lr = 1e-4
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-5

# input settings
message_weight = 10
stego_weight = 1
message_length = 64

# Train:
batch_size = 50
cropsize = 128

# Val:
batchsize_val = 16
cropsize_val = 128

# Data Path
TRAIN_PATH = 'Dataset/'
VAL_PATH = 'Dataset/'

format_train = 'png'
format_val = 'png'

# Saving checkpoints:
MODEL_PATH = 'experiments/JPEG'
SAVE_freq = 1

suffix = ''
train_continue = True






