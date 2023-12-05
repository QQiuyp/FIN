epochs = 500
clamp = 2.0

# optimizer
lr = 1e-3
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-5

noise_flag = True

# input settings
message_weight = 100
stego_weight = 1
message_length = 64

# Train:
batch_size = 50
cropsize = 128

# Val:
batchsize_val = 16
cropsize_val = 128

# Data Path
TRAIN_PATH = '/root/Dataset/DIV2K/DIV2K_train_HR'
VAL_PATH = '/root/Dataset/test'

format_train = 'png'
format_val = 'png'

# Saving checkpoints:
MODEL_PATH = 'experiments/JPEG'
SAVE_freq = 5

suffix = ''
train_continue = False
diff = False






