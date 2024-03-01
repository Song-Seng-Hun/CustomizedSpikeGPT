########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import logging
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
from src.spikingjelly.clock_driven import functional
from src.binidx import MMapIndexedDataset
from accelerate import accelerator
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

### Step 1: set training data ##########################################################################

datafile_train = "enwik8_split/train.txt" # txt file or binidx file
datafile_valid = "enwik8_split/valid.txt"
datafile_test = "enwik8_split/test.txt"
datafile_encoding = 'utf-8'
# datafile_encoding = 'utf-16le'

### Step 2: set model size #############################################################################

ctx_len = 1024 

model_type = 'RWKV'

### Step 3: set batch size #############################################################################

# ===> batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in model.py
# For example, if your batch_size = 20, you can set B_GROUP_FORWARD = 4, B_GROUP_BACKWARD = 2
# If you see "CUDA out of memory", reduce it. Use GPU-Z to find the highest value for your VRAM.
batch_size = 12

### Step 4: set learning rate, training mini-epochs #######################################################

lr_init = 6e-4
lr_final = 1e-5
# the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
n_epoch = 1000
# 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_frequency = 10
epoch_save_path = 'weights'

epoch_length_fixed = 10000

########################################################################################################

import src.utils
src.utils.set_seed(42) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

grad_norm_clip = 1.0
warmup_tokens = 0

betas = (0.9, 0.99)
eps = 4e-9

num_workers = 0

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile_train)
train_dataset = Dataset(open(
    datafile_train, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

#train_dataset = Dataset(MMapIndexedDataset(datafile_train), ctx_len, epoch_length_fixed) #use it when you use binidx files

# valid_dataset = Dataset(open(
#     datafile_valid, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed) 

# test_dataset = Dataset(open(
#     datafile_test, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)
########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(GPTConfig()).to(device)

    # # load a trained model. remember to change random seed
#     m2 = torch.load('medium/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
#     model.load_state_dict(m2)
    valid_dataset = None
    test_dataset = None
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, tconf)

    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
