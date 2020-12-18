import math
import jax.experimental.optimizers
import experiment_buddy
import os
from fabric.config import  Config
# from fabric import Config
from fabric.connection import contextmanager
# contextmanager.

initial_lr = .0001

decay_steps = 500000
num_hidden = 1024
decay_factor = .5

batch_size = 128
momentum_mass = 0.99
weight_norm = 0.00

num_epochs = 10

experiment_buddy.register(locals())

################################################################
# Derivative parameters
################################################################
learning_rate = jax.experimental.optimizers.inverse_time_decay(initial_lr, decay_steps, decay_factor, staircase=True)
eval_every = math.ceil(num_epochs / 1000)

# for remote
# user: buzatu
os.environ["password"] =  "LZ46GyzX"
# ssh-keygen
# or if your server uses custom port no:
# ssh-copy-id "buzatu@hawk.ml.jku.at -p
os.environ["port"] = "5792"
os.environ["hostname"] = "buzatu@hawk.ml.jku.at"
# os.environ["MY_BUDDY_HOSTNAME"] = "root@159.69.11.199"
# os.environ["MY_BUDDY_PASSWORD"] = "0oKG1L2sz0lVqNe1xo82HQLGc3ah"

tensorboard = experiment_buddy.deploy(use_remote=True, sweep_yaml="")
