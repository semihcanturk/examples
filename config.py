import math

import jax.experimental.optimizers

import experiment_buddy
import os

initial_lr = .0001

decay_steps = 500000
num_hidden = 1024
decay_factor = .5

batch_size = 128
momentum_mass = 0.99
weight_norm = 0.00

num_epochs = 1000000

experiment_buddy.register(locals())

################################################################
# Derivative parameters
################################################################
learning_rate = jax.experimental.optimizers.inverse_time_decay(initial_lr, decay_steps, decay_factor, staircase=True)
eval_every = math.ceil(num_epochs / 1000)

# for remote
# user: buzatu
# os.environ["password"] =  "LZ46GyzX"
# ssh-keygen
# or if your server uses custom port no:
# ssh-copy-id "buzatu@hawk.ml.jku.at -p
# os.environ["port"] = "5792"
# os.environ["hostname"] = "buzatu@hawk.ml.jku.at"
os.environ["BUDDY_HOSTNAME"] = "root@159.69.11.199"
os.environ["BUDDY_PASSWORD"] = "0oKG1L2sz0lVqNe1xo82HQLGc3ah"


import os
where_am_i = os.getenv('WHEREAMI')
if where_am_i == 'remote':
    tensorboard = experiment_buddy.deploy(use_remote=False, sweep_yaml="")
else:
    tensorboard = experiment_buddy.deploy(use_remote=True, sweep_yaml="")
