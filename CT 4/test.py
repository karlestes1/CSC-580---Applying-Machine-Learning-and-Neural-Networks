from contextlib import redirect_stdout
import subprocess
import progressbar
import itertools
import os
import numpy as np

if os.path.isfile('/tmp/fcnet-tox-21/results.csv'):
    os.remove('/tmp/fcnet-tox-21/results.csv')


hidden_units = [25,50,100,175,250]
learning_rates = [0.001, 0.0005, 0.0001]
epochs = [10, 20, 30, 40, 50]
dropout_probs = [0.25, 0.5, 0.75]

combos = itertools.product(hidden_units, learning_rates, epochs, dropout_probs)

for n_hidden, lr, n_epochs, dropout in progressbar.progressbar(list(combos), redirect_stdout=True):
    args = ['python', 'main.py', f'--n_hidden={n_hidden}', f'--lr={lr}', f'--n_epochs={n_epochs}', f'--keep_prob={dropout}', '--verbose=0', f'--name=units_{n_hidden}_lr_{lr}_epochs_{n_epochs}_dropout_{dropout}']

    subprocess.run(args)

# for i in progressbar.progressbar(range(20), redirect_stdout=True):
#     args = ['python', 'main.py', '--verbose=0', f'--name=standard_{i}']
#     subprocess.run(args)

# for i in progressbar.progressbar(range(20), redirect_stdout=True):
#     args = ['python', 'main.py', '--keep_prob=0.7', '--n_epochs=20', f'--n_hidden=170','--verbose=0', f'--name=optimized_{i}']
#     subprocess.run(args)