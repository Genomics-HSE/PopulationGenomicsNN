import argparse
import importlib
import json
from datetime import datetime

import torch
import numpy as np

import utilities


parser = argparse.ArgumentParser()
parser.add_argument('-Ne', type=float, default=1.0, help='Ask VL')
parser.add_argument('-rho', type=float, default=1.6*10e-9, help='')
parser.add_argument('-mu', type=float, default=1.25*10e-8, help='')
parser.add_argument('-num_repl', type=int, default=int(1e5), help='')
parser.add_argument('-l', type=int, default=int(3e3), help='')
parser.add_argument('-ratio_train_examples', type=float, default=0.9, help='')
parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available()
                    else 'cpu', help='Specify use cpu/cuda')
parser.add_argument('-result_dir', default='results/',
                    help='')
parser.add_argument('-random_seed', default=np.random.randint(
    10000000000), help='')
parser.add_argument('-steps', type=int, default=0)
parser.add_argument('model')

args = parser.parse_args()

#NN = importlib.import_module(args.model)
import models.simplest as NN

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

traindata, test_data = utilities.make_dataset(args)
train_loader = torch.utils.data.DataLoader(traindata, batch_size=64,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data)

model = NN.model()
model = model.to(args.device)
model.double()  # Why?


criterion = NN.LOSS()
criterion = criterion.to(args.device)


optimizer = NN.optimizer(model)


train_step = NN.make_train_step(model, criterion, optimizer)
total_step = len(train_loader)


loss_per_step = []

for i in range(args.steps):
    total_loss = 0
    for i, (x_batch, y_batch) in enumerate(train_loader):
        loss = train_step(x_batch.to(args.device), y_batch.to(args.device))
        total_loss += loss
    loss_per_step.append(total_loss)


results = 0.0
if False:
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            results += abs(model(input) - target)


now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
output_info = {
    'date': now,
    'model': args.model,
    'total_loss_per_step': loss_per_step,
    'score': results,
    'parametrs':
    {
        'random_seed': args.random_seed,
        'train_steps': args.steps,
        'device': args.device,
        'events': None,
        'Ne': args.Ne,
        'rho': args.rho,
        'mu': args.mu,
        'num_repl': args.num_repl,
        'l': args.l,
        'ration_train_examples': args.ratio_train_examples
    }
}

filename = args.result_dir + str(now) + '.json'
with open(filename, 'w+') as outfile:
    json.dump(output_info, outfile)
