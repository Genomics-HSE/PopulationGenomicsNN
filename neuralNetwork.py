# -*- coding: utf-8 -*-

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

import utilities


# TODO
# +) Splite main to functions;
# 2) Separete consts;
# 3) Use arguments;
# 4) Add db to store results;
# 5) Separate NN class, rename in main;
# 6) Separate LOSS;


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0]  # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def make_dataset():
    """Create data generator"""

    events = utilities.generate_demographic_events()

    Ne = 1
    rho = 1.6*10e-9*Ne
    mu = 1.25*10e-8*Ne
    num_repl = int(1e5)  # 5
    l = int(3e3)

    dg = utilities.DataGenerator(recombination_rate=rho,
                                 mutation_rate=mu,
                                 demographic_events=events,
                                 num_replicates=num_repl, lengt=l)
    dg.run_simulation()

    """Create datasets"""

    number_train_examples = int(num_repl*.9)

    trX, trY = [], []
    for _ in range(number_train_examples):
        example = next(dg)
        trX.append(example[0])
        trY.append(example[1])

    teX, teY = [], []
    for example in dg:
        teX.append(example[0])
        teY.append(example[1])

    del dg

    input = torch.from_numpy(np.array(trX, dtype=np.float_))  # .to(device)
    target = torch.from_numpy(np.array(trY))  # .to(device)
    test_input = torch.from_numpy(
        np.array(teX, dtype=np.float_))  # .to(device)
    test_target = torch.from_numpy(np.array(teY))  # .to(device)

    del trX, trY, teX, teY

    return MyDataset(input, target), test_input, test_target


def main():

    traindata, _, _ = make_dataset()

    loss_per_step = []

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=64,
                                               shuffle=True)

    # build the model
    seq = Sequence()
    seq = seq.to(device)
    seq.double()
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    #
    optimizer = optim.SGD(seq.parameters(), lr=.001)
    # begin to train

    train_step = make_train_step(seq, criterion, optimizer)
    total_step = len(train_loader)

    for i in range(15):
        print('STEP: ', i)
        total_loss = 0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            loss = train_step(x_batch.to(device), y_batch.to(device))
            total_loss += loss
            if i % 5 == 0:
                print(f'BATCH: {i+1}/{total_step} LOSS: {loss}')
        loss_per_step.append(total_loss)

    print(loss_per_step)


main()
