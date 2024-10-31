import torch
import crypten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from plaintext import MLPBatchAvg, LRBatchAvg
from dllp import CryptenDLLP, CryptenLR
import multiprocessing
import os

multiprocessing.set_start_method('fork')


# dataset is the same for both torch and crypten
def load_parquet(data_path):
    data = pd.read_parquet(data_path)

    X = data.drop(columns=["y", "bag"]).to_numpy()
    y = data["y"].to_numpy()
    bags = data["bag"].to_numpy()

    # split data
    X_train, X_test, y_train, y_test, bags_train, bags_test = train_test_split(X, y, bags, test_size=0.2)

    # Creating the proportions
    proportions = np.zeros((10, 2))
    for i in range(len(np.unique(bags))):
        bag_i = np.where(bags_train == i)[0]
        proportions[i][1] = y_train[bag_i].sum() / len(bag_i)
    proportions[:,0] = 1 - proportions[:,1]

    # create datadict
    data_dict = {str(key) : (torch.tensor(X_train[bags_train == b]), torch.tensor(proportions[b])) for key, b in zip(range(len(np.unique(bags))), range(len(np.unique(bags))))}
    
    return X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict

def train(model, optimizer, n_epochs, loss_fc, data_dict):
    # model.train()
    for i in range(n_epochs):
        # print('epoch', i)
        for b, bag in data_dict.items():
            bag_X = bag[0]
            bag_prop = bag[1].unsqueeze(0)
            # compute outputs
            batch_avg, outputs = model(bag_X)
            batch_avg = batch_avg.unsqueeze(0)
            '''
            if type(batch_avg) == torch.Tensor:
                print(batch_avg)
            else:
                plain = batch_avg.get_plain_text()
                if crypten.comm.DistributedCommunicator.get().get_rank() == 0:
                    print(plain)
            '''
            # compute loss and backprop
            optimizer.zero_grad()
            assert batch_avg.shape == bag_prop.shape
            loss = loss_fc(batch_avg, bag_prop)
            loss.backward()
            optimizer.step()

def run_torch(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict):
    # model hyperparams
    model = MLPBatchAvg(in_features=X_train.shape[1], out_features=2, hidden_layer_sizes=(100,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = torch.nn.BCELoss()
    n_epochs = 100

    train(model, optimizer, n_epochs, loss_fc, data_dict)

    # eval model
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test)
        _, outputs = model(X_test)
        y_pred = outputs.argmax(dim=1).numpy()
    acc = accuracy_score(y_test, y_pred)
    print(f'mlp torch accuracy: {acc * 100}%')

def run_lr_torch(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict):
    # model hyperparams
    model = LRBatchAvg(in_features=X_train.shape[1], out_features=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = torch.nn.BCELoss()
    n_epochs = 100

    train(model, optimizer, n_epochs, loss_fc, data_dict)

    # eval model
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test)
        _, outputs = model(X_test)
        y_pred = outputs.argmax(dim=1).numpy()
    acc = accuracy_score(y_test, y_pred)
    print(f'lr torch accuracy: {acc * 100}%')

@crypten.mpc.run_multiprocess(world_size=3)
def run_crypten(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict):
    rank = crypten.comm.DistributedCommunicator.get().get_rank()

    X_train *= 2 ** 16
    proportions *= 2 ** 16

    data_dict_enc = {str(key) : (crypten.cryptensor(X_train[bags_train == b] / 2 ** 16), crypten.cryptensor(proportions[b] / 2 ** 16)) for key, b in zip(range(10), range(10))}
        
    model = CryptenDLLP(in_features=X_train.shape[1], out_features=2, hidden_layer_sizes=(100,))
    optimizer = crypten.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = crypten.nn.BCELoss()
    n_epochs = 100
    model.encrypt()
    
    train(model, optimizer, n_epochs, loss_fc, data_dict_enc)
    
    model.eval()
    with crypten.no_grad():
        X_test *= 2 ** 16
        X_test = crypten.cryptensor(X_test) / 2 ** 16
        _, outputs = model(X_test)
        y_pred = outputs.get_plain_text().argmax(dim=1)
    acc = accuracy_score(y_test, y_pred)
    print(f'crypten accuracy: {acc * 100}%')

@crypten.mpc.run_multiprocess(world_size=3)
def run_lr_crypten(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict):
    rank = crypten.comm.DistributedCommunicator.get().get_rank()

    X_train *= 2 ** 16
    proportions *= 2 ** 16

    data_dict_enc = {str(key) : (crypten.cryptensor(X_train[bags_train == b] / 2 ** 16), crypten.cryptensor(proportions[b] / 2 ** 16)) for key, b in zip(range(10), range(10))}
        
    model = CryptenLR(in_features=X_train.shape[1], out_features=2)
    optimizer = crypten.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = crypten.nn.BCELoss()
    n_epochs = 100
    model.encrypt()
    
    train(model, optimizer, n_epochs, loss_fc, data_dict_enc)
    
    model.eval()
    with crypten.no_grad():
        X_test *= 2 ** 16
        X_test = crypten.cryptensor(X_test) / 2 ** 16
        _, outputs = model(X_test)
        y_pred = outputs.get_plain_text().argmax(dim=1)
    acc = accuracy_score(y_test, y_pred)
    print(f'crypten accuracy: {acc * 100}%')

if __name__ == "__main__":
    # disable GPU if there is one
    # crypten needs a full GPU per party, so it can only use GPUs in distributed deployments, not local testing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    crypten.init()

    for data_path in ["dummy_dataset_easy.parquet", "dummy_dataset_harder.parquet"]:
        print(data_path)

        # make the same dataset for both torch and crypten
        X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict = load_parquet(data_path)

        # run_torch(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict)
        # run_lr_torch(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict)
        run_crypten(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict)
        run_lr_crypten(X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict)