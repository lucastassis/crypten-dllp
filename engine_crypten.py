import torch
import crypten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_crypten import CryptenDLLP
import random
np.random.seed(42)
torch.manual_seed(42)
crypten.init()

def make_dataset():
    X, y = make_blobs(n_samples=100, n_features=2, centers=2)
    bags = np.random.randint(0, 5, size=100)
    
    # split data
    X_train, X_test, y_train, y_test, bags_train, bags_test = train_test_split(X, y, bags, test_size=0.2)

    # Creating the proportions
    proportions = np.zeros((5, 2))
    for i in range(5):
        bag_i = np.where(bags_train == i)[0]
        proportions[i][1] = y[bag_i].sum() / len(bag_i)
    proportions[:,0] = 1 - proportions[:,1]
    
    return X_train, X_test, y_train, y_test, bags_train, bags_test, proportions

def train_dllp(model, optimizer, n_epochs, loss_fc, data_dict):
    # model.train()    
    for i in range(n_epochs):
        for b, bag in data_dict.items():
            bag_X = bag[0]
            bag_prop = bag[1].unsqueeze(0)
            # compute outputs
            batch_avg, outputs = model(bag_X) 
            batch_avg = batch_avg.unsqueeze(0)
            # compute loss and backprop
            optimizer.zero_grad()
            assert batch_avg.shape == bag_prop.shape
            loss = loss_fc(batch_avg, bag_prop)
            loss.backward()
            optimizer.step()            
            
if __name__ == "__main__":
    # dataset
    X_train, X_test, y_train, y_test, bags_train, bags_test, proportions = make_dataset()

    data_dict = {
        "0" : (crypten.cryptensor(X_train[bags_train == 0]), crypten.cryptensor(proportions[0])),
        "1" : (crypten.cryptensor(X_train[bags_train == 1]), crypten.cryptensor(proportions[1])),
        "2" : (crypten.cryptensor(X_train[bags_train == 2]), crypten.cryptensor(proportions[2])),
        "3" : (crypten.cryptensor(X_train[bags_train == 3]), crypten.cryptensor(proportions[3])),
        "4" : (crypten.cryptensor(X_train[bags_train == 4]), crypten.cryptensor(proportions[4]))
    }

    # model hyperparams
    model = CryptenDLLP(in_features=2, out_features=2, hidden_layer_sizes=(100,))
    optimizer = crypten.optim.SGD(model.parameters(), lr=0.001)
    loss_fc = crypten.nn.BCELoss()
    n_epochs = 100
    model.encrypt()

    train_dllp(model, optimizer, n_epochs, loss_fc, data_dict)

    # eval model
    model.eval()
    with crypten.no_grad():
        X_test = crypten.cryptensor(X_test)
        _, outputs = model(X_test)
        y_pred = outputs.get_plain_text().argmax(dim=1)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc * 100}%')
