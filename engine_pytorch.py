import torch
import crypten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_pytorch import MLPBatchAvg
import random
np.random.seed(42)
torch.manual_seed(42)

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

def train_dllp(model, optimizer, n_epochs, loss_fc, X, bags, proportions):
    # model.train()    
    for i in range(n_epochs):
        for bag in np.unique(bags):
            bag_X = torch.from_numpy(X[bags == bag])
            bag_prop = torch.from_numpy(proportions[bag]).unsqueeze(0)
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

    # model hyperparams
    model = MLPBatchAvg(in_features=2, out_features=2, hidden_layer_sizes=(100,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = torch.nn.BCELoss()
    n_epochs = 100

    train_dllp(model, optimizer, n_epochs, loss_fc, X_train, bags_train, proportions)

    # eval model
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test)
        _, outputs = model(X_test)
        y_pred = outputs.argmax(dim=1).numpy()
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc * 100}%')
