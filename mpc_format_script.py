import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from plaintext import MLPBatchAvg, LRBatchAvg
import pickle

def process_dict(data_dict, state_dict):
    train_dict = dict()
    test_dict = dict()
    for month, month_data in data_dict.items():
        if month in ["aug", "sep", "oct", "nov"]:
            for state, state_data in month_data.items():
                train_dict[f"{state}_{month}"] = (torch.tensor(state_data), torch.tensor(state_dict[state]))
        if month in ["dec"]:
            for state, state_data in month_data.items():
                test_dict[f"{state}_{month}"] = (torch.tensor(state_data), torch.tensor(state_dict[state]))
    
    return train_dict, test_dict
            

def train(model, optimizer, n_epochs, loss_fc, data_dict):
    model.train()
    for i in range(n_epochs):
        # print('epoch', i)
        for b, bag in data_dict.items():
            bag_X = bag[0]
            bag_prop = bag[1]
            # compute outputs
            batch_avg, outputs = model(bag_X)
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
        
if __name__ == "__main__":
    data_dict = pickle.load(open("dict_synth_data.pkl", "rb"))
    state_dict = pickle.load(open("election_props.pkl", "rb"))

    # process dicts
    train_dict, test_dict = process_dict(data_dict, state_dict)

    # define model
    model = MLPBatchAvg(in_features=1034, out_features=2, hidden_layer_sizes=(100,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = torch.nn.BCELoss()
    n_epochs = 100

    # train
    train(model, optimizer, n_epochs, loss_fc, train_dict)

    # eval model
    model.eval()
    with torch.no_grad():
        for bag, X_test in test_dict.items():
            batch_avg, outputs = model(X_test[0])
            print(f"pred result for {bag} = {batch_avg}")






        