import torch
import torch.nn as nn
import pickle
torch.set_default_dtype(torch.float64)

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

# train
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

def train(model, optimizer, n_epochs, loss_fc, data_dict, y_dict):
    model.train()
    for _ in range(n_epochs):
        for b, bag in data_dict.items():
            bag_X = bag[0]
            bag_y = y_dict[b]
            # compute outputs
            outputs = model(bag_X)
            # compute loss and backprop
            optimizer.zero_grad()
            loss = loss_fc(outputs, bag_y)
            loss.backward()
            optimizer.step()

# define data format
# data_dict = { "bag_0" : (torch.rand((10, 10)), torch.tensor([0.7, 0.3])),
#               "bag_1" : (torch.rand((10, 10)), torch.tensor([0.3, 0.7])) }

data_dict = pickle.load(open("mpc_format_dllp/dict_synth_data.pkl", "rb"))
state_dict = pickle.load(open("mpc_format_dllp/election_props.pkl", "rb"))

# process dicts
train_dict, test_dict = process_dict(data_dict, state_dict)

# initialize thresholds and y by majority
thresholds = { b : 0.5 for b in train_dict.keys() }
y_dict = { b : torch.ones(len(d[0]), dtype=torch.long) if d[1].argmax() == 1 else torch.zeros(len(d[0]), dtype=torch.long) for b, d in train_dict.items() }

# define hyperparams
MAX_ITER = 500

# start training
for i in range(MAX_ITER):
    # train model
    model = LogisticRegression(input_dim=train_dict[list(train_dict.keys())[0]][0].shape[-1], output_dim=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fc = torch.nn.CrossEntropyLoss()
    n_epochs = 500
    train(model, optimizer, n_epochs, loss_fc, train_dict, y_dict)

    # redefine thresholds and y
    model.eval()
    for b, b_data in train_dict.items():
        b_y = y_dict[b]
        ones = int(torch.round(b_data[1][-1] * len(b_data[0])))
        new_y = torch.full((len(b_y),), -1)
        scores = torch.softmax(model(b_data[0]), dim=1)[:,1]
        shuffle = scores.argsort(descending=True) # sorting
        new_y[shuffle[:ones]] = 1
        new_y[shuffle[ones:]] = 0
        thresholds[b] = float(scores[shuffle[ones - 1]])

    # print last model result
    if i == MAX_ITER - 1:
        for b, b_data in test_dict.items():
            # b_threshold = thresholds[b]
            scores = float(torch.mean(torch.softmax(model(b_data[0]), dim=1), dim=0)[-1])
            print(b, scores)



    




