import torch
import crypten

# batch average layer, not natively supported in crypten
class BatchAvgLayer(crypten.nn.Module):
    def __init__(self):
        super(BatchAvgLayer, self).__init__()

    def forward(self, x):
        # mean() can be found at crypten.common.functions.regular.mean(input=x, dim=0)
        return x.mean(dim=0)

# neural network for learning from label proportions
class CryptenDLLP(crypten.nn.Module):
    
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(CryptenDLLP, self).__init__()
        self.layers = crypten.nn.ModuleList()
        for size in hidden_layer_sizes:
            self.layers.append(crypten.nn.Linear(in_features, size))
            self.layers.append(crypten.nn.ReLU())
            in_features = size
        self.layers.append(crypten.nn.Linear(hidden_layer_sizes[-1], out_features))
        self.layers.append(crypten.nn.Softmax(dim=1))
        self.batch_avg = BatchAvgLayer()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        softmax = x.clone()
        x = self.batch_avg(x)
        return x, softmax
