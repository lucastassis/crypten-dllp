import torch
import crypten
from model_crypten import CryptenDLLP
from model_pytorch import MLPBatchAvg
crypten.init()

# x = torch.randn(10)
# x_enc = crypten.cryptensor(x)

# print(f"torch tensor: {x}")
# print(f"cryptensor: {x_enc}")
# print(f"plain cryptensor: {x_enc.get_plain_text()}")

# crypten_dllp = CryptenDLLP(10, 2)
# dllp = MLPBatchAvg(10, 2)

# print(f"torch dllp: {dllp(x)}")
# print(f"crypten_dllp: {crypten_dllp(x)[0]}, {crypten_dllp(x)[1]}")

loss_torch = torch.nn.BCELoss()
loss = crypten.nn.BCELoss() # Choose loss functions

y_real = torch.tensor([[.75, .25]])
y_pred = torch.tensor([[.25, .75]])

print(loss_torch(y_pred, y_real))

y_real = crypten.cryptensor(y_real)
y_pred = crypten.cryptensor(y_pred)

print(loss(y_real, y_pred).get_plain_text())

# print(loss_torch(y_pred, y_real))


