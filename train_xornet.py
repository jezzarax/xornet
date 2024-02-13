import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2) # Input to hidden layer
        self.fc2 = nn.Linear(2, 1) # Hidden to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float)
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
with torch.no_grad():
    predictions = model(X)
    predicted = predictions.round()
    print(f'Predictions: {predicted.view(-1)}')

save_model(model, "./models/xornet.safetensors")