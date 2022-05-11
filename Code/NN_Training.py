from constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

torch.manual_seed(420)

class Nonlinear(nn.Module):
    def __init__(self, activation=F.leaky_relu, input_dim=1, output_dim=1):
        super(Nonlinear, self).__init__()
        self.lin1 = nn.Linear(2,16)
        self.lin2 = nn.Linear(16,64)
        self.lin3 = nn.Linear(64,8)
        self.lin4 = nn.Linear(8,1)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.lin1(x),0.1)
        x = self.activation(self.lin2(x),0.2)
        x = self.activation(self.lin3(x),0.1)
        x = self.lin4(x)
        return x

    def get_weights(self):
        return self.lin1.weight, self.lin2.weight, self.lin3.weight

    def get_biases(self):
        return self.lin1.bias, self.lin2.bias, self.lin3.bias

class Mydata(torch.utils.data.Dataset):
    #scale data with standard scaler and return tensors of scaled data
    def __init__(self, data, labels, scaler=None):
        self.X = data
        self.y = labels
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
        else:
            self.scaler = scaler
        self.X = self.scaler.transform(self.X)
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_scaler(self):
        return self.scaler

def train():
    X = train.iloc[:,[2,3]].values
    y = train.iloc[:,1].values
    trainloader = Mydata(X, y)
    testloader = Mydata(test.iloc[:,[2,3]].values, test.iloc[:,1].values, trainloader.get_scaler())

    model = Nonlinear()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 8192
    wc = 0
    train_losses, test_losses = [], []
    for e in range(epochs):
        tot_train_loss = 0
        for X, y in trainloader:
            X = X.view(-1,2)
            y = y.view(-1,1)

            y_hat = model(X)
            loss = criterion(y_hat, y)
            tot_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            if e % 100 == 0:
                tot_test_loss = 0

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for X, y in testloader:
                        X = X.view(-1, 2)
                        y = y.view(-1, 1)
                        y_hat = model(X)
                        loss = criterion(y_hat, y)
                        tot_test_loss += loss.item()

                # Get mean loss to enable comparison between train and test sets
                train_loss = tot_train_loss / len(trainloader)
                test_loss = tot_test_loss / len(testloader)

                # At completion of epoch
                train_losses.append(train_loss)
                test_losses.append(test_loss)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss),
                      "Test Loss: {:.3f}.. ".format(test_loss))
                PATH = root.joinpath(f"Models/Checkpoints/FC64/check{e}.pth")
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                }, PATH)
    else:
        PATH = root.joinpath(f"Models/Checkpoints/FC64/check8192.pth")
        torch.save({
            'epoch': 8192,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
        }, PATH)

