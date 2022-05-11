from NN_Training import *
from constants import *


def load():
    model = Nonlinear()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainloader = Mydata(train.iloc[:, [2, 3]].values, train.iloc[:, 1].values)
    scaler = trainloader.get_scaler()
    checkpoint = torch.load(root.joinpath('Models/Best/FC64/fc64.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Model loaded')
    model.eval()
    return model, scaler


def predict(test, model, scaler):
    testloader = Mydata(test.iloc[:, [2, 3]].values, test.iloc[:, [1]].values, scaler)
    y_hat = []
    with torch.no_grad():
        for X, y in testloader:
            y_hat.append(model.forward(X))

    return y_hat


model, scaler = load()
y_hat_test = predict(test, model, scaler)
y_hat_test = [y.item() for y in y_hat_test]
y_hat_train = predict(train, model, scaler)
y_hat_train = [y.item() for y in y_hat_train]

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=test.iloc[:, 2], y=test.iloc[:, 3], z=y_hat_test, name='NN_Predicted_Test_Scatter'))
fig.add_trace(go.Scatter3d(x=test.iloc[:, 2], y=test.iloc[:, 3], z=test.iloc[:, 1], name='Test_Scatter'))
fig.add_trace(go.Scatter3d(x=train.iloc[:, 2], y=train.iloc[:, 3], z=y_hat_train, name='NN_Predicted_Training_Scatter'))
fig.add_trace(go.Scatter3d(x=train.iloc[:, 2], y=train.iloc[:, 3], z=train.iloc[:, 1], name='Training_Scatter'))
x = np.linspace(train.iloc[:, 2].min(), test.iloc[:, 2].max(), 100)
y = np.linspace(train.iloc[:, 3].min(), test.iloc[:, 3].max(), 100)
xgrid, ygrid = np.meshgrid(x, y)
beta = [-720.71013816, 3.20553725, 11.47092466]
xs = beta[1] * x
ys = beta[2] * y
z = beta[0] + x * beta[1] + y * beta[2]

fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name='Linear_Regression_Line', marker={"size": 20}))
fig.show()

import numpy as np
yp = y_hat_test
yt = test.iloc[:, 1].values
corr_matrix = np.corrcoef(yt, yp)
corr = corr_matrix[0,1]
R_sq = corr**2

print('R_sq: ', R_sq)
#Hey that's not bad at all
