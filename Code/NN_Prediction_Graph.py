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


def predict(testset, model, scaler):
    if type(testset) == str and testset == "test":
        testloader = Mydata(test.iloc[:, [2, 3]].values, test.iloc[:, [1]].values, scaler)
    elif type(testset) == str and testset == "train":
        testloader = Mydata(train.iloc[:, [2, 3]].values, train.iloc[:, [1]].values, scaler)
    else:
        testloader = Mydata(testset,np.array([1] * testset.shape[0]), scaler)
    y_hat = []
    with torch.no_grad():
        for X, y in testloader:
            y_hat.append(model.forward(X))

    return [y.item() for y in y_hat]


model, scaler = load()
y_hat_test = predict("test", model, scaler)
y_hat_test = [y for y in y_hat_test]
y_hat_train = predict("train", model, scaler)
y_hat_train = [y for y in y_hat_train]

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

preticted = []
import random
x = [x + random.random()*random.randint(1,10) for x in x]
y = [y + random.random()*random.randint(1,10) for y in y]
preticted.append(predict(np.array((x,y)).T, model, scaler))

fig.add_trace(go.Scatter3d(x=x, y=y, z=preticted[0], mode="lines", name='NN_Predicted_Line', marker={"size": 20}))

fig.show()

y_lin_train = beta[0] + beta[1] * train.iloc[:, 2] + beta[2] * train.iloc[:, 3]
y_lin_test = beta[0] + beta[1] * test.iloc[:, 2] + beta[2] * test.iloc[:, 3]

from sklearn.metrics import r2_score

print('R2_Score_Train_Linear: ', r2_score(train.iloc[:, 1], y_lin_train))
print("R2_score_Test_Linear: ", r2_score(test.iloc[:, 1], y_lin_test))
print("R2_Score_Train_NN: ", r2_score(train.iloc[:, 1], y_hat_train))
print("R2_score_Test_NN: ", r2_score(test.iloc[:, 1], y_hat_test))