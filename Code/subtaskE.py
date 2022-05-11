from basil import *


beta = [-720.71013816,    3.20553725,   11.47092466]
#according to basil method

y_hat = X @ beta # Predikcie pre tranovacie data
k = (y_hat - y)
u = k**2
mean_y = np.mean(y_hat)
l = (mean_y - y)**2
vys = 1 - (sum(u)/sum(l))

print(f"koecifient determinácie 1950-1980(MNS): {vys}\n")

test_x = test.iloc[:, [2, 3]].to_numpy()
y_t = test.iloc[:, 1].to_numpy()
X_t = np.hstack([np.ones([test.shape[0], 1]), test_x])
y_vys = X_t@beta
print(f"Predikovanie 1981-1983\nReálne dáta(dané): {list(y_t)}\nPredikovane cez priamku MNS: {list(y_vys)}")
print(f"Absolutný rozdiel hodnôt: {list(abs(y_t - y_vys))}")


#plot training data
fig = go.Figure()
#fig.add_trace(go.Scatter(x=train.iloc[:, 2], y=train.iloc[:, 1], mode='markers', name='Training data'))
#fig.add_trace(go.Scatter(x=test.iloc[:, 2], y=test.iloc[:, 1], mode='markers', name='Test data'))
#fig.add_trace(go.Scatter(x=test.iloc[:, 2], y=y_vys, mode='lines', name='Predicted data'))
fig.add_trace(go.Scatter(x=test.iloc[:, 2],y=test.iloc[:,2] * beta[1] + beta[0],name="Predicted"))
fig.add_trace(go.Scatter(x=test.iloc[:,2],y=test.iloc[:,1],name = "True"))
fig.update_layout(title='Training and test data')
#fig.write_image("training_test_data.png")
fig.show()