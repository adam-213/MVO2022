from constants import *
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np


def graf_funkcie():
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=train.iloc[:, 2], y=train.iloc[:, 3], z=train.iloc[:, 1], name='Training_Scatter'))
    x = np.linspace(train.iloc[:, 2].min(), test.iloc[:, 2].max(), 100)
    y = np.linspace(train.iloc[:, 3].min(), test.iloc[:, 3].max(), 100)
    xgrid, ygrid = np.meshgrid(x, y)
    beta = [-720.71013816, 3.20553725, 11.47092466]
    xs = beta[1] * x
    ys = beta[2] * y
    z = beta[0] + x * beta[1] + y * beta[2]

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name='Regression_Line', marker={"size": 20}))

    fig.add_trace(go.Scatter3d(x=test.iloc[:, 2], y=test.iloc[:, 3], z=test.iloc[:, 1], name='Test_Scatter'))

    fig.show()


graf_funkcie()
