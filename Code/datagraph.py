import pandas as pd

from backtracking import *
from cauchy import *
from basil import *
from constant_step import *
from NN_Training import *
from constants import *




def datagraph():
    full_x = []
    full_y = []
    full_z = []
    full_xx = []
    full_yy = []
    full_zz = []
    full_x.append(train.iloc[:,2].tolist())
    full_y.append(train.iloc[:,3].tolist())
    full_z.append(train.iloc[:,1].tolist())
    full_xx.append(test.iloc[:,2].tolist())
    full_yy.append(test.iloc[:,3].tolist())
    full_zz.append(test.iloc[:,1].tolist())

    #flatten lists
    full_x = [item for sublist in full_x for item in sublist]
    full_y = [item for sublist in full_y for item in sublist]
    full_z = [item for sublist in full_z for item in sublist]
    full_xx = [item for sublist in full_xx for item in sublist]
    full_yy = [item for sublist in full_yy for item in sublist]
    full_zz = [item for sublist in full_zz for item in sublist]
    fig = go.Figure(data=go.Scatter3d(x=full_x, y=full_y, z=full_z, mode='markers'))
    fig.add_trace(go.Scatter3d(x=full_xx, y=full_yy, z=full_zz, mode='markers', marker=dict(color='red')))
    fig.show()

datagraph()
