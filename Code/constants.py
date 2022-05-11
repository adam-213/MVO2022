import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from pathlib import Path
import math
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from utils import *
pio.renderers.default = "iframe"




root = Path(__file__).parent.parent

train_data = 'https://github.com/adam-213/MVO2022/raw/main/Data/HDP.txt'
test_data = 'https://github.com/adam-213/MVO2022/raw/main/Data/HDP_test.txt'
for file in root.joinpath('Data').glob("*"):
    if file.suffix == '.zip':
        print("Unzipping")
        unzip()
        break

train = pd.read_csv(train_data)
test = pd.read_csv(test_data)

n = int(1e+7)
eps = 1e-2
c = 1e-7
generate = False

train_x = train.iloc[:,[2,3]].to_numpy() # x1,x2 -> np.array
X = np.hstack([np.ones([train.shape[0],1]),train_x]) # add column vector to the front of the matrix
b = np.array([1,1,1]) #default coefficients for vector of variables beta.
y = train.iloc[:,1].to_numpy() #actual data from real world



def f(b):
    """L2_norm^2"""
    return b.T@X.T@X@b -2*y.T@X@b + y.T@y

def df(b):
    """derivative of L2_norm^2"""
    return 2*(X.T@X@b) - 2*(X.T@y)

def ddf(b):
    return 2*(X.T@X)