import matplotlib.pyplot as plt
import pandas as pd

from backtracking import *
from cauchy import *
from basil import *
from constant_step import *
from NN_Training import *
from constants import *



constant_step = grad_const_file()
cauchy = cauchy_file()
basil = basil_file()
backtracking = backtracking_file()

data = pd.DataFrame(columns = ["type","result","iters","norms","beginning","stepsize","alpha","delta"])
data["result"] = data["result"].astype(object)
data["norms"] = data["norms"].astype(object)
data["beginning"] = data["beginning"].astype(object)

for c in constant_step:
    const = {"type":"const","result":c[0][0],"iters":c[0][1],"norms":c[0][2],"beginning":c[1],"stepsize":c[2]}
    data = data.append(const,ignore_index=True)

for c in cauchy:
    cauchy = {"type":"cauchy","result":c[0],"iters":c[1],"norms":c[2],"beginning":c[3]}
    data = data.append(cauchy,ignore_index=True)

for c in backtracking:
    back = {"type":"back","result":c[0],"iters":c[1],"norms":c[2],"beginning":c[3],"stepsize":c[4],"alpha":c[5],"delta":c[6]}
    data = data.append(back,ignore_index=True)

for c in basil:
    bas = {"type":"basil","result":c[0][0],"iters":c[0][1],"norms":c[0][2],"beginning":c[1]}
    data = data.append(bas,ignore_index=True)


#begining to tuple
data["beginning"].apply(lambda x: tuple(x))
#plot all norms for all methods
# plt.figure(figsize=(10,10))
# plt.title("Norms for all methods")
# plt.xlabel("Iterations")
# plt.ylabel("Norm")
# plt.yscale("log")
# plt.xscale("log")
# plt.grid(True)
#
# for idx, (norm,it) in enumerate(zip(data["norms"],data["iters"])):
#
#     if norm == [] or it == None:
#         continue
#     if idx == 98:
#         pass
#         plt.plot(range(it),norm,label="Basil")
#     elif idx <= 98:
#         plt.plot(range(it // 10000 + 1),tuple(norm),label=data["type"][idx])
# plt.legend()
# plt.show()

#plotly
fig = go.Figure()
for idx, (norm,it) in enumerate(zip(data["norms"],data["iters"])):
    if norm == [] or it == None:
        continue
    if idx == 98:
        pass
        fig.add_trace(go.Scatter(x=tuple(range(it)),y=norm,name="Basil"))
    elif idx <= 98:
        fig.add_trace(go.Scatter(x=tuple(range(it // 10000 + 1)),y=tuple(norm),name=data["type"][idx]))
fig.update_xaxes(title_text="iters", type="log", autorange=True)
fig.update_yaxes(title_text="norm", type="log", autorange=True)

fig.show()