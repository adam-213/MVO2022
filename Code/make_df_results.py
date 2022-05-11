from constants import *
from cauchy import *
from basil import *
from backtracking import *
from constant_step import *


def make_df_results():
    # takes about forever to run
    const = grad_const_result_file()
    print("const_done")
    cauchy = cauchy_result_file()
    print("cauchy_done")
    basil = barbor_result_file()
    print("basil_done")
    back = back_result_file()
    print("back_done")

    # create the df that contains all the results
    df = pd.DataFrame(columns=["type", "xis", "start", "c", "alpha", "delta"])
    df["xis"] = df["xis"].astype(object)
    df["start"] = df["start"].astype(object)

    for c in const:
        d = {type: "const", 'xis': c[0], 'start': c[1], 'c': c[2], }
        df = df.append(d, ignore_index=True)
    print("const_done")
    for c in cauchy:
        d = {type: "cauchy", 'xis': c[0], 'start': c[1]}
        df = df.append(d, ignore_index=True)
    print("cauchy_done")
    for c in basil:
        d = {type: "basil", 'xis': c[0], 'start': c[1]}
        df = df.append(d, ignore_index=True)
    print("basil_done")
    for c in back:
        d = {type: "back", 'xis': c[0], 'start': c[1], 'c': c[2], 'alpha': c[3], 'delta': c[4]}
        df = df.append(d, ignore_index=True)
    print("back_done")
    df.to_csv(root.joinpath("Data/xis_hodnoty.csv"), index=False)
    print("csv_done")
    return df
