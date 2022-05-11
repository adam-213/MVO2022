from constants import *
from utils import *

if root.joinpath("Data/xis_hodnoty.csv").is_file():
    df = pd.read_csv(root.joinpath("Data/xis_hodnoty.csv"))
    # just importing because it was already created
else:
    from make_df_results import make_df_results

    df = make_df_results()

con = df.query("type == 'const'")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
ax1.set_xscale("log")
# #ax1.set_yscale("log")
ax2.set_xscale("log")
# #ax2.set_yscale("log")
ax3.set_xscale("log")
# ax3.set_yscale("log")
ax1.set_xlabel("Iteration // 100")
ax2.set_xlabel("Iteration // 100")
ax3.set_xlabel("Iteration // 100")
ax1.set_ylabel("Value of Beta0")
ax2.set_ylabel("Value of Beta1")
ax3.set_ylabel("Value of Beta2")

bss = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 7, -7], [0.1, 0.9, 1.1]]
bss = [tuple(i) for i in bss]
linestylemap = dict(zip(bss, ["-", "--", ":", "-.", "-"]))
cs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
colormap = dict(zip(cs, ["red", "orange", "yellow", "green", "blue", "purple", "lime", "brown", "pink"]))

for c in tqdm(range(len(con))):
    c = con.iloc[c]
    xis = listostr2arr(c["xis"])
    xs = tuple(range(len(xis)))
    # print(xs,xis)
    x1 = [max(-1000, min(i[0], 1000)) for i in xis]
    x2 = [max(-25, min(i[1], 25)) for i in xis]
    x3 = [max(-50, min(i[2], 50)) for i in xis]
    ax1.plot(xs, x1, label=str(c["c"]), linestyle=linestylemap[
        tuple(eval("[" + c["start"][1:-1].strip().replace(" ", ",").replace(",,", ",").strip(",") + "]"))],
             color=colormap[c["c"]])
    ax2.plot(xs, x2, label=str(c["c"]), linestyle=linestylemap[
        tuple(eval("[" + c["start"][1:-1].strip().replace(" ", ",").replace(",,", ",").strip(",") + "]"))],
             color=colormap[c["c"]])
    ax3.plot(xs, x3, label=str(c["c"]), linestyle=linestylemap[
        tuple(eval("[" + c["start"][1:-1].strip().replace(" ", ",").replace(",,", ",").strip(",") + "]"))],
             color=colormap[c["c"]])


ax1.axhline(y=-720.71, linestyle="dotted", color="black")
ax2.axhline(y=3.2055, linestyle="dotted", color="black")
ax3.axhline(y=11.4709, linestyle="dotted", color="black")
fig.tight_layout()
fig.suptitle("Metody konstantneho kroku")
fig.savefig(root.joinpath("images/konstantny_krok.png"))
fig.show()
