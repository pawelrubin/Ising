from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ising.txt", sep="\s+")

data_grouped = OrderedDict(
    (l, data[(data["l"] == l)].sort_values("t")) for l in [6, 15, 40, 70]
)
markers = ["-s", "-^", "-o", "-+"]
for (l, d), marker in zip(data_grouped.items(), markers):
    plt.plot(d["t"], d["m"], marker, label=f"L={l}", ms=5)

plt.legend()
plt.xlabel("T*")
plt.ylabel(r"$\langle m \rangle $")
plt.show()

for (l, d), marker in zip(list(data_grouped.items())[:-1], markers):
    plt.plot(d["t"], d["s"], marker, label=f"L={l}", ms=5)

plt.legend()
plt.xlabel("T*")
plt.ylabel(r"$\chi$")

plt.show()
