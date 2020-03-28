import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

csv_files = glob.glob("rangefinder\\*.csv")

fig, ax = plt.subplots()

expected = []
X = []
for path in csv_files:
    try:
        df = pd.read_csv(path, header=None)
        exp = int(path[path.find("\\") + 1: path.find(".csv")]) * 10
        x = np.array(df[2]) - exp
        y = []
        for z in x:
            if z > -500:
                y.append(z)
        X.append(y)
        q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1])
        expected.append(str(exp))
    except:
        break

try: 
    ax.boxplot(X, showfliers=False, labels=expected)
except:
    pass
ax.set_ylabel("error (mm)")
ax.set_xlabel("expected (mm)")
ax.set_title("RPLidar boxplot")
plt.show()