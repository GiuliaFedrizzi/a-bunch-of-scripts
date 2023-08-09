import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('xypoints.txt',header = None,names=["x","y"])

print(df)

df2 = pd.crosstab(df['y'], df['x'])#.div(len(df))
sns.heatmap(df2)
# plt.legend(loc=2, prop={'size': 6})
plt.axis('equal')
plt.show()