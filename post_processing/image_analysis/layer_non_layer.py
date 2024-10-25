import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


filename = "my_experiment09000.csv"
df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file

df = df[(df["x coord"]>0.02) & (df["x coord"]<0.98)]#["Broken Bonds"]

bb_df_1 = df[(df["y coord"]>0.0) & (df["y coord"]<0.1428)]#["Broken Bonds"]
bb_df_2 = df[(df["y coord"]>0.1428) & (df["y coord"]<0.2857)]#["Broken Bonds"]
bb_df_3 = df[(df["y coord"]>0.2857) & (df["y coord"]<0.428)]#["Broken Bonds"]
bb_df_4 = df[(df["y coord"]>0.428) & (df["y coord"]<0.57)]#["Broken Bonds"]
bb_df_5 = df[(df["y coord"]>0.57) & (df["y coord"]<0.714)]#["Broken Bonds"]
bb_df_6 = df[(df["y coord"]>0.714) & (df["y coord"]<0.857)]#["Broken Bonds"]
bb_df_7 = df[(df["y coord"]>0.857) & (df["y coord"]<0.98)]#["Broken Bonds"]

sns.scatterplot(data=bb_df_1,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"

plt.show()
sns.scatterplot(data=bb_df_2,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
plt.show()
sns.scatterplot(data=bb_df_3,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
plt.show()
sns.scatterplot(data=bb_df_4,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
plt.show()
sns.scatterplot(data=bb_df_5,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
plt.show()
sns.scatterplot(data=bb_df_6,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
plt.show()
sns.scatterplot(data=bb_df_7,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
plt.show()