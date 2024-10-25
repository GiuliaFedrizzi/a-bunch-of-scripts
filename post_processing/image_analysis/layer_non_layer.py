import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


filename = "my_experiment09000.csv"
df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file

df = df[(df["x coord"]>0.02) & (df["x coord"]<0.98)]#["Broken Bonds"]

bb_df_1 = df[(df["y coord"]>0.0) & (df["y coord"]<0.1428) & (df["Broken Bonds"]>0) ]
bb_df_2 = df[(df["y coord"]>0.1428) & (df["y coord"]<0.2857)& (df["Broken Bonds"]>0) ]
bb_df_3 = df[(df["y coord"]>0.2857) & (df["y coord"]<0.428) & (df["Broken Bonds"]>0) ]
bb_df_4 = df[(df["y coord"]>0.428) & (df["y coord"]<0.57) & (df["Broken Bonds"]>0) ]
bb_df_5 = df[(df["y coord"]>0.57) & (df["y coord"]<0.714) & (df["Broken Bonds"]>0) ]
bb_df_6 = df[(df["y coord"]>0.714) & (df["y coord"]<0.857) & (df["Broken Bonds"]>0) ]
bb_df_7 = df[(df["y coord"]>0.857) & (df["y coord"]<0.98) & (df["Broken Bonds"]>0) ]

# bb_fast = 
def ratio(df):
    return df["Broken Bonds"].sum()/len(df)

print(f'len 1  {len(bb_df_1)}, sum: {bb_df_1["Broken Bonds"].sum()}, ratio {ratio(bb_df_1)}')
print(f'len 2  {len(bb_df_2)}, sum: {bb_df_2["Broken Bonds"].sum()}, ratio {bb_df_2["Broken Bonds"].sum()/len(bb_df_2)}')
print(f'len 3  {len(bb_df_3)}, sum: {bb_df_3["Broken Bonds"].sum()}, ratio {bb_df_3["Broken Bonds"].sum()/len(bb_df_3)}')
print(f'len 4  {len(bb_df_4)}, sum: {bb_df_4["Broken Bonds"].sum()}, ratio {bb_df_4["Broken Bonds"].sum()/len(bb_df_4)}')
print(f'len 5  {len(bb_df_5)}, sum: {bb_df_5["Broken Bonds"].sum()}, ratio {bb_df_5["Broken Bonds"].sum()/len(bb_df_5)}')
print(f'len 6  {len(bb_df_6)}, sum: {bb_df_6["Broken Bonds"].sum()}, ratio {bb_df_6["Broken Bonds"].sum()/len(bb_df_6)}')
print(f'len 7  {len(bb_df_7)}, sum: {bb_df_7["Broken Bonds"].sum()}, ratio {bb_df_7["Broken Bonds"].sum()/len(bb_df_7)}')

fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(8,6))
ax2 = ax1.twiny()
ax3 = ax1.twiny()

ax1.plot([ratio(bb_df_1),ratio(bb_df_2),ratio(bb_df_3),ratio(bb_df_4),ratio(bb_df_5),ratio(bb_df_6),ratio(bb_df_7)],[1,2,3,4,5,6,7],'o-',color='green')
ax2.plot([bb_df_1["Broken Bonds"].sum(),bb_df_2["Broken Bonds"].sum(),bb_df_3["Broken Bonds"].sum(),bb_df_4["Broken Bonds"].sum(),bb_df_5["Broken Bonds"].sum(),bb_df_6["Broken Bonds"].sum(),bb_df_7["Broken Bonds"].sum()],[1,2,3,4,5,6,7],'o-',color='blue')
ax3.plot([len(bb_df_1),len(bb_df_2),len(bb_df_3),len(bb_df_4),len(bb_df_5),len(bb_df_6),len(bb_df_7)],[1,2,3,4,5,6,7],'o-',color='orange')
plt.show()