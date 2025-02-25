import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# in the csv:   time  n_0  n_I  n_2  n_3  n_4  n_5  branches_tot_length  n_branches  CV_hor  CV_ver

def get_image_size(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width * height # area

B_22_scaled_list = []

numbers = [1,2]

for i in numbers:
# i = 1
    filename = "py_branch_info_"+str(i)+".csv"
    df = pd.read_csv(filename, header=0)
    scale_unit = get_image_size("long-orange_blue_"+str(i)+".png")
    print(f'scale: {scale_unit}')

    n_0 = df["n_0"].iloc[0]
    n_I = df["n_I"].iloc[0]
    n_2 = df["n_2"].iloc[0]
    n_3 = df["n_3"].iloc[0]
    n_4 = df["n_4"].iloc[0]
    n_5 = df["n_5"].iloc[0]
    branches_tot_length = df["branches_tot_length"].iloc[0]
    n_branches = df["n_branches"].iloc[0]
    CV_hor = df["CV_hor"].iloc[0]
    CV_ver = df["CV_ver"].iloc[0]

    # print(branch_data["n_branches"])
    n_B = 0.5 * (n_I +3*n_3 + 4*n_4+5*n_5)

    print(f'n_B {n_B}')

    av_length = branches_tot_length/n_branches

    B_22 = n_B * (av_length)**2

    B_22_scaled = B_22/scale_unit

    print(f'B_22 {B_22}, B_22_scaled {B_22_scaled}')

    B_22_scaled_list.append(B_22_scaled)

plt.scatter(numbers,B_22_scaled_list)
plt.show()


# n_B = 0.5*sum(df_bi_t$n_I+3*(df_bi_t$n_3)+4*(df_bi_t$n_4)+5*(df_bi_t$n_5))

    # # I,X,Y nodes
    # n_I <- df_bi_t$n_I
    # # n_Y <- df_bi_t$n_2+df_bi_t$n_3  # including connectivity=2 in Y nodes
    # n_Y <- df_bi_t$n_3   # NOT including connectivity=2 in Y nodes
    # n_X <- df_bi_t$n_4+df_bi_t$n_5

    # CV_hor <- df_bi_t$CV_hor
    # CV_ver <- df_bi_t$CV_ver

    # ## n of branches    n_B = 0.5 * (NI + 3NY + 4NX)
    # n_B <- 0.5*sum(df_bi_t$n_I+3*(df_bi_t$n_3)+4*(df_bi_t$n_4)+5*(df_bi_t$n_5))
    # n_L <- 0.5*(n_I+n_Y)   # n of lines

    # ## B_20 'Frequency' : number of branches (from the node types) / Area 
    # B_20 <- n_B / (scale_unit^2)
    # # if (B_20 > 0){     # if there are fractures                  
    # ## B_21  'Intensity': tot length / Area  (unit: m^-1)
    # B_21 <- df_bi_t$branches_tot_length / (scale_unit^2)
    # ## B_C  'Characteristic length'
    # if (B_20 > 0){     # if there are fractures                  
    #     B_C <- df_bi_t$branches_tot_length/n_B
    # } else {
    #     B_C <- 0
    # }
    # ## B_22  'Dimensionless intensity'
    # B_22 <- B_20 * (B_C)^2

