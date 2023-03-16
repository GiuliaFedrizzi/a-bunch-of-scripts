# ------------------------------------------
# analyse py_branch_info.csv files
# specify name of subdirectories (viscosity/def rate + melt rate)
# plot parameters such as 'Frequency', 'Intensity', 'Dimensionless intensity'
# import and plot number and ratios of types of nodes: I, Y and X

# run in the conda environment "r_env"
#   it contains tidyverse, patchwork
# ------------------------------------------

## Utils
# libraries
library(tidyverse)
library(patchwork)
# first part of path
base_path <- "/nobackup/scgf/myExperiments/threeAreas/prod/p01/"

# list of viscosity and melt rate values 
x_variable <- c('1e1','1e2','5e2','1e3','5e3','1e4')#,'5e3','1e4')#,'2e4','4e4')  # the values of the x variable to plot (e.g. def rate)
melt_rate_list <- c('01','02','03','04','06','08','09','1','2')
time = 6e7   # normalised time  (e.g. 6e7=60e6==60th file in mr_01)

# open file, extract values
build_branch_df <- function(x,m,time) {
        # to keep the amount of melt constant, we look at smaller times if melt rate is higher, later times if it's lower. This num will build the file name
        norm_time = round(time/1e5/as.double(m))*1e5

        if (startsWith(m,'0'))
        {
            true_m_rate = as.double(m)/1000
        }
        else {
            true_m_rate = as.double(m)/100
        }
        ##  build the path. unlist(strsplit(x,"e"))[2] splits '5e3' into 5 and 3 and takes the second (3)
        file_to_open <- paste(base_path,'visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/py_branch_info.csv',sep="")
        # print(file_to_open)
        if (file.exists(file_to_open)) {    
            df_bi <- read.csv(file=file_to_open)
            #print(paste("m rate",m,", file ",file_num))
            df_bi_t <- df_bi[df_bi$time == norm_time,]  # get the line that corresponds to the time
            if (nrow(df_bi_t)>0){
                # I,X,Y nodes
                n_I <- df_bi_t$n_I
                n_Y <- df_bi_t$n_2+df_bi_t$n_3
                n_X <- df_bi_t$n_4+df_bi_t$n_4

                ## n of branches, n of lines
                n_B <- sum(df_bi_t$n_I+df_bi_t$n_2+df_bi_t$n_3+df_bi_t$n_4+df_bi_t$n_5)
                n_L <- 0.5*(n_I+n_Y)

                ## B_20 'Frequency' 
                B_20 <- sum(df_bi_t$n_I+df_bi_t$n_2+df_bi_t$n_3+df_bi_t$n_4+df_bi_t$n_5)
                ## B_21  'Intensity'
                B_21 <- df_bi_t$branches_tot_length
                ## B_C  'Characteristic length'
                B_C <- B_21/B_20
                ## B_22  'Dimensionless intensity'
                B_22 <- B_20 * (B_C)^2
                
                ## build dataframe
                de <- list(viscosity=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,time=time,norm_time=norm_time)
                df_m <- rbind(df_m,de)
            }
            else {
            print(paste("no visc",x,", melt rate ",m," at time ",time,sep=""))            
            }

            } 
        else {
            print(paste("file does not exist:",file_to_open))
        }
}

# initialise empty dataframe
df_m <- data.frame(viscosity=double(),melt_rate=double(),
B_20=double(),B_21=double(),B_C=double(),B_22=double(),
n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
time=double(),norm_time=double())



for (x in x_variable) {
    for (m in melt_rate_list) {
        build_branch_df(x,m,time)
    }
} 

print(df_m)