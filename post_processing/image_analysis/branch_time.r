# ------------------------------------------
# analyse py_branch_info.csv files
# specify name of subdirectories (viscosity/def rate + melt rate)
# plot parameters such as 'Frequency', 'Intensity', 'Dimensionless intensity'
# import and plot number and ratios of types of nodes: I, Y and X

# plot evolution with time

# run in the conda environment "r_env"
#   it contains tidyverse, patchwork
# specify time as command line argument e.g.
# > Rscript $MS/post_processing/image_analysis/branch_analysis.r 6e7 
#
# or submit a task array job from visc_*/vis*
#
# Giulia April 2023
# ------------------------------------------

## Utils
# libraries
library(tidyverse)
library(patchwork)
# first part of path
base_path <- getwd( )

args <- commandArgs(trailingOnly = TRUE)  # store them as vectors

# list of viscosity and melt rate values 
x_variable <- c('1e1','5e1','1e2','5e2','1e3','5e3','1e4')#,'5e3','1e4')#,'2e4','4e4')  # the values of the x variable to plot (e.g. def rate)
melt_rate_list <- c('01','02','03','04','06','08','09')#,'1','2')
time_all <- c(2e7,3e7,4e7)

open_file <- function(file_to_open,norm_time,df_m) {
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
        df_m <- rbind(df_m,de)#,stringsAsFactors=FALSE)
        # df_m <- rbind(df_m,de,stringsAsFactors=FALSE)
    }
    else {
        print(paste("no visc",x,", melt rate ",m," at time ",time,sep=""))            
    }

    return(df_m)


}


# open file, extract values
build_branch_df <- function(x,m,time) {
        # to keep the amount of melt constant, we look at smaller times if melt rate is higher, later times if it's lower. This num will build the file name
        norm_time = round(time/1e7/as.double(m))*1e7

        if (startsWith(m,'0'))
        {
            true_m_rate = as.double(m)/1000
        }
        else {
            true_m_rate = as.double(m)/100
        }
        ##  build the path. unlist(strsplit(x,"e"))[2] splits '5e3' into 5 and 3 and takes the second (3)
        file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/py_branch_info.csv',sep="")
        file_to_open_bot <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/py_branch_info_top.csv',sep="")
        # print(file_to_open)
        if (file.exists(file_to_open)) {    
            df_m <- open_file(file_to_open,norm_time,df_m)
        }
        else if (file.exists(file_to_open_bot)){
            df_m <- open_file(file_to_open_bot,norm_time,df_m)
        }
        else {
            print(paste("file does not exist:",file_to_open))
        }
        return(df_m)
}

# initialise empty dataframe
# df_m <- data.frame(viscosity=double(),melt_rate=character(),
# B_20=double(),B_21=double(),B_C=double(),B_22=double(),
# n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
# time=double(),norm_time=double(),stringsAsFactors=FALSE)

# melt_rate=factor(levels=melt_rate_list) I'm specifying the possible values (levels) for melt rate 
df_m <- data.frame(viscosity=double(),melt_rate=factor(levels=melt_rate_list),
B_20=double(),B_21=double(),B_C=double(),B_22=double(),
n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
time=double(),norm_time=double())#,stringsAsFactors=FALSE)

for (t in time_all) {
    for (x_var in x_variable) {
        for (melt in melt_rate_list) {
            df_m <- build_branch_df(x_var,melt,t)
        }
    } 
}

# branches/lines
df_m["n_B_n_L"] <- df_m$n_B/df_m$n_L
df_m["C_L"] <- 2*(df_m$n_Y+df_m$n_X)/df_m$n_L
df_m["n_XY"] <- df_m$n_Y+df_m$n_X

df_m


#  -------------------  ternary plots -------------------

# import the library to create ternary plots
library("ggplot2")
library("ggtern")

if (TRUE) {
    png_name <- paste(base_path,"/branch_plots/br_ter_visc_time.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    ptv <- ggtern(data=df_m,aes(x=n_XY,y=n_I,z=norm_time)) + geom_point(aes(color = viscosity)) + scale_colour_continuous(trans='reverse') #+ geom_path()
    print(ptv)
    dev.off()
}

