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

var_is_visc = 1
var_is_def = 0

# list of viscosity and melt rate values 
if (var_is_visc){
    x_variable <- c('1e1','5e1','1e2','5e2','1e3','5e3','1e4') # the values of the x variable to plot (viscosity)
} else {
    x_variable <- c('1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')  # the values of the x variable to plot (def rate)
}
melt_rate_list <- c('01','02','03','04','05','06','07','08','09')#,'1','2')
time_all <- c(40e6,60e6)


if (grepl("prod",base_path)){
    # prod zone, whole domain
    csv_file_name <- "py_branch_info.csv"
} else if (grepl("through",base_path)) {
    # through zone, only top
    csv_file_name <- "py_branch_info_top.csv"
} else {
    csv_file_name <- "py_branch_info.csv"
}

open_file <- function(file_to_open,norm_time,df_m,x,m,true_m_rate,time) {
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
        if (B_20 > 0){     # if there are fractures 
        ## B_21  'Intensity'
        B_21 <- df_bi_t$branches_tot_length
        ## B_C  'Characteristic length'
        B_C <- B_21/B_20
        ## B_22  'Dimensionless intensity'
        B_22 <- B_20 * (B_C)^2
        
        ## build dataframe
        de <- list(viscosity=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
        n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,time=time,norm_time=norm_time)

        # print("de")
        # print(de)
        df_m <- rbind(df_m,de)#,stringsAsFactors=FALSE)
        }
    }
    else {
        print(paste("no visc",x,", melt rate ",m," in ",file_to_open,sep=""))            
    }

    return(df_m)


}


# open file, extract values
build_branch_df <- function(x,m,time) {
        # to keep the amount of melt constant, we look at smaller times if melt rate is higher, later times if it's lower. This num will build the file name
        norm_time = round(time/1e6/as.double(m))*1e6  # from accurate number, round so that it opens a file that exists

        if (startsWith(m,'0'))
        {
            true_m_rate = as.double(m)/1000
        }
        else {
            true_m_rate = as.double(m)/100
        }
        if (var_is_visc) {
            ##  build the path. unlist(strsplit(x,"e"))[2] splits '5e3' into 5 and 3 and takes the second (3)
            file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',csv_file_name,sep="")
            # file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
            # file_to_open_bot <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
            file_to_open_bot <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',csv_file_name,sep="")
            # print(file_to_open)
        }else if (var_is_def) {
            file_to_open <- paste(base_path,'/thdef',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
        }
        if (file.exists(file_to_open)) {    
            df_m <- open_file(file_to_open,norm_time,df_m,x,m,true_m_rate,time)
        }
        else if (file.exists(file_to_open_bot)){
            df_m <- open_file(file_to_open_bot,norm_time,df_m,x,m,true_m_rate,time)
        }
        else {
            print(paste("file does not exist:",file_to_open))
        }
        return(df_m)
}

# initialise empty dataframe
df_m <- data.frame(viscosity=double(),melt_rate=factor(levels=melt_rate_list),
B_20=double(),B_21=double(),B_C=double(),B_22=double(),
n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
time=double(),norm_time=double())#,stringsAsFactors=FALSE)

df_average <- data.frame(viscosity=double(),melt_rate=factor(levels=melt_rate_list),
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
df_m[df_m$time == time_all[1],]
#  -------------------  ternary plots -------------------

# import the library to create ternary plots
# library("ggplot2")
library("ggtern")

if (FALSE) {
    png_name <- paste(base_path,"/branch_plots/br_ter_visc_time1e9.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    ptv <- ggtern(data=df_m,aes(x=n_XY,y=n_I,z=norm_time/1e9)) + geom_point(aes(color = viscosity)) + 
      scale_colour_continuous(trans='reverse')+ labs(
    x = expression('N'[Y+X]), 
    y = expression('N'[I]), 
    z = 'Time', 
    colour = "Viscosity")+
    guides(color = guide_legend(reverse=TRUE))
    
    # + 
    # scale_R_continuous(limits=c(0.0,0.5))#,breaks=seq(0,1,by=0.1)) 
    print(ptv)
    dev.off()
}

if (TRUE) {
    # colour by melt rate
    df_m$melt_rate_factor <- factor(df_m$melt_rate, ordered = TRUE)
    # pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(fill=as.factor(true_m_rate)),shape = 21,stroke=2,size=2,colour="black")+ 
    # pt1 <- ggtern(data=df_m[df_m$time == time_all[1],],aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=as.factor(true_m_rate)))+ 
    pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=as.factor(true_m_rate),alpha = as.factor(time)))+ 
    scale_colour_brewer(palette='Blues')+
    scale_fill_distiller(direction=+1)+
    scale_fill_discrete(guide = guide_legend(reverse=TRUE))+    
    labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Melt Rate")+
    guides(color = guide_legend(reverse=TRUE))+    # low at the bottom, high at the top
    theme(plot.background = element_rect(fill='transparent', color=NA),
        #panel.grid.major = element_line(linetype = "dotted",colour = "black"),
        legend.background = element_rect(fill='transparent'),
        panel.background = element_rect(fill = "#e6dbd5"),
        legend.key = element_rect(fill = "#e6dbd5"),
        legend.position = c(.85, .65))#,alpha=0.8))
    ggsave(paste(base_path,"/branch_plots/br_ter_melt_time.png",sep=''), pt1, bg='transparent')
}
