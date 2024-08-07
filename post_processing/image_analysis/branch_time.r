# ------------------------------------------
# analyse py_branch_info.csv files
# specify name of subdirectories (viscosity/def rate + melt rate)
# plot parameters such as 'Frequency', 'Intensity', 'Dimensionless intensity'
# import and plot number and ratios of types of nodes: I, Y and X

# plot evolution with time

# run in the conda environment "r_env"
#   it contains tidyverse, patchwork etc
#
# Giulia April 2023
# ------------------------------------------

## Utils
# libraries
library(tidyverse)
library(patchwork)

source("/home/home01/scgf/myscripts/post_processing/image_analysis/useful_functions.R")

# first part of path
base_path <- getwd( )

args <- commandArgs(trailingOnly = TRUE)  # store them as vectors

var_is_visc = 1
var_is_def = 0

# list of viscosity and melt rate values 
if (var_is_visc){
    x_variable <- find_dirs('visc')  # the values of the x variable to plot (viscosity)
} else {
    x_variable <- c('1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')  # the values of the x variable to plot (def rate)
}
# melt_rate_list <- c('01','02','03','04','05','06','07','08','09')#,'1','2')
# melt_rate_list <- c('02','04','06','08')#,'1','2')
melt_rate_list <- c('03','08') # the values of the y variable to plot (melt rate)
# melt_rate_list <- find_dirs('melt_rate')  # the values of the y variable to plot (melt rate)
# print(paste("melt_rate_list",melt_rate_list))

#  list of times: first 4 values are for mu 1e3 mr 08, last 4 values are for mu1.5, mr03  
time_all <- c(8000e6,24000e6,56000e6,128000e6,90000e6,150000e6,240000e6,300000e6)#,60e6,70e6,80e6,90e6)


if (grepl("prod",base_path)){
    # prod zone, whole domain
    csv_file_name <- "py_branch_info_x.csv"
} else if (grepl("through",base_path)) {
    # through zone, only top
    csv_file_name <- "py_branch_info_top.csv"
} else {
    csv_file_name <- "py_branch_info.csv"
}

open_file <- function(file_to_open,norm_time,df_m,x,m,true_m_rate,time,full_exponent) {
    df_bi <- read.csv(file=file_to_open)
    #print(paste("m rate",m,", file ",file_num))
    df_bi_t <- df_bi[df_bi$time == norm_time,]  # get the line that corresponds to the time
    if (nrow(df_bi_t)>0){
        # I,X,Y nodes
        n_I <- df_bi_t$n_I
        # n_Y <- df_bi_t$n_2+df_bi_t$n_3  # including connectivity=2 in Y nodes
        n_Y <- df_bi_t$n_3
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
        if (var_is_visc) {
            if (nchar(x) == 4) {  # 1e15 -> 15 -> 1.5 -> 10^(1.5)
                visc_value= 10^(as.double(full_exponent)/10)

            } else {
                visc_value= as.double(x)
            }
            de <- list(viscosity=x,true_viscosity=visc_value,melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,time=time,norm_time=norm_time)  
            
            } else if (var_is_def) {
                x <- gsub("e+","e-", x)
                de <- list(def_rate=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,time=time,norm_time=norm_time) 
            }

        df_m <- rbind(df_m,de)#,stringsAsFactors=FALSE)
        }
    }
    else {
        print(paste("no visc ",x,", melt rate ",m," in ",file_to_open,", at norm time ",norm_time,sep=""))            
    }

    return(df_m)


}


# open file, extract values
build_branch_df <- function(x,m,time) {
        var_is_visc <- 1
        var_is_def <- 0
        # to keep the amount of melt constant, we look at smaller times if melt rate is higher, later times if it's lower. This num will build the file name
        # norm_time = round(time/1e6/as.double(m))*1e6  # from accurate number, round so that it opens a file that exists
        norm_time = round(time/1e9/as.double(m))*1e9  # from accurate number, round so that it opens a file that exists

        if (startsWith(m,'0'))
        {
            true_m_rate = as.double(m)/1000
        }
        else {
            true_m_rate = as.double(m)/100
        }
        if (var_is_visc) {
            ##  build the path. unlist(strsplit(x,"e"))[2] splits '5e3' into 5 and 3 and takes the second (3)
            # file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',csv_file_name,sep="")
            full_exponent <- unlist(strsplit(x,"e"))[2]  # could be 1 or 15, because there is no point (1.5)
            true_exponent <- unlist(strsplit(full_exponent,""))[1]  # takes the first character (1 in 1.5, 2 in 2.5 etc)
            potential_file_path <- paste(base_path,'/visc_',true_exponent,'_',x,'/vis',x,'_mR_',m,'/',sep="")
            # potential_file_path <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',sep="")
            if (dir.exists(potential_file_path)) {
            }else {   # try a different version
                potential_file_path <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis1e2_mR_',m,'/',sep="")
                if (dir.exists(potential_file_path)) {
                }else{
                    print("I've tried the path twice without success")
                }
            }
            # file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
            file_to_open <- paste(potential_file_path,csv_file_name,sep="")
            # print(file_to_open)
        }else if (var_is_def) {
            file_to_open <- paste(base_path,'/thdef',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
            # file_to_open_bot <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',csv_file_name,sep="")
        }
        if (file.exists(file_to_open)) {    
            df_m <- open_file(file_to_open,norm_time,df_m,x,m,true_m_rate,time,full_exponent)
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

# df_m
#df_m[df_m$time == time_all[1],]
#  -------------------  ternary plots -------------------

# import the library to create ternary plots
library("ggplot2")
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

if (FALSE) {
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


# average quantities over all visc and mrate values
#  so I end up with one value for each time value
if (FALSE) {
    df_average <- data.frame(B_20=double(),B_21=double(),B_C=double(),B_22=double(),
    n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
    time=double(),norm_time=double())#,stringsAsFactors=FALSE)

    for (t in time_all)
    {
        print(paste("time = ",t))
        # condition: select all times that correspond to t
        av_n_I <- mean(df_m[df_m$time==t,]$n_I)
        av_n_Y <- mean(df_m[df_m$time==t,]$n_Y)
        av_n_X <- mean(df_m[df_m$time==t,]$n_X)
        print(paste("length of dataframe at this time step:",nrow(df_m[df_m$time==t,])))
        ## build dataframe
        df_av <- list(B_20=0.0,B_21=0.0,B_C=0.0,B_22=0.0,
        n_I=av_n_I,n_Y=av_n_Y,n_X=av_n_X,n_B=0.0,n_L=0.0,
        time=t,norm_time=0.0)
        # add the new line to the dataframe
        df_average <- rbind(df_average,df_av)#,stringsAsFactors=FALSE)
    }

    df_average

    if (TRUE) {
        
        # pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(fill=as.factor(true_m_rate)),shape = 21,stroke=2,size=2,colour="black")+ 
        # pt1 <- ggtern(data=df_m[df_m$time == time_all[1],],aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=as.factor(true_m_rate)))+ 
        pt1 <- ggtern(data=df_average,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=as.factor(time)))+ 
        scale_colour_brewer(palette='Blues')+
        scale_fill_distiller(direction=+1)+
        scale_fill_discrete(guide = guide_legend(reverse=TRUE))+    
        labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Time")+
        guides(color = guide_legend(reverse=TRUE))+    # low at the bottom, high at the top
        theme(plot.background = element_rect(fill='transparent', color=NA),
            #panel.grid.major = element_line(linetype = "dotted",colour = "black"),
            legend.background = element_rect(fill='transparent'),
            panel.background = element_rect(fill = "#e6dbd5"),
            legend.key = element_rect(fill = "#e6dbd5"),
            legend.position = c(.85, .65))#,alpha=0.8))
        ggsave(paste(base_path,"/branch_plots/br_ter_time_IXY.png",sep=''), pt1, bg='transparent')
    }

}

# select only two combinations of viscosity and melt rate and plot their evolution with time
if (TRUE){
    


    df_filtered <- df_m %>%
    filter(as.character(melt_rate) == "08" & viscosity == "1e3")

    df_filtered <- df_filtered %>%
    filter(time == 8000e6 | time == 24000e6 | time == 56000e6 | time == 128000e6)
    
    print(df_filtered)


    my_colors_blues <- c("#c6dbef", "#9ecae1", "#6baed6","#3182bd") # "#eff3ff")

    ternary_plot_a <- ggtern(data = df_filtered, mapping = aes(x=n_Y,y=n_I,z=n_X)) +
    transparent_background_for_tern() +
    geom_point(aes(color = as.factor(time)), size = 3) +
        scale_color_manual(values = my_colors_blues) +
        # scale_colour_brewer(palette='Blues')+
    theme_bw() +
    labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Time")   # labels for the vertices

    # ternary_plot_a <- ternary_plot_a + transparent_background_for_tern()

    ggsave(paste(base_path,"/branch_plots/br_ter_mu3mr08_time.png",sep=''), ternary_plot_a, bg='transparent')


    # second ternary plot
    df_filtered2 <- df_m %>%
    filter(as.character(melt_rate) == "03" & viscosity == "1e15")

    print(df_filtered2)

    df_filtered2 <- df_filtered2 %>%
    filter(time == 90000e6 | time == 150000e6 | time == 240000e6 | time == 300000e6) # 90000e6,150000e6,240000e6,300000e6

    print(df_filtered2)

    ternary_plot_b <- ggtern(data = df_filtered2, mapping = aes(x=n_Y,y=n_I,z=n_X)) +
    geom_point(aes(color = as.factor(time)), size = 3) +
        scale_colour_brewer(palette='Reds')+
    # theme_bw() +
    labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Time")   # labels for the vertices

    ternary_plot_b <- ternary_plot_b + transparent_background_for_tern()

    ggsave(paste(base_path,"/branch_plots/br_ter_mu15mr03_time.png",sep=''), ternary_plot_b, bg='transparent')
}
