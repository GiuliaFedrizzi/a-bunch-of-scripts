# ------------------------------------------
# analyses py_branch_info.csv files
# 
# plots parameters such as 'Frequency', 'Intensity', 'Dimensionless intensity'
# imports and plots number and ratios of types of nodes: I, Y and X
#
# run in the conda environment "r_env"
#   it contains tidyverse, patchwork
# specify time as command line argument e.g.
# > Rscript $MS/post_processing/image_analysis/branch_analysis.r 60e6   (remember: always format the number with e6, which means 10^6)
#
# or submit a task array job with sub_branchAn.sh from visc_*/vis*
#
# WARNING: delete the file called branches_df_time_*.csv if the data from extract_topology.py changed.
#
# Giulia March 2023
# ------------------------------------------

## Utils
# libraries
# library(tidyverse)
library(patchwork)
library(plyr)

source("/home/home01/scgf/myscripts/post_processing/image_analysis/useful_functions.R")

args <- commandArgs(trailingOnly = TRUE)  # store them as vectors
time = as.numeric(args[1])   # time for the 1st   (e.g. 60e6 = 60th file in mr_01). Don't use 6e7

# option: read the csv file that excludes the margins?
#  + no_margins <- 0 : py_branch_info.csv,   use "branch_plots"
#  + no_margins <- 1 : py_branch_info_x.csv, use "branch_plots_x"
no_margins <- 1

time_string <- sprintf("%02i",time/1e6)  # pad with zeros until string is 2 characters long
time_string <- paste(time_string,"e6",sep="")

if (no_margins) {
    csv_time_name <- paste("branches_df_time_",time_string,"_x.csv",sep="")
} else {
    csv_time_name <- paste("branches_df_time_",time_string,".csv",sep="")

}

scale_unit <- 1000  # edge of the square that gives an area comparable to the simulation (in pixels)
                    # can be checked from the size of the analysed png in extr topology:
                    # (1470-449)*(1189-183) = 1e6 , then I'll take the square of 1000, so I'll get 1e6 as the area

# some options for different sets of simulations
two_subdirs <- TRUE  # is it visc_1_1e1/vis1e1_mR01 (TRUE)  or just vis1e2_mR_01  (FALSE)?

# get what the variable is: viscosity or deformation rate?
dirs <- list.dirs()
if (sum(grepl("visc",dirs), na.rm=TRUE)>0){  # count how many times it finds "visc" in the subdirectories. if it is > 0 ...
    var_is_visc = 1  # the variable is "viscosity"
    var_is_def = 0
}else if (sum(grepl("def",dirs), na.rm=TRUE)>0){ # count how many times it finds "thdef" in the subdirectories
    var_is_visc = 0
    var_is_def = 1
}else{
    print(dirs)
    stop("I can't find the variable (viscosity or defomation rate)")
}


if (var_is_visc){
    if (two_subdirs){
        # x_variable <- c('1e1','1e15','1e2','1e25','1e3','1e35','1e4')  # the values of the x variable to plot (viscosity)
        x_variable <- find_dirs('visc')  # the values of the x variable to plot (viscosity)
    } else {
        x_variable <- c('1e2')  # just one value
    }
} else if (var_is_def) {
    # x_variable <- c('1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')#,'5e3','1e4')#,'2e4','4e4')  # the values of the x variable to plot (e.g. def rate)
    x_variable <- c('0e8','1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')  # the values of the x variable to plot (e.g. def rate)
}
melt_rate_list <- c('01','02','03','04','05','06','07','08','09')#,'1','2')
# melt_rate_list <- c('01','03','05','07','09')#,'08')#,'09')#,'1','2')
# melt_rate_list <- find_dirs('melt_rate')  # the values of the y variable to plot (melt rate)
# print(paste("melt_rate_list",melt_rate_list))

# first part of path
base_path <- getwd( )

if (grepl("prod",base_path)){    # prod zone, whole domain
    if (no_margins) {   # read py_branch_info_x, which excludes the margins (2% from each side)
        csv_file_name <- "py_branch_info_x.csv"
    } else {
        csv_file_name <- "py_branch_info.csv"
    }
} else if (grepl("through",base_path)) {
    # through zone, only top
    csv_file_name <- "py_branch_info_top.csv"
} else {
    csv_file_name <- "py_branch_info.csv"
}


# open file, extract values
build_branch_df <- function(x,m,time) {
        # to keep the amount of melt constant, we look at smaller times if melt rate is higher, later times if it's lower. This num will build the file name
        dirs <- list.dirs()
        if (sum(grepl("visc",dirs), na.rm=TRUE)>0){  # count how many times it finds "visc" in the subdirectories. if it is > 0 ...
            var_is_visc = 1
            var_is_def = 0
        }else if (sum(grepl("def",dirs), na.rm=TRUE)>0){ # count how many times it finds "thdef" in the subdirectories
            var_is_visc = 0
            var_is_def = 1
        }else{
            stop("I can't find the variable (viscosity or defomation rate")
        }

        norm_time = round(time/1e9/as.double(m))*1e9  # from accurate number, round so that it opens a file that exists

        if (startsWith(m,'0'))
        {
            true_m_rate = as.double(m)/1000
        } else {
            true_m_rate = as.double(m)/100
        }
        if (var_is_visc) {
            ##  build the path. unlist(strsplit(x,"e"))[2] splits '5e3' into 5 and 3 and takes the second (3)
            if (two_subdirs){
                ## 2 levels
                full_exponent <- unlist(strsplit(x,"e"))[2]  # could be 1 or 15, because there is no point (1.5)
                true_exponent <- unlist(strsplit(full_exponent,""))[1]  # takes the first character (1 in 1.5, 2 in 2.5 etc)
                potential_file_path <- paste(base_path,'/visc_',true_exponent,'_',x,'/vis',x,'_mR_',m,'/',sep="")
                # print(potential_file_path)
                if (dir.exists(potential_file_path)) {
                    # print("it exists!")
                }else {   # try a different version, the one that doesn't change with viscosity
                    potential_file_path <- paste(base_path,'/visc_',true_exponent,'_',x,'/vis1e2_mR_',m,'/',sep="")
                    print("trying a different version")
                    if (dir.exists(potential_file_path)) {
                        print("this one exists")
                    }else{
                        print("I've tried twice without success")
                    }
                }   
            file_to_open <- paste(potential_file_path,csv_file_name,sep="")
                # file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',csv_file_name,sep="")
                # file_to_open <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
            } else {
                ## only 1 level
                file_to_open <- paste(base_path,'/vis',x,'_mR_',m,'/',csv_file_name,sep="")
            }
        } else if (var_is_def) {
            # file_to_open <- paste(base_path,'/thdef',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
            file_to_open <- paste(base_path,'/pdef',x,'/vis1e2_mR',m,'/',csv_file_name,sep="")
        }
        # print(file_to_open)
        if (file.exists(file_to_open)) {    
            df_bi <- read.csv(file=file_to_open)
            print(paste("m rate",m,", norm time ",norm_time))
            df_bi_t <- df_bi[df_bi$time == norm_time,]  # get the line that corresponds to the time
            # print(df_bi_t)
            if (nrow(df_bi_t)>0){
                # I,X,Y nodes
                n_I <- df_bi_t$n_I
                # n_Y <- df_bi_t$n_2+df_bi_t$n_3  # including connectivity=2 in Y nodes
                n_Y <- df_bi_t$n_3   # NOT including connectivity=2 in Y nodes
                n_X <- df_bi_t$n_4+df_bi_t$n_5

                CV_hor <- df_bi_t$CV_hor
                CV_ver <- df_bi_t$CV_ver

                ## n of branches    n_B = 0.5 * (NI + 3NY + 4NX)
                n_B <- 0.5*sum(df_bi_t$n_I+3*(df_bi_t$n_3)+4*(df_bi_t$n_4)+5*(df_bi_t$n_5))
                n_L <- 0.5*(n_I+n_Y)   # n of lines

                ## B_20 'Frequency' : number of branches (from the node types) / Area 
                B_20 <- n_B / (scale_unit^2)
                # if (B_20 > 0){     # if there are fractures                  
                ## B_21  'Intensity': tot length / Area  (unit: m^-1)
                B_21 <- df_bi_t$branches_tot_length / (scale_unit^2)
                ## B_C  'Characteristic length'
                if (B_20 > 0){     # if there are fractures                  
                    B_C <- df_bi_t$branches_tot_length/n_B
                } else {
                    B_C <- 0
                }
                ## B_22  'Dimensionless intensity'
                B_22 <- B_20 * (B_C)^2
                if (m == "08"){
                    print(paste(x,"tot length",df_bi_t$branches_tot_length,"   B_20",B_20,"   B_21",B_21,"   B_C",B_C,"   B_22",B_22))
                }
                ## build dataframe
                if (var_is_visc) {
                    if (nchar(x) == 4) {  # 1e15 -> 15 -> 1.5 -> 10^(1.5)
                        de <- list(viscosity=10^(as.double(full_exponent)/10),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                    n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,CV_hor=CV_hor,CV_ver=CV_ver,time=time,norm_time=norm_time)
                        # visc <- 10^(as.double(full_exponent)/10)  
                    } else {
                        de <- list(viscosity=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                    n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,CV_hor=CV_hor,CV_ver=CV_ver,time=time,norm_time=norm_time)
                        # visc <-as.double(x)
                    }

                } else if (var_is_def) {
                    x <- gsub("e+","e-", x)
                    de <- list(def_rate=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                    n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,CV_hor=CV_hor,CV_ver=CV_ver,time=time,norm_time=norm_time)
                }
                de <- data.frame(de)
                df_m <- rbind.fill(df_m,de)#,stringsAsFactors=FALSE)

                print(paste("found ",x,", melt rate ",m," at time ",time,", norm time ",norm_time,sep=""))

                # df_m <- rbind(df_m,de,stringsAsFactors=FALSE)
            } else {
            print(paste("no visc",x,", melt rate ",m," at time ",time,", norm time ",norm_time,sep=""))            
            }

            } else {
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

if (var_is_visc) {
        df_m <- data.frame(viscosity=double(),melt_rate=factor(levels=melt_rate_list),true_m_rate=double(),
        B_20=double(),B_21=double(),B_C=double(),B_22=double(),
        n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),CV_hor=double(),CV_ver=double(),
        time=double(),norm_time=double())#,stringsAsFactors=FALSE)
    } else if (var_is_def) {
        df_m <- data.frame(def_rate=double(),melt_rate=factor(levels=melt_rate_list),true_m_rate=double(),
        B_20=double(),B_21=double(),B_C=double(),B_22=double(),
        n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),CV_hor=double(),CV_ver=double(),
        time=double(),norm_time=double())#,stringsAsFactors=FALSE)
    }

if (file.exists(csv_time_name)) {
    df_m <- read.csv(csv_time_name, header = TRUE, sep = ",")
    print("file exists")
} else {
    print("file doesn't exist, reading all csvs from directories")

    for (x_var in x_variable) {
        for (melt in melt_rate_list) {
            df_m <- build_branch_df(x_var,melt,time)
        }
    } 

    # branches/lines
    df_m["n_B_n_L"] <- df_m$n_B/df_m$n_L
    df_m["C_L"] <- 2*(df_m$n_Y+df_m$n_X)/df_m$n_L
    df_m["viscosity_scaled"] <-df_m$viscosity * 1e3
    write.csv(df_m, csv_time_name, row.names=FALSE)

}
df_m
# warnings()

# plot + save plot

# Define the directory name
if (no_margins){
    dir_name <- "branch_plots_x"  # differentiate the plots
} else {
    dir_name <- "branch_plots"
}

# Check if the directory exists
if (!dir.exists(dir_name)) {
  # The directory does not exist, so create it
  dir.create(dir_name)
  cat("Directory created: ", dir_name, "\n")
}

if (FALSE) {
    write.csv(df_m, csv_time_name, row.names=FALSE)
}

# update base_path to be where I save figures:
base_path <- paste(base_path,dir_name,sep='/')
print(base_path)


#  -------------------  ternary plots -------------------

# import the library to create ternary plots
library("ggplot2")
library("ggtern")
rows_to_keep <- !(df_m$n_Y == 0 & df_m$n_I == 0 & df_m$n_X == 0)  # remove rows if n_Y, n_I AND n_X are zero
df_no_zeros <- df_m[rows_to_keep, ]
if (FALSE) {

    # # connected-connected - isolated-connected - isolated-isolated
    # df_m["P_I"] <- df_m$N_I/(df_m$N_I+3*df_m$N_Y+4*df_m$N_X)   # probability of an I node
    # df_m["P_C"] <- (3*df_m$N_Y+4*df_m$N_X)/(df_m$N_I+3*df_m$N_Y+4*df_m$N_X)   # probability of a C node
    # df_m["P_II"] <- (df_m$P_I)^2  # probability of a branch with 2 I nodes if random distr
    # df_m["P_IC"] <- (df_m$P_I)*(df_m$P_C)  # probability of a branch with 1 I node and 1 C node if random distr
    # df_m["P_CC"] <- (df_m$P_C)^2  # probability of a branch with 2 C nodes if random distr
    
    # pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(fill=as.factor(true_m_rate)),shape = 21,stroke=2,size=2,colour="black")+ 
    pt1 <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=as.factor(true_m_rate)))+ 
    scale_colour_brewer(palette='YlGn')+
    scale_fill_distiller(direction=+1)+
    scale_fill_discrete(guide = guide_legend(reverse=TRUE))+    
    labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Melt Rate")+   # labels for the vertices
    guides(color = guide_legend(reverse=TRUE))+    # low at the bottom, high at the top
    theme(plot.background = element_rect(fill='transparent', color=NA),
        tern.panel.background = element_rect(fill = "#e6e6e6"),
        legend.key = element_rect(fill = "#e6e6e6"),
        legend.position = c(.85, .65))+#,alpha=0.8))
    transparent_background_for_tern()
    ggsave(paste(base_path,"/br_ter_melt_t",time_string,".png",sep=''), pt1, bg='transparent')


    df_no_zeros["mu_mr"] = (df_no_zeros$true_m_rate*1000)/log10(df_no_zeros$viscosity_scaled)
    print("df_no_zeros[mu_mr]")
    print(df_no_zeros["mu_mr"])
    pt_ratio <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=(mu_mr)))+ 
    # scale_colour_brewer(palette='BuGn')+
    scale_color_continuous(low="purple", high="orange") +
    # scale_fill_distiller(direction=+1)+
    # scale_fill_discrete(guide = guide_legend(reverse=TRUE))+    
    labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Melt Rate/log(Viscosity)")+   # labels for the vertices
    # guides(color = guide_legend(reverse=TRUE))+    # low at the bottom, high at the top
    theme(plot.background = element_rect(fill='transparent', color=NA),
        #panel.grid.major = element_line(linetype = "dotted",colour = "black"),
        # legend.background = element_rect(fill='transparent'),
        # panel.background = element_rect(fill = "#e6dbd5"),
        # legend.key = element_rect(fill = "#e6dbd5"),
        legend.position = c(.85, .65))+#,alpha=0.8))
    transparent_background_for_tern()
    ggsave(paste(base_path,"/br_ter_ratio_t",time_string,".png",sep=''), pt_ratio, bg='transparent')

    if (var_is_visc) {
        df_no_zeros$visc_factor <- factor(df_no_zeros$viscosity_scaled, ordered = TRUE)
        ptv <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X)) + 
        geom_point(aes(color = visc_factor)) + 
        scale_colour_brewer(palette='YlGnBu')+
        scale_fill_distiller(direction=+1)+
        scale_fill_discrete(guide = guide_legend(reverse=TRUE))+
        labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Viscosity")+
        guides(color = guide_legend(reverse=TRUE)) +    # low at the bottom, high at the top
        transparent_background_for_tern()+

        theme(
            # plot.background = element_rect(fill='transparent', color=NA),
            # legend.background = element_rect(fill='transparent'),
            tern.panel.background = element_rect(fill = "#e6e6e6"),
            legend.key = element_rect(fill = "#e6e6e6"),
            legend.position = c(.85, .65))#,alpha=0.8))

        ggsave(paste(base_path,"/br_ter_visc_t",time_string,"_trsp.png",sep=''), ptv, bg='transparent')
    } else if (var_is_def) {
        df_no_zeros$def_rate_factor <- factor(df_no_zeros$def_rate, ordered = TRUE)
        ptv <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X))+
        geom_point(aes(color=as.factor(def_rate_factor)))+#scale_colour_continuous(trans='reverse')+
        scale_colour_brewer(palette='Blues')+
        scale_fill_distiller(direction=+1)+
        scale_fill_discrete(guide = guide_legend(reverse=TRUE))+
        labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Deformation Rate")+
        guides(color = guide_legend(reverse=TRUE))+    # low at the bottom, high at the top
        # scale_size(breaks = rev(as.double(x_variable)))+
        theme(plot.background = element_rect(fill='transparent', color=NA),
        legend.background = element_rect(fill='transparent'),
        panel.background = element_rect(fill = "#e6dbd5"),
        legend.key = element_rect(fill = "#e6dbd5"),
        legend.position = c(.85, .65))#,alpha=0.8))  
        ggsave(paste(base_path,"/br_ter_def_t",time_string,"_trsp_1.png",sep=""), ptv, bg='transparent')
    }
}

# TERNARY PLOTS: one for each mr (or mu) category, coloured by mu (or mr).
if (FALSE){
    library(dplyr)
    if (var_is_visc){
        visc_or_def <- "viscosity_scaled"
    } else if (var_is_def){
        visc_or_def <- "def_rate"
    }
    if (var_is_visc){
        # create a column with categories of viscosity
        df_no_zeros <- df_no_zeros %>%
        mutate(
            var_category = case_when(
            viscosity_scaled < 1e5 ~ "1 - Low",
            viscosity_scaled == 1e5 ~ "2 - Intermediate",
            viscosity_scaled < 1e6 ~ "3 - High",
            viscosity_scaled >= 1e6 ~ "4 - Very high",
            TRUE ~ NA_character_  # This line handles any cases that don't match the above conditions
            )
        )
    } else if (var_is_def) {
        df_no_zeros <- df_no_zeros %>%
        mutate(
            var_category = case_when(
            def_rate == 0e-8 ~ "1 - No deformation",
            def_rate <= 3e-8 ~ "2 - Intermediate",
            def_rate <= 5e-8 ~ "3 - High",
            def_rate <= 9e-8 ~ "4 - Very high",
            TRUE ~ NA_character_  # This line handles any cases that don't match the above conditions
            )
        )
    }


    # create a column with categories of mr
    df_no_zeros <- df_no_zeros %>%
    mutate(
        mr_category = case_when(
        true_m_rate <= 0.002 ~ "1 - Low",
        true_m_rate <= 0.004 ~ "2 - Intermediate",
        true_m_rate <= 0.006  ~ "3 - High",
        true_m_rate >= 0.007 ~ "4 - Very high",
        TRUE ~ NA_character_  # This line handles any cases that don't match the above conditions
        )
    )
    print(df_no_zeros)

    # categories = viscosity   -  colour = melt rate

    png_name <- paste(base_path,"/br_ter_viscPanels_t",time_string,".png",sep='')  # build name of png
    # generate colours
    colours_mr <- colorRampPalette(c("#c2823a", "#33231E"))(length(unique(df_no_zeros$true_m_rate)))
    pt_pan_mr <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color = as.factor(true_m_rate)))+
        scale_color_manual(values = colours_mr)
    pt_pan_mr1 <- pt_pan_mr + facet_grid(cols = vars(var_category))+
        transparent_background_for_tern()+
        labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Melt\nProduction\nRate")+
        theme(
        tern.panel.background = element_rect(fill = "#e6e6e6"),
        legend.key = element_rect(fill = "#e6e6e6"))
    ggsave(png_name, pt_pan_mr1, bg='transparent', width = 30, height = 8, units = "cm")


    # categories = melt rate  -  colour = viscosity
    png_name <- paste(base_path,"/br_ter_meltPanels_t",time_string,".png",sep='')  # build name of png
    print("unique ----------")
    print(length(unique(df_no_zeros[[visc_or_def]])))
    colours_mu <- colorRampPalette(c("#aae88e","#397367", "#140021"))(length(unique(df_no_zeros[[visc_or_def]])))
    pt_pan_mu <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color = as.factor(.data[[visc_or_def]])))+
        scale_color_discrete(labels = function(x) format(as.numeric(x), scientific = TRUE))
    if (var_is_visc){
        colour_legend = "Melt\nViscosity"
    } else if (var_is_def) {
        colour_legend = "Deformation\nRate"

    }
    pt_pan_mu <- pt_pan_mu + labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = colour_legend)+
        # scale_fill_distiller(direction=+1)+
        scale_color_manual(values = colours_mu)
        
    pt_pan_mu1 <- pt_pan_mu + facet_grid(cols = vars(mr_category))+
        transparent_background_for_tern() +
        theme(
        tern.panel.background = element_rect(fill = "#e6e6e6"),
        legend.key = element_rect(fill = "#e6e6e6"))

    ggsave(png_name, pt_pan_mu1, bg='transparent', width = 30, height = 8, units = "cm")
}


plot_options <- theme(   # x and y here are not affected by flipping. Same AFTER flipping.
    plot.background = element_blank(),
    panel.background = element_blank(),
    # panel.grid.major = element_line(color = "#8ccde3",linewidth = 0.5,linetype = 2),
    legend.key=element_blank(),
    # axis.title.y=element_blank(),
	# axis.text.x=element_blank(),
	axis.line=element_line(colour = "black"),
	# axis.ticks=element_blank()
    )

# heatmaps combined with lineplots 
if (TRUE) {
    # heatmaps
    png_name <- paste(base_path,"/br_heat_B_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 6000,height = 3600,res=400)
    if (var_is_visc){
        p_heat1 <- ggplot(df_m,aes(factor(x=viscosity_scaled),factor(true_m_rate), fill=n_B))  + scale_fill_distiller(direction = +1,palette = 'Purples')+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  "Number of\nBranches")+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

        p_heat2 <- ggplot(df_m,aes(factor(x=viscosity_scaled),factor(true_m_rate), fill=B_21))  + scale_fill_distiller(direction = +1,palette = 'BuGn')+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill = "Total Branch\nLength")+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

        p_heat3 <- ggplot(df_m,aes(factor(x=viscosity_scaled),factor(true_m_rate), fill=B_C))  + scale_fill_distiller(direction = +1,palette = 'Greys')+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  "Average Branch\nLength")+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

        p_heat4 <- ggplot(df_m,aes(factor(x=viscosity_scaled),factor(true_m_rate), fill=B_22))  + scale_fill_distiller(direction = +1,,palette = 'Blues')+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  "Dimensionless\nIntensity")+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

    } else if (var_is_def){
        p_heat1 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=n_B))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Def Rate",y = "Melt Rate",fill =  "Number of\nBranches")
        p_heat2 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_21))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Def Rate",y = "Melt Rate",fill = "Total Branch\nLength")
        p_heat3 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_C))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Def Rate",y = "Melt Rate",fill =  "Average Branch\nLength")
        p_heat4 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_22))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Def Rate",y = "Melt Rate",fill =  "Dimensionless\nIntensity")
    }
    # melt rate v B   lineplots
    if (var_is_visc){
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=n_B)) + geom_point(aes(color = factor(x=viscosity_scaled)))+ geom_line(aes(color = factor(x=viscosity_scaled)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm1 <- pm1 + labs(x ="Melt Rate",y =  expression('N'[B]),colour = "Viscosity")
        pm1 <- pm1  + plot_options + scale_colour_brewer(palette='Spectral')+ 
        theme(legend.position="none")
        # scale_y_discrete(breaks =seq(200,500, by = 100)) +  # , expand = c(0, 0)
        # coord_fixed(xlim = c(min(df_m$true_m_rate), max(df_m$true_m_rate)),ylim = c(min(df_m$B_20), max(df_m$B_20)))+
        # theme(panel.grid.major.y = element_line(color = "#8ccde3",linewidth = 0.5,linetype = 2))

        # pm1 <- pm1 + theme(axis.title.y=element_blank(),axis.text.y=element_blank())
        # pm1 <- pm1 + coord_flip()  # rotate data by 90 degrees
        # set axes: expand is to stop extra space around the plot area (tight around the data)
        # BUT I need to manually set 'limits', to have the grid 
        # pm1 <- pm1 + scale_x_discrete(breaks = seq(min(df_m$B_20), max(df_m$B_20), by = 10),expand = c(0, 0)) +   #  limits=factor(as.double(melt_rate_list)/1000)  didn't work
        # scale_y_discrete(breaks = seq(min(df_m$B_20), max(df_m$B_20), by = 10),expand = c(0, 0)) 

        pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=viscosity_scaled)))+ geom_line(aes(color = factor(x=viscosity_scaled)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # pm2 + theme_bw()
        pm2 <- pm2 + plot_options + scale_colour_brewer(palette='Spectral')# + theme(axis.title.y=element_blank())
        # pm2 <- pm2 + coord_flip()+  # rotate data by 90 degrees
        pm2 <- pm2 +labs(x = "Melt Rate",y = expression('B'[21]),colour = "Viscosity")#+
        # scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))

        pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=viscosity_scaled)))+ geom_line(aes(color = factor(x=viscosity_scaled)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # pm3 + theme_bw()
        pm3 <- pm3 + plot_options + scale_colour_brewer(palette='Spectral') #+ theme(axis.title.y=element_blank())
        # pm3 <- pm3 + coord_flip()+  # rotate data by 90 degrees
        pm3 <- pm3 +labs(x = "Melt Rate",y = expression('B'[C]),colour = "Viscosity")   #  labels as they'd be before flipping
        #scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))
        
        pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=viscosity_scaled)))+ geom_line(aes(color = factor(x=viscosity_scaled)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # pm4 + theme_bw()
        pm4 <- pm4 + plot_options + scale_colour_brewer(palette='Spectral') #+ theme(axis.title.y=element_blank())
        # pm4 <- pm4 + coord_flip()+  # rotate data by 90 degrees
        pm4 <- pm4 +labs(x = "Melt Rate",y = expression('B'[22]),colour = "Viscosity")+ 
        theme(legend.position="none")
        #scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))
        #p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    } else if (var_is_def){
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=n_B)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        #p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    }

    # viscosity v B   lineplots
    if (var_is_visc){
        pv1 <- ggplot(data=df_m,mapping = aes(x=viscosity_scaled,y=n_B)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate))) + scale_x_continuous(trans='log10')+#,expand = c(0, 0)) +
        labs(x = "Viscosity",y = expression('N'[B]),colour = "Melt Rate")+ 
        theme(legend.position="none")  #  remove legend so that the plot takes 100% of the space
        pv1 <- pv1 + plot_options 

        pv2 <- ggplot(data=df_m,mapping = aes(x=viscosity_scaled,y=B_21)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate))) + scale_x_continuous(trans='log10',expand = c(0, 0))+
        labs(x = "Viscosity",y = expression('B'[21]),colour = "Melt Rate")+ 
        guides(fill = guide_legend(reverse = TRUE))  # reverse order so that lowest mr is at the bottom
        pv2 <- pv2 + plot_options 

        pv3 <- ggplot(data=df_m,mapping = aes(x=viscosity_scaled,y=B_C)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate))) + scale_x_continuous(trans='log10',expand = c(0, 0))+
        labs(x = "Viscosity",y = expression('B'[C]),colour = "Melt Rate")
        pv3 <- pv3 + plot_options 

        pv4 <- ggplot(data=df_m,mapping = aes(x=viscosity_scaled,y=B_22)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate))) + scale_x_continuous(trans='log10',expand = c(0, 0))+
        labs(x = "Viscosity",y = expression('B'[22]),colour = "Melt Rate")+ 
        theme(legend.position="none")
        pv4 <- pv4 + plot_options 

    } else if (var_is_def){
        pv1 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=n_B)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))# + scale_x_continuous(trans='log10') 
        pv2 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_21)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))# + scale_x_continuous(trans='log10')
        pv3 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_C)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate))) #+ scale_x_continuous(trans='log10')
        pv4 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_22)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))# + scale_x_continuous(trans='log10')
    }

    if (FALSE){
    # define the layout
        hlay <- rbind(c(1,1,1,NA,NA,NA, 2,2,2, NA,NA),  # NA means empty
                c(3,3,3, 4,4,   NA,  5,5,5,  6,6),
                c(3,3,3, 4,4,   NA,  5,5,5,  6,6),
                c(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA),
                c(7,7,7,NA,NA,  NA,  8,8,8, NA,NA),
                c(9,9,9,10,10,  NA,  11,11,11,12,12),
                c(9,9,9,10,10,  NA,  11,11,11,12,12))
    }
    if(TRUE){
        hlay <- rbind(     # NA means empty
                c(3,3, 1,1,  5,5,  2,2),
                c(3,3, 4,4,  5,5,  6,6),
                # c(NA,NA,NA,NA,NA,NA,NA,NA,NA),
                c(9,9, 7,7,  11,11,8,8),
                c(9,9,10,10, 11,11,12,12))
    }

    pg <- grid.arrange(pv1,pv2,p_heat1,pm1,p_heat2,pm2,pv3,pv4,p_heat3,pm3,p_heat4,pm4, layout_matrix=hlay)
    print(pg)
    dev.off()
}


if (FALSE) {
    png_name <- paste(base_path,"/br_heat_nb_cl_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 2800,height = 2800,res=100)

    if (var_is_visc){
        p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=n_B_n_L))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
        p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=C_L))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
        # plot melt rate
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=n_B_n_L)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=C_L)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # plot viscosity
        pv1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B_n_L)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
        pv2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=C_L)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')

    } else if (var_is_def){
        p_heat1 <- ggplot(df_m,aes(factor(x=def_rate),melt_rate, fill=n_B_n_L))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
        p_heat2 <- ggplot(df_m,aes(factor(x=def_rate),melt_rate, fill=C_L))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
        # plot melt rate
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=n_B_n_L)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=C_L)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # plot def_rate
        pv1 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=n_B_n_L)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))#,linetype = "dashed") 
        pv2 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=C_L)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))#,linetype = "dashed") 
        # ggplot(data=df_m,mapping = aes(x=def_rate,y=B_20)) 
    }


    # define the layout
    hlay <- rbind( c(NA,NA, NA,NA), 
                c(3,1, 5,2),
                c(3,4, 5,6),
                c(NA,NA, NA,NA))

    pg <- grid.arrange(pv1,pv2,p_heat1,pm1,p_heat2,pm2,layout_matrix=hlay)
    # print(pg)
    dev.off()
}

if (TRUE) {     
    df_CV_hor <- df_m[df_m$CV_hor != 0, ]  # filter out if CV_hor is 0
    df_CV_ver <- df_m[df_m$CV_ver != 0, ]  # filter out if CV_ver is 0
    print(df_CV_hor)
    png_name <- paste(base_path,"/br_heat_CV_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 2800,height = 2800,res=100)
    if (var_is_visc){
        visc_or_def <- "viscosity"
    } else if (var_is_def){
        visc_or_def <- "def_rate"
    }
        p_heat1 <- ggplot(df_CV_hor,aes(factor(x=.data[[visc_or_def]]),melt_rate, fill=CV_hor))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
        p_heat2 <- ggplot(df_CV_ver,aes(factor(x=.data[[visc_or_def]]),melt_rate, fill=CV_ver))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))

        # plot melt rate
        pm1 <- ggplot(data=df_CV_hor,mapping = aes(x=(true_m_rate),y=CV_hor)) + geom_point(aes(color = factor(x=.data[[visc_or_def]])))+ geom_line(aes(color = factor(x=.data[[visc_or_def]])))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm2 <- ggplot(data=df_CV_ver,mapping = aes(x=(true_m_rate),y=CV_ver)) + geom_point(aes(color = factor(x=.data[[visc_or_def]])))+ geom_line(aes(color = factor(x=.data[[visc_or_def]])))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')

        # plot viscosity or def rate
        # pv1 <- ggplot(data=df_CV_hor,mapping = aes(factor(x=.data[[visc_or_def]]),y=CV_hor)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))#,linetype = "dashed")# + scale_x_continuous(trans='log10') 
        pv1 <- ggplot(data=df_CV_hor,mapping = aes((x=.data[[visc_or_def]]),y=CV_hor)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))#,linetype = "dashed")# + scale_x_continuous(trans='log10') 
        pv2 <- ggplot(data=df_CV_ver,mapping = aes((x=.data[[visc_or_def]]),y=CV_ver)) + geom_point(aes(color = factor(true_m_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(true_m_rate)))#,linetype = "dashed")# + scale_x_continuous(trans='log10')
    if (var_is_visc){
        pv1 <- pv1 + scale_x_continuous(trans='log10') 
        pv2 <- pv2 + scale_x_continuous(trans='log10') 
    }

    # define the layout
    hlay <- rbind( c(NA,NA, NA,NA), 
                c(3,1, 5,2),
                c(3,4, 5,6),
                c(NA,NA, NA,NA))

    pg <- grid.arrange(pv1,pv2,p_heat1,pm1,p_heat2,pm2,layout_matrix=hlay)
    # print(pg)
    dev.off()
}