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
library(dplyr)
library(purrr)

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
    csv_time_name <- paste("branches_df_const_str_time_",time_string,"_x.csv",sep="")
} else {
    csv_time_name <- paste("branches_df_const_str_time_",time_string,".csv",sep="")

}
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
        x_variable <- find_dirs('visc')  # the values of the x variable to plot (viscosity)
    } else {
        x_variable <- c('1e2')  # just one value
    }
} else if (var_is_def) {
    # x_variable <- c('1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')#,'5e3','1e4')#,'2e4','4e4')  # the values of the x variable to plot (e.g. def rate)
    x_variable <- c('1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')#,'5e3','1e4')#,'2e4','4e4')  # the values of the x variable to plot (e.g. def rate)
}

melt_rate_list <- c('01','03','05','07','09')

target_mr_def_ratios <- c(3.0,2.5,2.0,1.5,1.0,1/2,1/3,1/5,1/6,1/7)  # save figures with this melt rate - deformation rate ratio (= constant strain)

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

                ## n of branches, n of lines
                n_B <- sum(df_bi_t$n_I+df_bi_t$n_2+df_bi_t$n_3+df_bi_t$n_4+df_bi_t$n_5)
                n_L <- 0.5*(n_I+n_Y)

                ## B_20 'Frequency' 
                B_20 <- sum(df_bi_t$n_I+df_bi_t$n_2+df_bi_t$n_3+df_bi_t$n_4+df_bi_t$n_5)
                # if (B_20 > 0){     # if there are fractures                  
                ## B_21  'Intensity'
                B_21 <- df_bi_t$branches_tot_length
                ## B_C  'Characteristic length'
                if (B_20 > 0){     # if there are fractures                  
                    B_C <- B_21/B_20
                } else {
                    B_C <- 0
                }
                ## B_22  'Dimensionless intensity'
                B_22 <- B_20 * (B_C)^2
                
                ## build dataframe
                x <- gsub("e+","e-", x)
                de <- list(def_rate=as.double(x),true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,CV_hor=CV_hor,CV_ver=CV_ver,norm_time=norm_time,mr_def_ratio=as.double(m)/as.double(x)*1e-8)
            
                de <- data.frame(de)
                df_m <- rbind.fill(df_m,de)

                print(paste("found ",x,", melt rate ",m," at time ",time,", norm time ",norm_time,sep=""))
            } else {
                print(paste("no visc",x,", melt rate ",m," at time ",time,", norm time ",norm_time,sep=""))            
            }

            } else {
                print(paste("file does not exist:",file_to_open))
        }
        return(df_m)
}

# initialise empty dataframe
df_m <- data.frame(def_rate=double(),true_m_rate=double(),
B_20=double(),B_21=double(),B_C=double(),B_22=double(),
n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),CV_hor=double(),CV_ver=double(),
norm_time=double(),mr_def_ratio=double())#,stringsAsFactors=FALSE)

# Define a function to find the closest target mr/def ratio using a relative threshold
find_closest <- function(x, targets, relative_threshold = 0.5) {
  distances <- abs(targets - x)
  relative_distances <- distances / targets
  min_relative_distance <- min(relative_distances)
  
  if (min_relative_distance <= relative_threshold) {
    closest_value <- targets[which.min(relative_distances)]
  } else {
    closest_value <- NA  # Return NA if no value is close enough
  }
  return(closest_value)
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
    write.csv(df_m, csv_time_name, row.names=FALSE)  # here it saves df to a file to save time
}
    # assign closest target
    df_m <- df_m %>%
        dplyr::mutate(closest_target = purrr::map_dbl(mr_def_ratio, ~find_closest(.x, target_mr_def_ratios, relative_threshold = 0.1)))
    
    # Filter out rows with no close target value
    # df_m <- df_m %>%
    # dplyr::filter(!is.na(closest_target))

    # Calculate the absolute difference between mr_def_ratio and closest_target
    df_m <- df_m %>%
    dplyr::mutate(diff = abs(mr_def_ratio - closest_target))

    # Group by closest_target and keep the row with the minimum difference
    df_m <- df_m %>%
    dplyr::group_by(closest_target) %>%
    dplyr::filter(diff == min(diff)) %>%
    dplyr::ungroup()

    # # Drop the diff column - no longer needed
    # df_m <- df_m %>%
    # select(-diff)
    
    
    # branches/lines
    df_m["n_B_n_L"] <- df_m$n_B/df_m$n_L
    df_m["C_L"] <- 2*(df_m$n_Y+df_m$n_X)/df_m$n_L
    # write.csv(df_m, csv_time_name, row.names=FALSE)  # here it saves df to a file to save time

# }
print(df_m, width = Inf)

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
    pt_pan_mu <- ggtern(data=df_no_zeros,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color = as.factor(.data[[visc_or_def]])))
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
	axis.line=element_line(colour = "black")
	# axis.ticks=element_blank()
    )

# heatmaps combined with lineplots 
if (TRUE) {
    # heatmaps
    png_name <- paste(base_path,"/br_heat_B_const_str_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 3000,height = 1800,res=100)

    p_heat1 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill =  "Number of\nBranches")
    p_heat2 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_21))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill = "Total Branch\nLength")
    p_heat3 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_C))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill =  "Average Branch\nLength")
    p_heat4 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_22))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill =  "Dimensionless\nIntensity")

    print(paste("min B20 =",min(df_m$B_20),"max = ",max(df_m$B_20)))
    # melt rate v B   lineplots

    pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_20)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
    pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    

    # def rate v B   lineplots

    pv1 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_20)) + geom_point(aes(color = factor(mr_def_ratio)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(mr_def_ratio)))+
        labs(x = "Def Rate",y = "Number of\nBranches",colour = "Melt Rate/Def Rate")
    pv2 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_21)) + geom_point(aes(color = factor(mr_def_ratio)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(mr_def_ratio)))+
        labs(x = "Def Rate",y = "Total Branch\nLength",colour = "Melt Rate/Def Rate")
    pv3 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_C)) + geom_point(aes(color = factor(mr_def_ratio)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(mr_def_ratio)))+
        labs(x = "Def Rate",y = "Average Branch\nLength",colour = "Melt Rate/Def Rate")
    pv4 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_22)) + geom_point(aes(color = factor(mr_def_ratio)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(mr_def_ratio)))+
        labs(x = "Def Rate",y = "Dimensionless\nIntensity",colour = "Melt Rate/Def Rate")
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



#  ONE HEATMAP at a time, combined with its lineplots
if(FALSE) {
    library(cowplot)
    png_name <- paste(base_path,"/br_heat_B20_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 2800,height = 2800,res=500)
    if (var_is_visc){
        p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),true_m_rate, fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  expression('B'[20]))
    } else if (var_is_def){
        p_heat1 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
    }

    # melt rate v B   lineplots
    if (var_is_visc){
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_20)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        # labs(x = "Melt Rate",y = expression('B'[20]),colour = "Viscosity")
    } else if (var_is_def){
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_20)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
    }
    pm1 <- pm1 + coord_flip()  # rotate data by 90 degrees
    pm1 <- pm1 + plot_options
    
    # viscosity v B   lineplots
    if (var_is_visc){
        pv1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') +
        labs(x = "Viscosity",y = expression('B'[20]),colour = "Melt Rate")
    } else if (var_is_def){
        pv1 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_20)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
    }

    # change margins 
    p_heat1 <- p_heat1  + theme(plot.margin = unit(c(0, 0, 0.5, 0.5), "cm"))  #   top, right, bottom, left
    pm1 <- pm1 + theme(plot.margin = unit(c(0.5, 0, 0, 0.7), "cm"))
    pv1 <- pv1 + theme(plot.margin = unit(c(0, 0.5, 0.5, 0.5), "cm"))

    # combine plots together in a grid
    first_col = plot_grid(pv1, p_heat1, ncol = 1, rel_heights = c(1, 3))
    second_col = plot_grid(NULL, pm1, ncol = 1, rel_heights = c(1, 3))
    full_plot = plot_grid(first_col, second_col, ncol = 2, rel_widths = c(3, 1))
    
    print(full_plot)
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

if (FALSE) {     
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
        p_heat1 <- ggplot(df_CV_hor,aes(factor(x=.data[[visc_or_def]]),true_m_rate, fill=CV_hor))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
        p_heat2 <- ggplot(df_CV_ver,aes(factor(x=.data[[visc_or_def]]),true_m_rate, fill=CV_ver))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))

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