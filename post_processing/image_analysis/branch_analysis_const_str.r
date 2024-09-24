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

scale_unit <- 1000  # edge of the square that gives an area comparable to the simulation (in pixels)

time_string <- sprintf("%02i",time/1e6)  # pad with zeros until string is 2 characters long
time_string <- paste(time_string,"e6",sep="")

if (no_margins) {
    csv_time_name <- paste("branches_df_const_str_time_",time_string,"_xx.csv",sep="")
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

melt_rate_list <- c('01','02','03','04','05','06','07','08','09')

target_mr_def_ratios <- c(1/5,1/3,1/2,1.0,1.5,2.0)  # save figures with this melt rate - deformation rate ratio (= constant strain)

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
                n_B <-  0.5*sum(df_bi_t$n_I+3*(df_bi_t$n_3)+4*(df_bi_t$n_4)+5*(df_bi_t$n_5))
                n_L <- 0.5*(n_I+n_Y)

                ## B_20 'Frequency' : number of branches (from the node types) / Area 
                B_20 <- n_B / (scale_unit^2)
                # if (B_20 > 0){     # if there are fractures                  
                ## B_21  'Intensity'
                B_21 <- df_bi_t$branches_tot_length / (scale_unit^2)
                ## B_C  'Characteristic length'
                if (B_20 > 0){     # if there are fractures                  
                    B_C <-  df_bi_t$branches_tot_length/n_B
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
find_closest <- function(x, targets, relative_threshold) {
  # Calculate the relative differences between x and all target values
  relative_differences <- abs(targets - x) / x

  # Find the indices of the targets where the relative difference is within the threshold
  close_indices <- which(relative_differences <= relative_threshold)
#   print(paste("Close Indices:", paste(close_indices, collapse=", ")))
  
  if (length(close_indices) == 0) {
    # If no close targets, return NA or some default value
    return(NA)
  } else {
    # If there are close targets, return the closest one
    closest_index <- close_indices[which.min(relative_differences[close_indices])]
    # print(paste("-- returning",closest_index))
    return(targets[closest_index])
  }
}

if (file.exists(csv_time_name)) {
    df_m <- read.csv(csv_time_name, header = TRUE, sep = ",")
    print("file exists")
    print(csv_time_name)
} else {
    print("file doesn't exist, reading all csvs from directories")

    for (x_var in x_variable) {
        for (melt in melt_rate_list) {
            df_m <- build_branch_df(x_var,melt,time)
        }
    } 
    write.csv(df_m, csv_time_name, row.names=FALSE)  # here it saves df to a file to save time
}

# options for printing more columns side by side
options(tibble.print_max = Inf, tibble.print_min = Inf)
options(width = 200)

print("original")
print(df_m)

# save original dataframe
df_original <- df_m

# assign closest target. The smaller the threshold, the pickier it is
df_m <- df_m %>%
    dplyr::mutate(closest_target = purrr::map_dbl(mr_def_ratio, ~find_closest(.x, target_mr_def_ratios, relative_threshold = 0.15)))

# clean the dataframe from rows that are too far from a target
df_m <- df_m %>%
  filter(!is.na(closest_target))

# Calculate the absolute difference between mr_def_ratio and closest_target
df_m <- df_m %>%
dplyr::mutate(differ = abs(mr_def_ratio - closest_target))


print("after removing NA")
print(df_m)


rows_to_keep <- !(df_m$B_20 == 0 & df_m$B_21 == 0 & df_m$B_C == 0)  # remove rows if B_20, B_21 AND B_C are zero
df_m <- df_m[rows_to_keep, ]

print("no zeros")
print(df_m)


# branches/lines
df_m["n_B_n_L"] <- df_m$n_B/df_m$n_L
df_m["C_L"] <- 2*(df_m$n_Y+df_m$n_X)/df_m$n_L
# write.csv(df_m, csv_time_name, row.names=FALSE)  # here it saves df to a file to save time
df_m_no_filter <- df_m

# df_m <- df_m %>%
#   group_by(def_rate, closest_target) %>%  # if they have the same def_rate and closest_target
#   filter(differ == min(differ) & id == min(id[differ == min(differ)]))  # pick the one that has the smallest difference. 
  # If that is still the same, pick the one with the smallest id

df_m <- df_m %>%
  group_by(def_rate, closest_target) %>%
  filter(row_number() == which.min(differ))

print("after filtering")
print(df_m)

eliminated_rows <- anti_join(df_m_no_filter, df_m, by = colnames(df_m_no_filter))
print("eliminated_rows")
print(eliminated_rows)


df_m$form_mr_def_ratio <- sprintf("%.2f", df_m$mr_def_ratio)
df_m$form_target_ratio <- sprintf("%.2f", df_m$closest_target)


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

# include all values and use their real ratio to assign them to classes (not just the ones close to a target ratio)
df_original$true_mr_def_ratio <- as.double(df_original$true_m_rate)/as.double(df_original$def_rate)*1e-5

rows_to_keep <- !(df_original$n_Y == 0 & df_original$n_I == 0 & df_original$n_X == 0)  # remove rows if n_Y, n_I AND n_X are zero
df_no_zeros <- df_original[rows_to_keep, ]



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
        mr_def_category = case_when(
        true_mr_def_ratio <= 0.25 ~ "1 - 0.25 and lower",
        true_mr_def_ratio <= 0.75 ~ "2 - between 0.25 and 0.75",
        true_mr_def_ratio <= 1.25  ~ "3 - around 1.0",
        true_mr_def_ratio > 1.25 ~ "4 - Greater than 1.25",
        TRUE ~ NA_character_  # This line handles any cases that don't match the above conditions
        )
    )
    # print(df_no_zeros)
    print(df_no_zeros[c('true_m_rate','def_rate','true_mr_def_ratio','mr_def_category')]) 
    

    # categories = viscosity   -  colour = melt rate

    png_name <- paste(base_path,"/br_ter_defPanels_cs_x_t",time_string,".png",sep='')  # build name of png
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
    png_name <- paste(base_path,"/br_ter_meltPanels_cs_x_t",time_string,".png",sep='')  # build name of png
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
        
    pt_pan_mu1 <- pt_pan_mu + facet_grid(cols = vars(mr_def_category))+
        transparent_background_for_tern() +
        theme(
        tern.panel.background = element_rect(fill = "#e6e6e6"),
        legend.key = element_rect(fill = "#e6e6e6"))

    ggsave(png_name, pt_pan_mu1, bg='transparent', width = 30, height = 8, units = "cm")
}


plot_options <- theme(   # x and y here are not affected by flipping. Same AFTER flipping.
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "#7d7d7d",linewidth = 0.3,linetype = 2),
    panel.grid.minor = element_line(color = "#7d7d7d",linewidth = 0.1,linetype = 2),
    legend.key=element_blank(),
    # axis.title.y=element_blank(),
	# axis.text.x=element_blank(),
	axis.line=element_line(colour = "black")
	# axis.ticks=element_blank()
    )

# heatmaps combined with lineplots 
if (TRUE) {
    # heatmaps
    png_name <- paste(base_path,"/br_heat_B_cs_x_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 3000,height = 1800,res=100)

    p_heat1 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=n_B))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill =  "Number of\nBranches")
    p_heat2 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_21))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill = "Total Branch\nLength")
    p_heat3 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_C))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill =  "Average Branch\nLength")
    p_heat4 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_22))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
    labs(x = "Def Rate",y = "Melt Rate/Def Rate",fill =  "Dimensionless\nIntensity")

    print(paste("min B20 =",min(df_m$B_20),"max = ",max(df_m$B_20)))
    # melt rate v B   lineplots

    pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=n_B)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
    pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    

    #  define palette for form_target_ratio
    colours_ratios <- colorRampPalette(c("#A9EED3","#84CFB7","#53B4C6","#3E89C6","#3B4FB1","#38099E", "#070120"))(length(unique(df_m$form_target_ratio)))
    print(c("unique ratios: ",length(unique(df_m$form_target_ratio))))

    # def rate v B   lineplots

    pv1 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=n_B)) + geom_point(aes(color = factor(form_target_ratio)))+ 
        theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(form_target_ratio)))+
        # scale_color_gradientn(colors = brewer.pal(9, "YlGnBu"), 
        #                 limits = c(min(df_m$form_target_ratio), max(df_m$form_target_ratio) * 0.9))+
        scale_color_manual(values = colours_ratios)+
        labs(x = "Deformation Rate",y = "Number of\nBranches",colour = "Melt Rate/Deformation \nRate")
    pv1 <- pv1 + plot_options 

    pv2 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_21)) + geom_point(aes(color = factor(form_target_ratio)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(form_target_ratio)))+
        labs(x = "Deformation Rate",y = "Total Branch\nLength",colour = "Melt Rate/Deformation \nRate")
    pv2 <- pv2 + plot_options 

    pv3 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_C)) + geom_point(aes(color = factor(form_target_ratio)))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(form_target_ratio)))+
        labs(x = "Deformation Rate",y = "Average Branch\nLength",colour = "Melt Rate/Deformation \nRate")
    pv3 <- pv3 + plot_options 

    pv4 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_22)) + geom_point(aes(color = factor(form_target_ratio)))+ 
    theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = factor(form_target_ratio)))+
        scale_color_manual(values = colours_ratios)+
        labs(x = "Deformation Rate",y = "Dimensionless\nIntensity",colour = "Melt Rate/Deformation \nRate")
    pv4 <- pv4 + plot_options 
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
    png_name <- paste(base_path,"/br_heat_nb_cl_cs_x_",time_string,".png",sep='')  # build name of png
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
    png_name <- paste(base_path,"/br_heat_CV_cs_x_",time_string,".png",sep='')  # build name of png
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