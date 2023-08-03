# ------------------------------------------
# analyses py_branch_info.csv files
# specify name of subdirectories (viscosity/def rate + melt rate)
# plots parameters such as 'Frequency', 'Intensity', 'Dimensionless intensity'
# imports and plots number and ratios of types of nodes: I, Y and X
#
# need to specify if the variable is viscosity or deformation rate (in 2 places!)
#  and if there are 2 orders of subdirectories or just 1 
#
# run in the conda environment "r_env"
#   it contains tidyverse, patchwork
# specify time as command line argument e.g.
# > Rscript $MS/post_processing/image_analysis/branch_analysis.r 60e6 
#
# or submit a task array job with sub_branchAn.sh from visc_*/vis*
#
# Giulia March 2023
# ------------------------------------------

## Utils
# libraries
# library(tidyverse)
library(patchwork)
library(plyr)
# library('cowplot')
# library(ggthemes)

args <- commandArgs(trailingOnly = TRUE)  # store them as vectors

# some options for different sets of simulations
two_subdirs <- TRUE  # is it visc_1_1e1/vis1e1_mR01 (TRUE)  or just vis1e2_mR_01  (FALSE)?

# get what the variable is: viscosity or deformation rate?
dirs <- list.dirs()
if (sum(grepl("visc",dirs), na.rm=TRUE)>0){  # count how many times it finds "visc" in the subdirectories. if it is > 0 ...
    var_is_visc = 1  # the variable is "viscosity"
    var_is_def = 0
}else if (sum(grepl("thdef",dirs), na.rm=TRUE)>0){ # count how many times it finds "thdef" in the subdirectories
    var_is_visc = 0
    var_is_def = 1
}else{
    stop("I can't find the variable (viscosity or defomation rate")
}
# var_is_visc = 0
# var_is_def = 1

if (var_is_visc){
    if (two_subdirs){
        # x_variable <- c('1e1','5e1','1e2','5e2','1e3','5e3','1e4')  # the values of the x variable to plot (viscosity)
        # x_variable <- c('1e1','1e2','5e2','1e3','5e3','1e4')  # the values of the x variable to plot (viscosity)
        x_variable <- c('1e1','1e2','1e3','1e4')  # the values of the x variable to plot (viscosity)
    } else {
        x_variable <- c('1e2')  # just one value
    }
} else if (var_is_def) {
    x_variable <- c('1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8')#,'5e3','1e4')#,'2e4','4e4')  # the values of the x variable to plot (e.g. def rate)
}
# melt_rate_list <- c('01','02','03','05','04','06','07','08','09')#,'1','2')
melt_rate_list <- c('02','04','06','08')#,'08')#,'09')#,'1','2')
# melt_rate_list <- c('01')#,'09')

# set some options automatically
time = as.numeric(args[1])   # time for the 1st   (e.g. 60e6 = 60th file in mr_01). Don't use 6e7

# first part of path
base_path <- getwd( )

if (grepl("prod",base_path)){
    # prod zone, whole domain
    csv_file_name <- "py_branch_info.csv"
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
        }else if (sum(grepl("thdef",dirs), na.rm=TRUE)>0){ # count how many times it finds "thdef" in the subdirectories
            var_is_visc = 0
            var_is_def = 1
        }else{
            stop("I can't find the variable (viscosity or defomation rate")
        }

        norm_time = round(time/1e6/as.double(m))*1e6  # from accurate number, round so that it opens a file that exists

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
                potential_file_path <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis',x,'_mR_',m,'/',sep="")
                # print(potential_file_path)
                if (dir.exists(potential_file_path)) {
                    # print("it exists!")
                }else {   # try a different version, the one that doesn't change with viscosity
                    potential_file_path <- paste(base_path,'/visc_',unlist(strsplit(x,"e"))[2],'_',x,'/vis1e2_mR_',m,'/',sep="")
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
            file_to_open <- paste(base_path,'/thdef',x,'/vis1e2_mR_',m,'/',csv_file_name,sep="")
        }
        # print(file_to_open)
        if (file.exists(file_to_open)) {    
            df_bi <- read.csv(file=file_to_open)
            print(paste("m rate",m,", norm time ",norm_time))
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
                    if (var_is_visc) {
                        de <- list(viscosity=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                        n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,time=time,norm_time=norm_time)
                    } else if (var_is_def) {
                        x <- gsub("e+","e-", x)
                        de <- list(def_rate=as.double(x),melt_rate=m,true_m_rate=true_m_rate,B_20=B_20,B_21=B_21,B_C=B_C,B_22=B_22,
                        n_I=n_I,n_Y=n_Y,n_X=n_X,n_B=n_B,n_L=n_L,time=time,norm_time=norm_time)
                    }
                    de <- data.frame(de)
                    df_m <- rbind.fill(df_m,de)#,stringsAsFactors=FALSE)
                }

                # df_m <- rbind(df_m,de,stringsAsFactors=FALSE)
            } else {
            print(paste("no visc",x,", melt rate ",m," at time ",time,sep=""))            
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
        n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
        time=double(),norm_time=double())#,stringsAsFactors=FALSE)
    } else if (var_is_def) {
        df_m <- data.frame(def_rate=double(),melt_rate=factor(levels=melt_rate_list),true_m_rate=double(),
        B_20=double(),B_21=double(),B_C=double(),B_22=double(),
        n_I=double(),n_Y=double(),n_X=double(),n_B=double(),n_L=double(),
        time=double(),norm_time=double())#,stringsAsFactors=FALSE)
    }


for (x_var in x_variable) {
    for (melt in melt_rate_list) {
        df_m <- build_branch_df(x_var,melt,time)
    }
} 

# branches/lines
df_m["n_B_n_L"] <- df_m$n_B/df_m$n_L
df_m["C_L"] <- 2*(df_m$n_Y+df_m$n_X)/df_m$n_L

df_m
# warnings()

# plot + save plot

# time_string<-sprintf("%0.1e",time)  # to build png file, convert time to a format like 1e+06
# time_string<-gsub("[+]","",time_string)  #  gsub(ch,new_ch, string) to remove '+'
# print(time_string)

# time_string = as.character(time/1e7)
time_string <- sprintf("%02i",time/1e6)  # pad with zeros until string is 2 characters long
time_string <- paste(time_string,"e6",sep="")

if (FALSE) {
    write.csv(df_m, paste("p03_branches_df_",time_string,".csv",sep=""), row.names=FALSE)
}

## -------------------- Line plots ------------------------
if (FALSE) {
    png_name <- paste(base_path,"/branch_plots/br_B_t",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p_b = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    print(p_b)
    dev.off()

    # make separate plots for each melt rate
    png_name <- paste(base_path,"/branch_plots/br_B_sep_t",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    ps = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    print(ps)
    dev.off()

    ## n_I, n_Y, n_X
    png_name <- paste(base_path,"/branch_plots/br_n_t",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_I)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_Y)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_X)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    pn = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    print(pn)
    dev.off()




    png_name <- paste(base_path,"/branch_plots/br_bl_t",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B_n_L)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=C_L)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    print(p1 + p2 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5)))
    #print(pbl)
    dev.off()
}


#  -------------------  ternary plots -------------------

# import the library to create ternary plots
library("ggplot2")
library("ggtern")
if (FALSE) {
    # # connected-connected - isolated-connected - isolated-isolated
    # df_m["P_I"] <- df_m$N_I/(df_m$N_I+3*df_m$N_Y+4*df_m$N_X)   # probability of an I node
    # df_m["P_C"] <- (3*df_m$N_Y+4*df_m$N_X)/(df_m$N_I+3*df_m$N_Y+4*df_m$N_X)   # probability of a C node
    # df_m["P_II"] <- (df_m$P_I)^2  # probability of a branch with 2 I nodes if random distr
    # df_m["P_IC"] <- (df_m$P_I)*(df_m$P_C)  # probability of a branch with 1 I node and 1 C node if random distr
    # df_m["P_CC"] <- (df_m$P_C)^2  # probability of a branch with 2 C nodes if random distr
    # colour by melt rate
    df_m$melt_rate_factor <- factor(df_m$melt_rate, ordered = TRUE)
    # pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(fill=as.factor(true_m_rate)),shape = 21,stroke=2,size=2,colour="black")+ 
    pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=as.factor(true_m_rate)))+ 
    scale_colour_brewer(palette='Blues')+
    scale_fill_distiller(direction=+1)+
    scale_fill_discrete(guide = guide_legend(reverse=TRUE))+    
    labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Melt Rate")+   # labels for the vertices
    guides(color = guide_legend(reverse=TRUE))+    # low at the bottom, high at the top
    theme(plot.background = element_rect(fill='transparent', color=NA),
        #panel.grid.major = element_line(linetype = "dotted",colour = "black"),
        legend.background = element_rect(fill='transparent'),
        panel.background = element_rect(fill = "#e6dbd5"),
        legend.key = element_rect(fill = "#e6dbd5"),
        legend.position = c(.85, .65))#,alpha=0.8))
    ggsave(paste(base_path,"/branch_plots/br_ter_melt_t",time_string,".png",sep=''), pt1, bg='transparent')

    if (var_is_visc) {
        df_m$visc_factor <- factor(df_m$viscosity, ordered = TRUE)
        ptv <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X)) + 
        geom_point(aes(color = visc_factor)) + 
        scale_colour_brewer(palette='Blues')+
        scale_fill_distiller(direction=+1)+
        scale_fill_discrete(guide = guide_legend(reverse=TRUE))+
        labs(x = expression('N'[Y]),y = expression('N'[I]),z = expression('N'[X]),colour = "Viscosity")+
        guides(color = guide_legend(reverse=TRUE)) +    # low at the bottom, high at the top
        theme(plot.background = element_rect(fill='transparent', color=NA),
        legend.background = element_rect(fill='transparent'),
        panel.background = element_rect(fill = "#e6dbd5"),
        legend.key = element_rect(fill = "#e6dbd5"),
        legend.position = c(.85, .65))#,alpha=0.8))
        ggsave(paste(base_path,"/branch_plots/br_ter_visc_t",time_string,"_trsp.png",sep=''), ptv, bg='transparent')
    } else if (var_is_def) {
        df_m$def_rate_factor <- factor(df_m$def_rate, ordered = TRUE)
        ptv <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+
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
        ggsave(paste(base_path,"/branch_plots/br_ter_def_t",time_string,"_trsp_1.png",sep=""), ptv, bg='transparent')
        # ggsave(paste(base_path,"/branch_plots/br_ter_def_t",time_string,"_trsp_1.png",sep=''), ptv, bg='transparent')
    }
}

#png_name <- paste(base_path,"/branch_plots/br_ter_meltPanels_t",time_string,"e07.png",sep='')  # build name of png
#png(file=png_name,width = 1400,height = 1400,res=200)
#p <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color = melt_rate))
# p + facet_grid(rows = vars(melt_rate),cols = vars(viscosity))
#pt3 <- p + facet_grid(cols = vars(viscosity))
#print(pt3)
#dev.off()

# --------------------- heatmaps --------------------
# heatmap for B20,B21,B_C,B22
if (FALSE) {
    #png_name <- paste(base_path,"/branch_plots/br_heat_B_",time_string,".png",sep='')  # build name of png
    #png(file=png_name,width = 1400,height = 1400,res=200)
    if (var_is_visc){
        p1 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_20)) + scale_fill_distiller(direction=+1) + geom_tile() # scale_fill_distiller's default is -1, which means higher values = lighter
        p2 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_21)) + scale_fill_distiller(direction=+1) + geom_tile() 
        p3 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_C)) + scale_fill_distiller(direction=+1) + geom_tile() 
        p4 <- ggplot(data=df_m,aes(factor(x=viscosity),as.factor(true_m_rate),fill=B_22)) + scale_fill_distiller(direction=+1) + geom_tile() 
    } else if (var_is_def){
            p1 <- ggplot(data=df_m,aes(factor(x=def_rate),melt_rate,fill=B_20)) + scale_fill_distiller(direction=+1) + geom_tile() # scale_fill_distiller's default is -1, which means higher values = lighter
            p2 <- ggplot(data=df_m,aes(factor(x=def_rate),melt_rate,fill=B_21)) + scale_fill_distiller(direction=+1) + geom_tile() 
            p3 <- ggplot(data=df_m,aes(factor(x=def_rate),melt_rate,fill=B_C)) + scale_fill_distiller(direction=+1) + geom_tile() 
            p4 <- ggplot(data=df_m,aes(factor(x=def_rate),melt_rate,fill=B_22)) + scale_fill_distiller(direction=+1) + geom_tile() 
    }
    phm = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5)) 


    p4<-p4+theme(plot.background = element_rect(fill='transparent', color=NA),
    legend.background = element_rect(fill='transparent'))+
    if (var_is_visc){
       labs(x = "Viscosity",y = "Melt Rate",fill = "Dimensionless \nintensity")
    }
    if (var_is_def){
        labs(x = "Deformation Rate",y = "Melt Rate",fill = "Dimensionless \nintensity")
    }
    # ggsave(paste(base_path,"/branch_plots/br_heat_B_",time_string,"_trsp.png",sep=''), phm, bg='transparent',dpi =100)
    ggsave(paste(base_path,"/branch_plots/br_heat_B22_",time_string,".png",sep=''),p4,bg='transparent',dpi =200)
    # p + facet_grid(rows = vars(melt_rate),cols = vars(viscosity))
    #print(phm)
    #dev.off()
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
    png_name <- paste(base_path,"/branch_plots/br_heat_B_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 2800,height = 1800,res=100)
    if (var_is_visc){
        p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),factor(true_m_rate), fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  expression('B'[20]))+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

        p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),factor(true_m_rate), fill=B_21))  + scale_fill_distiller(direction = +1,palette = 'PuBuGn')+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  expression('B'[21]))+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

        p_heat3 <- ggplot(df_m,aes(factor(x=viscosity),factor(true_m_rate), fill=B_C))  + scale_fill_distiller(direction = +1,palette = 'GnBu')+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  expression('B'[C]))+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

        p_heat4 <- ggplot(df_m,aes(factor(x=viscosity),factor(true_m_rate), fill=B_22))  + scale_fill_distiller(direction = +1,,palette = 'PuBu')+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))+
        labs(x = "Viscosity",y = "Melt Rate",fill =  expression('B'[22]))+
        scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))+
        theme(axis.text.x=element_text(size=12))

    } else if (var_is_def){
        p_heat1 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
        p_heat2 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_21))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
        p_heat3 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_C))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
        p_heat4 <- ggplot(df_m,aes(factor(x=def_rate),true_m_rate, fill=B_22))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
    }
    print(paste("min B20 =",min(df_m$B_20),"max = ",max(df_m$B_20)))
    # melt rate v B   lineplots
    if (var_is_visc){
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_20)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm1 <- pm1 + labs(x ="Melt Rate",y =  expression('B'[20]),colour = "Viscosity")
        pm1 <- pm1  + plot_options + scale_colour_brewer(palette='Spectral')
        # scale_y_discrete(breaks =seq(200,500, by = 100)) +  # , expand = c(0, 0)
        # coord_fixed(xlim = c(min(df_m$true_m_rate), max(df_m$true_m_rate)),ylim = c(min(df_m$B_20), max(df_m$B_20)))+
        # theme(panel.grid.major.y = element_line(color = "#8ccde3",linewidth = 0.5,linetype = 2))

        # pm1 <- pm1 + theme(axis.title.y=element_blank(),axis.text.y=element_blank())
        # pm1 <- pm1 + coord_flip()  # rotate data by 90 degrees
        # set axes: expand is to stop extra space around the plot area (tight around the data)
        # BUT I need to manually set 'limits', to have the grid 
        # pm1 <- pm1 + scale_x_discrete(breaks = seq(min(df_m$B_20), max(df_m$B_20), by = 10),expand = c(0, 0)) +   #  limits=factor(as.double(melt_rate_list)/1000)  didn't work
        # scale_y_discrete(breaks = seq(min(df_m$B_20), max(df_m$B_20), by = 10),expand = c(0, 0)) 

        pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # pm2 + theme_bw()
        pm2 <- pm2 + plot_options + scale_colour_brewer(palette='Spectral')# + theme(axis.title.y=element_blank())
        # pm2 <- pm2 + coord_flip()+  # rotate data by 90 degrees
        pm2 <- pm2 +labs(x = "Melt Rate",y = expression('B'[21]),colour = "Viscosity")#+
        # scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))

        pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # pm3 + theme_bw()
        pm3 <- pm3 + plot_options + scale_colour_brewer(palette='Spectral') #+ theme(axis.title.y=element_blank())
        # pm3 <- pm3 + coord_flip()+  # rotate data by 90 degrees
        pm3 <- pm3 +labs(x = "Melt Rate",y = expression('B'[C]),colour = "Viscosity")   #  labels as they'd be before flipping
        #scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))
        
        pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        # pm4 + theme_bw()
        pm4 <- pm4 + plot_options + scale_colour_brewer(palette='Spectral') #+ theme(axis.title.y=element_blank())
        # pm4 <- pm4 + coord_flip()+  # rotate data by 90 degrees
        pm4 <- pm4 +labs(x = "Melt Rate",y = expression('B'[22]),colour = "Viscosity")
        #scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0))
        #p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    } else if (var_is_def){
        pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_20)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
        pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=def_rate)))+ geom_line(aes(color = factor(x=def_rate)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
        #p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    }

    # viscosity v B   lineplots
    if (var_is_visc){
        pv1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10')+#,expand = c(0, 0)) +
        labs(x = "Viscosity",y = expression('B'[20]),colour = "Melt Rate")
        pv1 <- pv1 + plot_options 

        pv2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10',expand = c(0, 0))+
        labs(x = "Viscosity",y = expression('B'[21]),colour = "Melt Rate")
        pv2 <- pv2 + plot_options 

        pv3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10',expand = c(0, 0))+
        labs(x = "Viscosity",y = expression('B'[C]),colour = "Melt Rate")
        pv3 <- pv3 + plot_options 

        pv4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10',expand = c(0, 0))+
        labs(x = "Viscosity",y = expression('B'[22]),colour = "Melt Rate")
        pv4 <- pv4 + plot_options 

    } else if (var_is_def){
        pv1 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_20)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10') 
        pv2 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_21)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10')
        pv3 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_C)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10')
        pv4 <- ggplot(data=df_m,mapping = aes(x=def_rate,y=B_22)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate)) + scale_x_continuous(trans='log10')
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
                c(3,3, 1,1,   NA,  5,5,  2,2),
                c(3,3, 4,4,   NA,  5,5,  6,6),
                c(NA,NA,NA,NA,NA,NA,NA,NA,NA),
                c(9,9, 7,7,   NA,  11,11,8,8),
                c(9,9,10,10,  NA,  11,11,12,12))
    }

    pg <- grid.arrange(pv1,pv2,p_heat1,pm1,p_heat2,pm2,pv3,pv4,p_heat3,pm3,p_heat4,pm4, layout_matrix=hlay)
    print(pg)
    dev.off()
}



#  ONE HEATMAP at a time, combined with its lineplots
if(FALSE) {
    library(cowplot)
    png_name <- paste(base_path,"/branch_plots/br_heat_B20_",time_string,".png",sep='')  # build name of png
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
    png_name <- paste(base_path,"/branch_plots/br_heat_nb_cl_",time_string,".png",sep='')  # build name of png
    png(file=png_name,width = 2800,height = 2800,res=100)
    p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=n_B_n_L))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
    p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=C_L))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))

    # plot melt rate
    pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=n_B_n_L)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
    pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=C_L)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')

    # plot viscosity
    pv1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B_n_L)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
    pv2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=C_L)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')

    # define the layout
    hlay <- rbind(c(1,1,1,NA,NA, 2,2,2, NA,NA),
                c(3,3,3, 4,4,  5,5,5,  6,6),
                c(3,3,3, 4,4,  5,5,5,  6,6))

    pg <- grid.arrange(pv1,pv2,p_heat1,pm1,p_heat2,pm2,layout_matrix=hlay)
    print(pg)
    dev.off()
}
