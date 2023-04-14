# ------------------------------------------
# analyse py_branch_info.csv files
# specify name of subdirectories (viscosity/def rate + melt rate)
# plot parameters such as 'Frequency', 'Intensity', 'Dimensionless intensity'
# import and plot number and ratios of types of nodes: I, Y and X

# run in the conda environment "r_env"
#   it contains tidyverse, patchwork
# specify time as command line argument e.g.
# > Rscript $MS/post_processing/image_analysis/branch_analysis.r 6e7 
#
# or submit a task array job with sub_branchAn.sh from visc_*/vis*
#
# Giulia March 2023
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
time = as.numeric(args[1])   # time for the 1st   (e.g. 6e7 = 60e6 = 60th file in mr_01)

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
                df_m <- rbind(df_m,de)#,stringsAsFactors=FALSE)
                # df_m <- rbind(df_m,de,stringsAsFactors=FALSE)
            }
            else {
            print(paste("no visc",x,", melt rate ",m," at time ",time,sep=""))            
            }

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
time_string = sprintf("%02i",time/1e7)  # pad with zeros until string is 2 characters long

if (FALSE) {
    write.csv(df_m, paste("p03_branches_df_",time_string,".csv",sep=""), row.names=FALSE)
}

## -------------------- Line plots ------------------------
if (FALSE) {
    png_name <- paste(base_path,"/branch_plots/br_B_t",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p_b = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    print(p_b)
    dev.off()

    # make separate plots for each melt rate
    png_name <- paste(base_path,"/branch_plots/br_B_sep_t",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') + facet_grid(rows=vars(melt_rate))
    ps = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    print(ps)
    dev.off()

    ## n_I, n_Y, n_X
    png_name <- paste(base_path,"/branch_plots/br_n_t",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_I)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_Y)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_X)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    pn = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    print(pn)
    dev.off()




    png_name <- paste(base_path,"/branch_plots/br_bl_t",time_string,"e07.png",sep='')  # build name of png
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
if (TRUE) {
    # colour by melt rate
    png_name <- paste(base_path,"/branch_plots/br_ter_melt_t",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    pt1 <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X))+ geom_point(aes(color=true_m_rate)) + scale_colour_continuous(trans='reverse') #,size=factor(viscosity,ordered=T,alpha=0.5))) + scale_colour_gradient2(low="#aeadef",high="#04308f")
    print(pt1)
    dev.off()

    # connected-connected - isolated-connected - isolated-isolated
    df_m["P_I"] <- df_m$N_I/(df_m$N_I+3*df_m$N_Y+4*df_m$N_X)   # probability of an I node
    df_m["P_C"] <- (3*df_m$N_Y+4*df_m$N_X)/(df_m$N_I+3*df_m$N_Y+4*df_m$N_X)   # probability of a C node
    df_m["P_II"] <- (df_m$P_I)^2  # probability of a branch with 2 I nodes if random distr
    df_m["P_IC"] <- (df_m$P_I)*(df_m$P_C)  # probability of a branch with 1 I node and 1 C node if random distr
    df_m["P_CC"] <- (df_m$P_C)^2  # probability of a branch with 2 C nodes if random distr

    # png_name <- paste(base_path,"/branch_plots/br_ter_IC_t",time_string,"e07.png",sep='')  # build name of png
    # png(file=png_name,width = 1400,height = 1400,res=200)
    # pt2 <- ggtern(data=df_m,aes(x=P_IC,y=P_II,z=P_CC))+ geom_point(aes(color = factor(viscosity,ordered=T)))
    # print(pt2)
    # dev.off()

    png_name <- paste(base_path,"/branch_plots/br_ter_visc_t",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    ptv <- ggtern(data=df_m,aes(x=n_Y,y=n_I,z=n_X)) + geom_point(aes(color = viscosity)) + scale_colour_continuous(trans='reverse') #+ geom_path()
    print(ptv)
    dev.off()
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
    png_name <- paste(base_path,"/branch_plots/br_heat_B_",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p1 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_20)) + scale_fill_distiller(direction=+1) + geom_tile() # scale_fill_distiller's default is -1, which means higher values = lighter
    p2 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_21)) + scale_fill_distiller(direction=+1) + geom_tile() 
    p3 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_C)) + scale_fill_distiller(direction=+1) + geom_tile() 
    p4 <- ggplot(data=df_m,aes(factor(x=viscosity),melt_rate,fill=B_22)) + scale_fill_distiller(direction=+1) + geom_tile() 
    phm = p1 + p2 + p3 + p4 + plot_annotation(title = paste("time =",time)) & theme(plot.title = element_text(hjust = 0.5))
    # p + facet_grid(rows = vars(melt_rate),cols = vars(viscosity))
    print(phm)
    dev.off()
}


# df_grid <- expand.grid(X=df_m$viscosity)
if (FALSE) {
    ##  four heatmaps per figure
    ## B_20, B_21, B_C, B_22
    png_name <- paste(base_path,"/branch_plots/br_heat_B_",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_21))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat3 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_C))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat4 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_22))  + scale_fill_distiller(direction = +1)+ geom_tile()

    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    phm_B = grid.arrange(p1,p2,p_heat1, p_heat2,p3,p4, p_heat3, p_heat4, ncol=2)
    print(phm_B)
    dev.off()

    ## Number of I,Y,X nodes
    png_name <- paste(base_path,"/branch_plots/br_heat_n_",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=N_I))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=N_Y))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat3 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=n_X))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat4 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=n_B))  + scale_fill_distiller(direction = +1)+ geom_tile()

    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=N_I)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=N_Y)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_X)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    phm_B = grid.arrange(p1,p2,p_heat1, p_heat2,p3,p4, p_heat3, p_heat4, ncol=2)
    print(phm_B)
    dev.off()

    ## n of branches per line, n of connections per line
    png_name <- paste(base_path,"/branch_plots/br_heat_nl_",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 1400,height = 1400,res=200)
    p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=n_B_n_L))  + scale_fill_distiller(direction = +1)+ geom_tile()
    p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=C_L))  + scale_fill_distiller(direction = +1)+ geom_tile()

    p1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B_n_L)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
    p2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=C_L)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    # p3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_X)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    # p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=n_B)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    phm_B = grid.arrange(p1,p2,p_heat1, p_heat2, ncol=2)
    print(phm_B)
    dev.off()
}

if (TRUE) {
    png_name <- paste(base_path,"/branch_plots/br_heat_B_",time_string,"e07.png",sep='')  # build name of png
    png(file=png_name,width = 2800,height = 2800,res=100)
    p_heat1 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_20))  + scale_fill_distiller(direction = +1)+ geom_tile() + theme(legend.key.size = unit(0.5, 'cm'))
    p_heat2 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_21))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
    p_heat3 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_C))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))
    p_heat4 <- ggplot(df_m,aes(factor(x=viscosity),melt_rate, fill=B_22))  + scale_fill_distiller(direction = +1)+ geom_tile()+ theme(legend.key.size = unit(0.5, 'cm'))

    pm1 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_20)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ coord_flip() #,linetype = "dashed") #+ scale_x_continuous(trans='log10') 
    pm2 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_21)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    pm3 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_C)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm')) #+ geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    pm4 <- ggplot(data=df_m,mapping = aes(x=true_m_rate,y=B_22)) + geom_point(aes(color = factor(x=viscosity)))+ geom_line(aes(color = factor(x=viscosity)))+ theme(legend.key.size = unit(0.5, 'cm'))# + geom_line(aes(color = viscosity),linetype = "dashed") + scale_x_continuous(trans='log10')
    #p4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate)) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')

    pv1 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_20)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10') 
    pv2 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_21)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    pv3 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_C)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')
    pv4 <- ggplot(data=df_m,mapping = aes(x=viscosity,y=B_22)) + geom_point(aes(color = melt_rate))+ theme(legend.key.size = unit(0.5, 'cm')) + geom_line(aes(color = melt_rate),linetype = "dashed") + scale_x_continuous(trans='log10')

    # define the layout
    hlay <- rbind(c(1,1,1,NA,NA, 2,2,2, NA,NA),
                c(3,3,3, 4,4,  5,5,5,  6,6),
                c(3,3,3, 4,4,  5,5,5,  6,6),
                c(7,7,7,NA,NA, 8,8,8, NA,NA),
                c(9,9,9,10,10,11,11,11,12,12),
                c(9,9,9,10,10,11,11,11,12,12))

    pg <- grid.arrange(pv1,pv2,p_heat1,pm1,p_heat2,pm2,pv3,pv4,p_heat3,pm3,p_heat4,pm4, layout_matrix=hlay)
    print(pg)
    dev.off()
}

if (TRUE) {
    png_name <- paste(base_path,"/branch_plots/br_heat_nb_cl_",time_string,"e07.png",sep='')  # build name of png
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
# p1 <- ggMarginal(p, type="histogram", size=10)
