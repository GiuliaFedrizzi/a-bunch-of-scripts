#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=48:00:00

#Request some memory per core
#$ -l h_vmem=8G

# multiple cores
#$ -pe smp 8

#module unload paraview
#module load paraview-osmesa

MS=/home/home01/scgf/myscripts
# pvbatch $MS/saveAnimationEveryFolder.py > pvbatchout.log
#pvbatch --mpi --force-offscreen-rendering $MS/saveAnimationEveryFolder.py &> pvbatchout.log
pvbatch --mpi $MS/saveAnimationEveryFolder.py &> pvbatchout.log

# to submit from outside sigma*:
# for s in sigma_*; do cd ${s}/${s}_gaussTime05; for d in tstep04_*; do cd $d; pwd; qsub $MS/sub_pvbatch.sh; cd ..; done; cd ../../; done

# then when I want to copy to rclone:
# rclone copy -P --exclude={*.csv,*.sh.*,core.*} gaussTime/ onedrive:/PhD/arc/myExperiments/gaussTime/ 

