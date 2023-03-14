#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=48:00:00

#Request some memory per core
#$ -l h_vmem=32G

#$ -t 20-100:10

set -e
set -o pipefail

dirName=size

# submits each simulation with a different timestep, always keeping the same amount of pressure/time constant
# pressure is added at every timestep. Smaller tsteps mean smaller pressure.

#   -> imagine there is a foor loop that increments the task array variable here
# for every SGE_TASK_ID:
  #    --------------- name of directory ---------------
if [[ $SGE_TASK_ID -lt 100 ]]; then
     #numberDir="${dirName}_000$SGE_TASK_ID"
     numberDir="${dirName}0${SGE_TASK_ID}"
    else
        numberDir="${dirName}$SGE_TASK_ID"
fi

# if the directory doesn't exist, make it
if [ ! -d ${numberDir} ]; then
    mkdir ${numberDir};
fi

# get into the directory
cd ${numberDir};

# get the files that we need to run the simulations
cp ../baseFiles/* .

LATTEPATH=/home/home01/scgf/elle_latte_melt/elle/processes/latte/
#cp $LATTEPATH/lattice.cc .; cp $LATTEPATH/experiment.cc .; cp $LATTEPATH/fluid_lattice.cc .; cp $LATTEPATH/particle.cc .

./clearAll || true

# from SGE_TASK_ID to the real number:
#tstep=$(echo "${SGE_TASK_ID}.0/100.0"|bc -l)
#tstep=$(echo "${SGE_TASK_ID}")
ymin=$(echo "200-2000/$SGE_TASK_ID"|bc -l)

# replace the temporary string with the true value
sed -i -e "s/replMeltYmin/${ymin}/g" input.txt
sed -i -e "s/replScale/${SGE_TASK_ID}/g" input.txt


# calculate the frequency, so it saves after the same amount of time passed, no matter the timestep
# sed replaces 'e' with 'times ten to the power'
# cut any decimal (including the point)
#freq=$(echo "10000000/(${tstep})"| sed 's/e/*10^/g' | bc -l | cut -f1 -d".")

# ----------------------------------------------------------------------
# set options for running
expEnd=100000000                  # when to stop the simulation (timestep number)
saveCsvInterval=10        # how often .csv output files are saved
# ----------------------------------------------------------------------

# run latte                  
./elle_latte_b -i res400.elle -n -u 27 $saveCsvInterval -s $expEnd -f 1000000000 >& latte.log;