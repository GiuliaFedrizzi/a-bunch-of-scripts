#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=48:00:00

#Request some memory per core
#$ -l h_vmem=2G

#$ -t 1-4

set -e
set -o pipefail

dirName=tstep

# submits each simulation with a different timestep, always keeping the same amount of pressure/time constant
# pressure is added at every timestep. Smaller tsteps mean smaller pressure.
# version for gaussScaleFixFrac2/tstsep/t_res200 or t_res400

#   -> imagine there is a foor loop that increments the task array variable here
# for every SGE_TASK_ID:
  #    --------------- name of directory ---------------
if [[ $SGE_TASK_ID -lt 10 ]]; then
     #numberDir="${dirName}_000$SGE_TASK_ID"
     numberDir="${dirName}_${SGE_TASK_ID}_1e${SGE_TASK_ID}"
    elif [[ $SGE_TASK_ID -lt 100 ]]; then
        numberDir="${dirName}_00$SGE_TASK_ID"
    elif [[ $SGE_TASK_ID -lt 1000 ]]; then
        numberDir="${dirName}_0$SGE_TASK_ID"
    else
        numberDir="${dirName}_$SGE_TASK_ID"
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
cp $LATTEPATH/lattice.cc .; cp $LATTEPATH/experiment.cc .; cp $LATTEPATH/fluid_lattice.cc .; cp $LATTEPATH/particle.cc .

./clearAll || true

# from SGE_TASK_ID to the real number:
#tstep=$(echo "${SGE_TASK_ID}.0/100.0"|bc -l)
tstep=$(echo "1e${SGE_TASK_ID}")
#tstep=$(echo "${SGE_TASK_ID}")

# how much pressure to add each tstep
# multiplied by 10 because I add 10^4 every 10^3
pincr=$(echo "${tstep}*10"| sed 's/e/*10^/g' | bc -l | cut -f1 -d".")

# replace the temporary string with the true value
sed -i -e "s/replTstep/${tstep}/g" input.txt

sed -i -e "s/replPincr/${pincr}/g" input.txt

# calculate the frequency, so it saves after the same amount of time passed, no matter the timestep
# sed replaces 'e' with 'times ten to the power'
# cut any decimal (including the point)
freq=$(echo "10000/(${tstep})"| sed 's/e/*10^/g' | bc -l | cut -f1 -d".")

# ----------------------------------------------------------------------
# set options for running
expEnd=100000000                  # when to stop the simulation (timestep number)
saveCsvInterval=${freq}        # how often .csv output files are saved
# ----------------------------------------------------------------------

# run latte                  
./elle_latte_b -i res200.elle -n -u 27 $saveCsvInterval -s $expEnd -f 1000000000 >& latte.log;