# submit a job that waits for a sequence of jobs to finish.
# give the first and last job ID as a command line argument
# e.g.   ./seq_jobs.sh 5430171 5430177  will wait until all jobs in the
#        range between 5430171 and 5430177 are done, then run the new one


# generate a sequence between the first and second number
# separated by a comma
job_names=$(echo `seq -s, $1 $2`)  

#echo $job_names

# the job will be held in the queue until all $job_names are completed
qsub -hold_jid $job_names hello1.sh
