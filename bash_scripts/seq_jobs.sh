# submit a job that waits for a sequence of jobs to finish.
# give the first and last job ID as a command line argument
# e.g.   ./seq_jobs.sh 5430171 5430177  will wait until all jobs in the
#        range between 5430171 and 5430177 are done, then run the new one

job_names=$(echo `seq -s, $1 $2`)
#echo $job_names
qsub -hold_jid $job_names hello1.sh
