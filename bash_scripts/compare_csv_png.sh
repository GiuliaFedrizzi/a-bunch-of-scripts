# check if the number of porosity png's is the same as the number of my_experiment files
# if they are not the same, copy paravParam.py there and submit sub_pvbatch.sh

for v in visc_*/vis*; do
  echo $v;
  myexp=$(ls -l $v/my_experiment* | wc -l); poro=$(ls -l $v/a_porosity_* | wc -l);  # save number of files of the 2 types
  #echo $myexp
  #echo $poro
  if [ "$myexp" == "$poro" ]; then
    echo "equal";
  else
    echo "not equal";
    cp paravParam.py $v
    cd $v
    qsub sub_pvbatch.sh
    cd ../..
  fi;
done