## copy all baseFiles dirs only

> rsync -vr --exclude={*.sh.e*,*.sh.o*,*.csv,*.png,brok*,*.log,vis*,out.*} ../prod/p01/ th01/  


## clean conda only in my personal/home directory 
> CONDA_PKGS_DIRS=$HOME/.conda/pkgs conda clean -a

## sort directories by size, include hidden dirs
> du -h -d 0 .[^.]* * | sort -h

## clean up vs code:
> Remote-SSH: Uninstall VS Code Server from Host.... 
rm -rf $HOME/.vscode-server

## activate environment in batch job submission script:
> eval "$(conda shell.bash hook)"
conda activate extr_py39

## sub all the sub_pvbatch.sh scripts in the subdirs
> for v in vis*; do cp paravParam.py $v; cd $v; qsub sub_pvbatch.sh; cd ..; done

## find csv files that are too small (contain NaNs)   CHECK SIZE - there are new variables being saved
> find . -type f -name "my*.csv" -size -9900k -delete 