#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=05:00:00

#Request some memory per core
#$ -l h_vmem=8G

module load ffmpeg/4.0

dir1=$(echo \"$PWD\" | cut -d "/" -f 5)  # split the path to current directory wherever you find "/"
dir2=$(echo \"$PWD\" | cut -d "/" -f 6)  
dir3=$(echo \"$PWD\" | cut -d "/" -f 7)
dir4=$(echo \"$PWD\" | cut -d "/" -f 8 | cut -d "\"" -f 1)  # this could be the last one, so remove quotation marks
dir5=$(echo \"$PWD\" | cut -d "/" -f 9 | cut -d "\"" -f 1)
vid_name=$(echo ${dir1}"_"${dir2}"_"${dir3}"_"${dir4}"_"${dir5}"_bb.mkv")

ffmpeg -y -framerate 1  -pattern_type glob -i "a_brokenBonds__0*.png" $vid_name

#   find . -name "*.mkv" | sort > videosToCopy.txt 
#   rclone copy -Pv --dry-run --files-from=videosToCopy.txt myExperiments/ onedrive:/PhD/arc/myExperiments/gaussAllTests