#!/bin/bash
nChannels=15
increment=1
# Loop through channels
for (( ch=1; ch<=$nChannels; ch=$ch+1 )); do
	echo $ch
	sbatch --job-name="$ch_multidose" --output="logs/multidose_${ch}.out" --error="logs/multidose_${ch}.err" bash_loop_sbatch.bash $ch
	echo "submitted channel $ch"
done
