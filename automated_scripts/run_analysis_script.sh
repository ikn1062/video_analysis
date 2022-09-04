#!/bin/bash
#SBATCH -A b1020
#SBATCH --partition=b1020
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name="job_submission"
#SBATCH --output=data/quest/quest_output/output.%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ishaannarain2022@u.northwestern.edu

module purge all
module load python-miniconda3
module load tesseract
source activate dlib-py39

while read -r line; do
    IFS=, read -r arg1 arg2 <<< "$line"
    python3 quest_run.py "$arg1" "$arg2"
done < input.txt