#!/bin/bash -l

#SBATCH --job-name="icon_timeseries"
#SBATCH --time=00:10:00
#SBATCH --partition=postproc
#SBATCH --output=icon_timeseries.%j.o
#SBATCH --error=icon_timeseries.%j.e
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive

eval "$(conda shell.bash hook)"
conda activate icon_timeseries

icon-timeseries meanmax \
    --exp "/store/s83/tsm/EXP_TST/102/22090612_102/icon/000/lfff00*0" "102" \
    --exp "/store/s83/tsm/EXP_TST/103/22090612_103/icon/000/lfff00*0" "103" \
    --exp "/store/s83/tsm/EXP_TST/104/22090612_104/icon/000/lfff00*0" "104" \
    --exp "/store/s83/tsm/EXP_TST/105/22090612_105/icon/000/lfff00*0" "105" \
    --varname "TOT_PREC" \
    --level 0 \
    --gridfile "/store/s83/tsm/ICON_INPUT/icon-1e_dev/ICON-1E_DOM01.nc" \
    --domain "ch" \
    --dask-workers 6
