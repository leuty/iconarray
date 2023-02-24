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
conda activate icon-timeseries

icon-timeseries meanmax \
    --exp "/store/s83/cmerker/test_data/icon_timeseries/data/104/lfff00*0" "104" \
    --exp "/store/s83/cmerker/test_data/icon_timeseries/data/106/lfff00*0" "106" \
    --varname "TOT_PREC" \
    --deagg \
    --level 0 \
    --gridfile "/store/s83/tsm/ICON_INPUT/icon-1e_dev/ICON-1E_DOM01.nc" \
    --domain "ch"
