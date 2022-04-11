#!/bin/bash
# load python3
if [[ $slave == 'daint' ]]; then 
	module load daint-gpu cray-python
elif [[ $slave == 'tsa' ]]; then 
	module load python/3.7.4
fi


if [[ $slave == 'daint' ]]; then 
	export EASYBUILD_PREFIX=/project/g110/pyvis
	module load EasyBuild-custom
	eb GEOS-3.10.2-CrayGNU-21.09-python3.eb -r
	eb Eigen-3.4.0-CrayGNU-21.09.eb -r
	eb ecCodes-2.23.0-fosscuda-2019b-python3.eb -r
   	module load GEOS Eigen PROJ Boost GSL
elif [[ $slave == 'tsa' ]]; then 
	# export EASYBUILD_PREFIX=/project/g110/pyvis
	# module load EasyBuild-custom
	# eb ecCodes-2.23.0-fosscuda-2019b-python3.eb -r

	module use /apps/common/UES/sandbox/kraushm/tsa-PROJ/modules/all
	module load PrgEnv-gnu
	module load proj/8.0.0-fosscuda-2019b
	module load geos
fi

VENV_PATH=/project/g110/pyvis/venv_$slave

rm -rf $VENV_PATH
mkdir -p ${VENV_PATH}


python3 -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

pip install --upgrade pip
pip install -r env/requirements.txt

if [[ $slave == 'tsa' ]]; then
    module load python
fi

source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh
eccodes=`spack location -i eccodes@2.19.0%gcc@8.3.0+build_shared_libs`
pip3 install --install-option="--prefix=${eccodes}" eccodes

if [[ $slave == 'daint' ]]; then
    pip install git+https://github.com/psyplot/psy-transect
fi
