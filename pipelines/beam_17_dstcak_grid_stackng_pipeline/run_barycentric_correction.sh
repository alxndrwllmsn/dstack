#!/bin/bash

#Submit a job/inputMS to create barycentric corrected MS for the pipleine

#=== Functions ===
remove_folder_if_exist(){
	if [ -d "$1" ]; 
	then 
		remove_folder_and_log $1;
	fi

	mkdir $1;
	echo "New empty directory created at: $1";
}

#=== MAIN ===
WORKING_DIR=$(pwd)
INPUT_DIR='/mnt/hidata2/dingo/pilot/uvgrid'
OUTPUT_DIR='/scratch/krozgonyi/beam17/barycentric_MS'

declare -a MS_LIST=('SB11006/scienceData_SB11006_G23_T0_B_01.beam17_SL.ms'
	'SB11003/scienceData_SB11003_G23_T0_B_02.beam17_SL.ms'
	'SB11000/scienceData_SB11000_G23_T0_B_03.beam17_SL.ms'
	'SB11010/scienceData_SB11010_G23_T0_B_04.beam17_SL.ms'
	'SB10994/scienceData_SB10994_G23_T0_B_05.beam17_SL.ms'
	'SB10991/scienceData_SB10991_G23_T0_B_06.beam17_SL.ms'
	'SB11026/scienceData_SB11026_G23_T0_B_07.beam17_SL.ms')


remove_folder_if_exist ${OUTPUT_DIR}
remove_folder_if_exist "${WORKING_DIR}/barycentric_correction/"

MS_INDEX=0
for MS in "${MS_LIST[@]}"; do	
	SLURM_FILE="${WORKING_DIR}/barycentric_correction/run_barycentric_correction_MS_${MS_INDEX}.sh"

	echo "#!/bin/bash
#SBATCH --job-name=barycentric_correction_MS_${MS_INDEX}.job
#SBATCH --output=${WORKING_DIR}/barycentric_correction/slurm_MS_${MS_INDEX}.out
#SBATCH --time=8:00:00
#SBATCH --mem=180000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rstofi@gmail.com

alias casa='/home/rdodson/Software/Casa/casa-pipeline-release-5.6.2-2.el7/ --log2term --nologger' && ${WORKING_DIR}/create_barycentric_MS.sh -i ${INPUT_DIR}/${MS} -o ${OUTPUT_DIR}/${MS} -l ${WORKING_DIR}/barycentric_correction/cvel_logfile_MS_${MS_INDEX}.log" > "${SLURM_FILE}"

	 cd "${WORKING_DIR}/barycentric_correction" ; sbatch ${SLURM_FILE}

	MS_INDEX=$(expr $MS_INDEX + 1)
done
