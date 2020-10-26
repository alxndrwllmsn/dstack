#!/bin/bash

#https://casa.nrao.edu/casadocs/casa-5.0.0/uv-manipulation/regridding-visibility-frequencies-and-velocities

EXTRA_ARGS=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--input)
    INPUT_MS="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--output)
    OUTPUT_MS="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--logfile)
    LOGFILE="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    EXTRA_ARGS+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${EXTRA_ARGS[@]}" # restore extra positional parameters

# run CASA crvel() task
source /home/krozgonyi/.bashrc dstack_env_setup ; echo "cvel(vis='${INPUT_MS}', outputvis='${OUTPUT_MS}', passall=False, mode='channel', nchan=-1, start=0, width=1, interpolation='linear', phasecenter='', spw='', restfreq='1420405751.786Hz', outframe='BARY')" | casa --nologger > ${LOGFILE}
