# Manual steps
----

For my Thesis, I had no time to properly implement (and test) several steps which is needed for my analysis and for a full scale data reduction. That is, several remaining steps needs to be completed manually after running the pipleine defined in the parent folder.

These steps are the following:
	
1. Run an imaging of the co-added visibilities by using Cotton-Schwaab style deconvolution 
2. Perform primary beam correction on all deep images
3. Convert the output images to fits format
4. Run SoFiA on all deep images transformed

This folder contains all the specific scripts needed and the commands for execution. The order should be as defined above. The steps are:

__I. Baseline imaging:__

The pipeline performs an apples-to-apples comparision, by creating deep images using `cdeconvolver-mpi` and the gridded-coadded visibilities as an input. However, the real baseline solution would be to image the concentrated visibilities by using imager and a Cotton-Schwaab-like deconvolution.

For this we gonna create a directory called `baseline_vis_imaging` in the directory where the `Snakefile` lives. There we place the script `baseline_vis_imaging_parset.in` given here. This should be run either through `slurm` or on _Hyrmine_ using _screen_ as it can be done. Just run the following command (on 25 cores):

	mpirun -np 25 imager -c baseline_vis_imaging_parset.in > ./logfile_baseline_imaging.log
	
... and it is done. The baseline images are there!

__II. Primary vbeam correction:__

For all deep images the PB correction needs to be done. I use `linmos` from YandaSoft to do this. Just put the `linmos_PB_correction.in` file to each respective direcory and run via:

	linmos -c linmos_PB_correction.in >./logfile_linmos_PB.log
	

Both the PB corrected deep images and the corresponding PB models (weight files) should be there!

__III. Convert to fits:__

Simply use the command-line app `cim2fits()` that I made in dstack. It is a wrapper around the same `casacore` task. That is simply run something like:

	cim2fits -i ./PB.deep.restored/ -o ./PB_deep_restored.fits
	cim2fits -i ./PB.deep.weight/ -o ./PB_deep_weight.fits

Boom. All input needed for source-finding is ready!

__IV. Source finding use SoFiA:__

For this, use the parameter file given in this folder for each deep imaging method respectively.