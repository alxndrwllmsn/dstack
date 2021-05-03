# Manual steps
----

For my Thesis, I had no time to properly implement (and test) several steps which is needed for my analysis and for a full scale data reduction. That is, several remaining steps needs to be completed manually after running the pipleine defined in the parent folder.

These steps are the following:

(0.) Measure the RMS on the nonPB corrected images
 	
1. Run an imaging of the co-added visibilities by using Cotton-Schwaab style deconvolution 
2. Convert the output images to fits format 
3. Perform primary beam correction on all deep images
4. Run SoFiA on all deep images transformed

This folder contains all the specific scripts needed and the commands for execution. The order should be as defined above. The steps are:

__0. Measure RMS:__

For the 2km baseline data I measzre the RMS on the central 8'x8' area equals 1+2*40 px^2 this would not really be affected by the PB correction.

I only use the first 1.5 MHz (76 channels) of the cube, for which we expect to measure only the thermal noise.

The following command is used:

	cimRMS -c /path_to_CASAIMAGE_/ -o rms.dat -ad -cmin 0 -cmax 77 -wh 40

Furthermore, by default the RMS is computed using a robust method. Values above/blow the  +/- 99% percentile cut are not counted towards the RMS. (default settings) 

__I. Baseline imaging:__

The pipeline performs an apples-to-apples comparision, by creating deep images using `cdeconvolver-mpi` and the gridded-coadded visibilities as an input. However, the real baseline solution would be to image the concentrated visibilities by using imager and a Cotton-Schwaab-like deconvolution.

For this we gonna create a directory called `baseline_vis_imaging` in the directory where the `Snakefile` lives. There we place the script `baseline_vis_imaging_parset.in` given here. This should be run either through `slurm` or on _Hyrmine_ using _screen_ as it can be done. Just run the following command (on 25 cores):

	mpirun -np 25 imager -c baseline_vis_imaging_parset.in > ./logfile_baseline_imaging.log
	
... and it is done. The baseline images are there!

__II. Convert to fits:__

Simply use the command-line app `cim2fits()` that I made in dstack. It is a wrapper around the same `casacore` task. That is simply run something like:

	cim2fits -i ./image.deep.restored/ -o ./deep_image.fits

Boom. All input needed for PB correction is there: a simple fits file. Interestingly, `linmos` works on a fits file, but fails on `casaimage` input of the same data. Therefore, the conversion to fits needs to be done first. Furthermore, SoFiA works only on fits images as well. This step should create the fits image of the same name for each deep imaging methods, so the next steps can be universal across the different methods.

This step should be straightforward to implement to the pipeline!

_NOTE_ for the baseline imaging, the header of the deconvolved image is different, and `cim2fits()` dies. That is, we need to use casa for converting to fits. However, we need to open cas fits, and inside casa the following command will create the right fits file:

	exportfits('image.restored.deep/','deep_image.fits', stokeslast=False)
	
Done. Also, delete the casa logfile crreated...

__III. Primary bbeam correction:__

For all deep images the PB correction needs to be done. I use `linmos` from YandaSoft to do this. Just put the `linmos_PB_correction.in` file to each respective direcory and run via:

	linmos -c linmos_PB_correction.in >./logfile_linmos_PB.log

Both the PB corrected deep images and the corresponding PB models (weight files) should be there! The cript work universaly on the input file named `deep_image.fits` for all different deep imaging methods.

__IV. Source finding use SoFiA:__

The SoFiA parameter file is given here, however, __the output directory has to be given as a full path, and so for each run it needs to be changed!__ The output directory should be where the PB corrected fits file lives, in the folder `./sofia_output/`. Nothe that this folder needs to be created before running SoFiA! The source-finding can be done by using SoFiA from Tobiases home directory:

	/home/twestmeier/SoFiA-2/sofia runsofia.par

This should create all necessary output files. The last step is to either copy the results to work with them locally, or run some `svalidation` scripts on Pleiades.

