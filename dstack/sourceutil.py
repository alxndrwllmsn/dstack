"""Utility functions for handeling sources and in broader context some hack for 
sorce finding and characterisation.

NOTE some quick & dirty code is used to deal with SoFiA in an automated way using
templetes... this is not as sophisticated as the interctions with imager...

For now, I have the code dealing with SoFiA parameter files here!
"""

__all__ = ['skycords_from_ra_dec_list', 'get_2D_pixel_coords_from_skycoords',
		'get_spectral_pixel_coords_from_freq','get_3D_pixel_positions_from_skycoord_and_freq', 
		'get_region_coords_from_skycoord_and_freq', 'create_SoFiA_par_from_template',
		'get_ID_and_pos_list_from_input_catalog']

import numpy as np
import logging
import os

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u

from spectral_cube import SpectralCube #to handle spectral axis

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def get_ID_and_pos_list_from_input_catalog(catalog_path):
	"""Simple routine to generate a list of source ID (name) ra and dec values from
	a simple text file formatted as:
	
	#ID ra dec freq
	ID_1 RA_1 Dec_1 freq_1
	...
	ID_N RA_N Dec_N freq_N

	This catalog serves as the input for most of the code used by `dstack` for
	targeted source finding

	The units of the catalog values should be:

	[-, deg, deg, Hz]

	Parameters
	==========
	catalog_path: str
		Full path to the input catalog

	Returns
	=======
	ID_list: list of strings
		A list containing the source ID values

	ra_list: list of float
		A list containing the source RA values

	dec_list: list of float
		A list containing the source Dec values

	freq_list: list of float
		A list containing the source freq values

	"""
	#Load file but only ID's which are treathed as string
	ID_list = np.loadtxt(catalog_path,skiprows=1,usecols=(0),dtype=str)

	#Load file and ignore the header and ID's
	source_catalog_data = np.loadtxt(catalog_path,skiprows=1,usecols=(1,2,3),dtype=float)

	ra_list = source_catalog_data[:,0]
	dec_list = source_catalog_data[:,1]
	freq_list = source_catalog_data[:,2]

	return ID_list, ra_list, dec_list, freq_list

def task_ID_and_source_ID_dict(ID_list):
	"""Simple routine to generate a dictionary generating a task ID for processing
	purposes to a source ID
	"""

	source_mapping_dir = {}

	for i in range(0,np.size(ID_list)):
		#source_mapping_dir[str(i)] = ID_list[i]
		source_mapping_dir[ID_list[i]] = i

	return source_mapping_dir

def skycords_from_ra_dec_list(ra_list,dec_list,frame='icrs'):
	"""Simple code to convert a list of RA and Dec values provided in degrees to
	astropy sky coordinates
	"""
	return SkyCoord(ra_list, dec_list, unit="deg", frame=frame)

def get_2D_pixel_coords_from_skycoords(fitspath, skycoord_list):
	"""Routine to get the Ra and Dec pixel coords of target sources in an image

	NOTE: there might be a pixel offset as the code is set to have 0 based indexing
	(i.e. numpy) rather than 1-based (i.e. fits standard). For most usecases this
	should not be a problem.

    Parameters
    ==========
    fitspath: str
        The input fits path

	skycoord_list: list of <astropy.SkyCoord.skycoord>
		A list containing the astropy skycoordinates of the sources of interest

    Returns
    =======
    ra_px_list: list of int
    	The list containing the RA pixel coordinates of the sources in the input image

    dec_px_list: list of int
    	The list containing the Dec pixel coordinates of the sources in the input image

	"""
	#Get the wcs info from the fits image of interest
	hdul = fits.open(fitspath) #Primary hdu list
	primary_table = hdul['PRIMARY']
	primary_header = primary_table.header

	fits_wcs = wcs.WCS(primary_header)

	hdul.close()


	#Get the 2D pixel values from the input skycoords
	#for source_coord in skycoord_list:
	ra_px_list, dec_px_list = wcs.utils.skycoord_to_pixel(coords=skycoord_list,
											wcs=fits_wcs,
											origin=0, #Fits standard is 1, but for
											#masking numpy conversion (i.e. 0) is needed
											mode='all')

	#Check if the coordinates makes sense...

	#Get the cube frame for checking
	cube_frame_dict = ds.fitsutil.get_fits_cube_params(fitspath)

	if any(ra_px_list) < 0 or any(ra_px_list) > cube_frame_dict['RA---SIN'][0]:
		raise ValueError('At least one of the input sources \
is not on the input image (RA)!')

	if any(dec_px_list) < 0 or any(dec_px_list) > cube_frame_dict['DEC--SIN'][0]:
		raise ValueError('At least one of the input sources \
is not on the input image (Dec)!')

	#Round to the nearest integer
	ra_px_list = np.round(ra_px_list)
	dec_px_list = np.round(dec_px_list)

	return ra_px_list, dec_px_list

def get_spectral_pixel_coords_from_freq(fitspath, freq_list):
	"""Routine to get the spectral pixel coordinate for sources of interest.

	NOTE that I use a 'non-standard' package (`spectral-cube') instead of 
	canonical astropy to speed up things.

	The indexing of the celestial sphere and frequency coordinates are separated,
	because the former is easy to perform using astropy, while the latter can be
	more tricky due to the possible different cordinate frames (e.g. frequency, velocity)

	That is there will be room for handling data with different spectral axis input

	NOTE the code only works for image cubes with a  frequency spectral axis, with
	units given in Hz!

	NOTE no check for the spectral coordinate frame (e.g. bary, lsrk) is done

	TO DO: handle different spectral coordinate frames and also implement support
	for cubes with velocity (radio, optical) spectral axis

    Parameters
    ==========
    fitspath: str
        The input fits path

	freq_list: list of float
		A list containing the frequencies (in Hz) of the sources of interest

    Returns
    =======
    sepctral_px_list: list of int
    	The list containing the spectral pixel coordinates of the sources in the
    	input image cube

	"""
	#Check if the spectral axis unit is Hz! (poor mans way)
	cube_frame_dict = ds.fitsutil.get_fits_cube_params(fitspath)

	if cube_frame_dict['FREQ'][4] != 'Hz':
		raise ValueError('The fits sepctral axis is not given in Hz!')

	scube = SpectralCube.read(fitspath) 

	spectral_axis = scube.spectral_axis.value #The axis values in Hz

	#Get the closest index
	spectral_px_list = []

	for f in freq_list:
		spectral_px_list.append((np.abs(spectral_axis - f)).argmin())

	#Check if the spectral index is within the image cube

	if any(np.array(spectral_px_list)) < 0 or \
		any(np.array(spectral_px_list)) > np.size(spectral_axis):
		raise ValueError('At least one of the input sources \
is not on the input image (frequency)!')

	return np.array(spectral_px_list)

def get_3D_pixel_positions_from_skycoord_and_freq(fitspath, skycoord_list, freq_list):
	"""
	"""
	#Check for imput data sizes
	if np.size(skycoord_list) != np.size(freq_list):
		raise ValueError('Inconsistent sizes of input spectral ans sparial coordinate lists!')

	#Get the Ra and Dec lists
	ra_px_list, dec_px_list = ds.sourceutil.get_2D_pixel_coords_from_skycoords(fitspath,
										skycoord_list)

	#Get the spectral index
	f_px_list = ds.sourceutil.get_spectral_pixel_coords_from_freq(fitspath,freq_list)	


	#The polarisation is expected to be a single (intensity) value for spectral cubes
	nu_px_list = np.zeros(np.size(f_px_list),dtype=np.int32)

	#Now create a list of indice touples using the fits axis order of:
	# [pol, freq, RA, Dec]
	
	source_pixel_list = []

	for i in range(0,np.size(f_px_list)):
		source_pixel_list.append((nu_px_list[i],
								f_px_list[i],
								int(ra_px_list[i]),
								int(dec_px_list[i])))

	#Do not convert list to numpy array as it converts the touples to numy arrays as well

	#TO DO: solve this following this guide maybe:
	# https://stackoverflow.com/questions/46569100/list-of-tuples-converted-to-numpy-array

	return source_pixel_list

def get_region_coords_from_skycoord_and_freq(fitspath, skycoord_list, freq_list,
											x_hflood_list=[7],
											y_hflood_list=[7],
											z_hflood_list=[5]):
	"""Since the optimal sub-reginon source-finding of a single source can e done
	by defining a region in SoFiA, rather tan providing a mask with the source
	positions of interest, we need to have the SoFiA compatible sub-region coordinates.

	This function provides the coordinates for each source as a list of strings.

	Similar to `ds.fitsutil.fill_fits_mask_from_pixel_position_list`

	Parameters
    ==========
    fitspath: str
        The input mask fits path

	px_position_list: list of touple
		A list for containing the indices of the pixels used as a mask. The touple
		has to be the same lenght as the fits image dimension.

		NOTE that in general the fits files are indiced as follows:

		[pol, freq, RA, Dec]

		As such, an example input enty of a single masked pixel looks as:

		[(0,0,10,10)]

	x_hflood_list: list of ints
		The list of the *half* flooding range in both x (RA) and y (Dec) directions

		The units are in pixels !

		Either the same size as the `px_position_list` or the first element is used

	y_hflood_list: list of ints
		The list of the *half* flooding range in both x (RA) and y (Dec) directions

		The units are in pixels !

		Either the same size as the `px_position_list` or the first element is used

	z_hflood_list: list of ints, optional
		The list of the *half* flooding width in the spectral (freq) direction,
		working similarly to `xy_hflood_list`

    Returns
    =======
	region_coords_str_list: list of strings
		A list containing the pixel values formatted as a string, that can be 
		passed to SoFiA's `input.region` variable

	"""
	#Get the central source pixel list
	px_position_list = ds.sourceutil.get_3D_pixel_positions_from_skycoord_and_freq(fitspath,
														skycoord_list, freq_list)

	#Now get the cube edge point list

	N_dim = ds.fitsutil.get_fits_Ndim(fitspath)

	#Check for image dim
	if N_dim != 4:
		raise ValueError('Flooding only supports images cubes with 4 dimensions!')


	#Check for flood ranges
	if np.size(x_hflood_list) != np.size(px_position_list):
		x_hflood_range = np.multiply(np.ones(np.size(px_position_list)),x_hflood_list)
	else:
		x_hflood_range = np.array(x_hflood_list)

	if np.size(y_hflood_list) != np.size(px_position_list):
		y_hflood_range = np.multiply(np.ones(np.size(px_position_list)),y_hflood_list)
	else:
		y_hflood_range = np.array(y_hflood_list)

	#NOTE that the two ranges treathed differently!
	if np.size(z_hflood_list) != np.size(px_position_list):
		z_hflood_range = np.multiply(np.ones(np.size(px_position_list)),z_hflood_list)
	else:
		z_hflood_range = np.array(z_hflood_list)

	#Get the list for the coordinates as a string
	region_coords_str_list = []

	for i, pos in zip(range(0, len(px_position_list)),px_position_list):
		region_coords_str_list.append('{0:d},{1:d},{2:d},{3:d},{4:d},{5:d}'.format(
							int(pos[2]-x_hflood_range[i]),
							int(pos[2]+x_hflood_range[i]),
							int(pos[3]-y_hflood_range[i]),
							int(pos[3]+y_hflood_range[i]),
							int(pos[1]-z_hflood_range[i]),
							int(pos[1]+z_hflood_range[i])))

	return region_coords_str_list

def create_SoFiA_par_from_template(template_path,
									out_par_path,
									data_path,
									region_string,
									output_fname,
									output_dir):
	"""A quick and dirty hack to create SoFiA parset files for single-source,
	targeted source-finding. The idea is that from a template parset file a new
	one is created with only the envinromental and region parameters changed.

	NOTE that the template parset (i.e. smoothing windows) and the region size
	compatibility is up to the user. This is simply a wrapper.

	NOTE: only working for ASCII files not UTF-8 !

	Parameters
	==========
	template_path: str
		Full path to the template SoFiA parset file. The file should contain
		the following lines:

		input.data\t= \n
		input.region\t= \n
		output.directory\t= \n
		output.filename\t= \n

		These lines will be overwritten by the code based on the other aguments
		of this function.

	out_par_path: str
		The full path to the output parset file created

	data_path: str
		Full path to the input fits cube. Substituted into the line:

		input.data\t= \n

	region_string: str
		A string of the region edges in pixels It should be given as:

		"x_min, x_max, y_min, y_max, z_min, z_max"

		Substituted into the line:

		input.region\t= \n	

	output_fname: str
		The filename ID string used by SoFiA. Substituted into the line:

		output.filename\t= \n

	output_dir: str
		Full path to the output directory, *withouth* the trailing '/' character.
		Substituted into the line:

		output.filename\t= \n

	Returns
	=======
	Create the SoFiA template file `out_par_path`

	"""

	#Reading the template into a list of line strings
	with open(template_path,'r') as template_par:
		template_par_lines = template_par.readlines()

	template_par.close()


	with open(out_par_path, 'w') as out_par:
		for l in template_par_lines:
			#Replace specific lines

			#NOTE that this is a really slow implementation... but the parset files
			#are expected to be small so this slow part should not be noticable

			#Adding a line for the input data
			if 'input.data' in l:
				out_par.write('input.data\t= {0:s}\t#Added by dstack'.format(
							data_path) + os.linesep)

				continue

			#Adding a line for the input region
			if 'input.region' in l:
				out_par.write('input.region\t= {0:s}\t#Added by dstack'.format(
							region_string) + os.linesep)

				continue

			#Adding a line for the output directory
			if 'output.directory' in l:
				out_par.write('output.directory\t= {0:s}/\t#Added by dstack'.format(
							output_dir) + os.linesep)

				continue


			#Adding a line for the output file string
			if 'output.filename' in l:
				out_par.write('output.filename\t= {0:s}\t#Added by dstack'.format(
							output_fname) + os.linesep)

				continue

			else:
				out_par.write(l)

	out_par.close()




#=== MAIN ===
if __name__ == "__main__":
    pass