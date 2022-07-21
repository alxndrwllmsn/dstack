"""Utility functions for handeling sources and in broader context some hack for 
sorce finding and characterisation.

NOTE some quick & dirty code is used to deal with SoFiA in an automated way using
templetes... this is not as sophisticated as the interctions with imager...
"""

__all__ = []

import numpy as np
import logging

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u

from spectral_cube import SpectralCube #to handle spectral axis

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
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


#=== MAIN ===
if __name__ == "__main__":
    pass