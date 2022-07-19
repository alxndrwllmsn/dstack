"""Collection of utility functions to interact with .fits images
"""

__all__ = ['get_synthesiseb_beam_params_from_fits_header']

import numpy as np
import logging

from astropy.io import fits

from astropy.coordinates import SkyCoord
from astropy import units as u

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def get_synthesiseb_beam_params_from_fits_header(fitspath, return_beam='mean'):
	"""This function attempts to read out the synthesised beam parameters from a
	fits header.
	
	These parameters are either in the PRIMARY header as an average value over the
	whole cube, or if the frequency variance is accounted for in the image forming
	process and so in controlling the synthesised beam shape.

	NOTE: the current code assumes the beam parameters being in degrees!

	TO DO: add code to check for units (not necessary existing in the header!)

	A 'better' solution is when the synthesised beam parameters are stored in a
	dedicated 'BEAMS' table for each channel an associated beam.

	NOTE: the code assumes the beam axis to be in units of arcseconds

	NOTE: the current code can only handle athe aformentined two cases.

	TO DO: account for other ways the synthsised beam info could be stored in fits headers

	The code returns the (mean) synthesised beam parameters in units of arcseconds
	for the axes and degrees for the position angle.

    Parameters
    ==========
    fitspath: str
        The input fits path

    return_beam: str, optional
		Only aplicible if the beams are stored as a function of frequency, it controls
		the single beam values returned (i.e. the averaging process). Allowed values
		are ['mean', 'median']        

    Returns
    =======
    b_maj: float
    	The (average) major axis in units of arcseconds

    b_min: float
    	The (average) minor axis in units of arcseconds

    b_pa: float
    	The (average) position angle in units of degrees

	"""
	hdul = fits.open(fitspath) #Primary hdu list

	#Select the 'BEAMS' table if exist
	try:
		log.info('Trying to read beams from the BEAMS table...')
		beams_table = hdul['BEAMS']

		beams_header = beams_table.header

		#Perform a check for the BEAMS table setup
		if beams_header['NAXIS'] != 2:
			raise ValueError('The BEAMS table has more than 1 columns' )
			#NOTE that the second column should be empty

		if beams_header['TTYPE1'] != 'BMAJ':
			raise ValueError('TTYPE1 is not BMAj in the BEAMS table!')
		if beams_header['TTYPE2'] != 'BMIN':
			raise ValueError('TTYPE2 is not BMIN in the BEAMS table!')
		if beams_header['TTYPE3'] != 'BPA':
			raise ValueError('TTYPE3 is not BPA in the BEAMS table!')

		if beams_header['TUNIT1'] != 'arcsec' or beams_header['TUNIT2'] != 'arcsec':
			raise ValueError('Either BMAJ or BMIN units are not in arcseconds!')
			#TO DO apply a conversion in this case
		if beams_header['TUNIT3'] != 'deg':
			raise ValueError('BPA is not in degrees!')

		if beams_header['TTYPE4'] != 'CHAN':
			raise ValueError('TTYPE4 is not CHAN in the BEAMS table!')
		if beams_header['TTYPE5'] != 'POL':
			raise ValueError('TTYPE5 is not POL in the BEAMS table!')

		#Get the Synthesised beam value from the BEAMS table
		beams_data = beams_table.data

		#Check the beam data shape
		if np.size(beams_data[0]) != 5:
			raise ValueError('Wrong BEAMS data shape!')

		allowed_averaging_methods = ['mean', 'median']

		if return_beam not in allowed_averaging_methods:
			raise ValueError('The returned beam specification is not allowed!')

		#Compute the *average* beam shape
		if return_beam == 'mean':
			b_maj = np.mean(beams_data['BMAJ'])
			b_min = np.mean(beams_data['BMIN'])
			b_pa = np.mean(beams_data['BPA'])
		elif return_beam == 'median':
			#The median in CARTA is not the median of all the values,
			#BUT the median channel value => need to add that into this code.
			b_maj = np.median(beams_data['BMAJ'])
			b_min = np.median(beams_data['BMIN'])
			b_pa = np.median(beams_data['BPA'])

		return b_maj, b_min, b_pa

	#Get the bem params from the main header
	except:
		log.info('Trying to get the beams from the PRIMARY table...')
		primary_table = hdul['PRIMARY']

		primary_header = primary_table.header

		#The values are expected to be in degrees, since no units are defined
		#in the example fits...

		b_maj_raw = primary_header['BMAJ']
		b_min_raw = primary_header['BMIN']
		b_pa = primary_header['BPA'] #deg by definition

		#convert b_maj and b_pa to arcsec
		b_maj = ds.miscutil.deg2arcsec(b_maj_raw)
		b_min = ds.miscutil.deg2arcsec(b_min_raw)

		return b_maj, b_min, b_pa

if __name__ == "__main__":
    pass
