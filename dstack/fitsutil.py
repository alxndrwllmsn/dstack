"""Collection of utility functions to interact with .fits images
"""

__all__ = ['get_synthesiseb_beam_params_from_fits_header', 'get_fits_cube_params',
			'create_empty_fits_mask_from_file', 'get_fits_Ndim', 
			'fill_fits_mask_from_pixel_position_list']

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

	Reding the primary header is the default approach

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

		hdul.close()

		return b_maj, b_min, b_pa

	#Get the bem params from the main header
	except:
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

		hdul.close()

		return b_maj, b_min, b_pa

def get_fits_Ndim(fitspath):
	"""Simple routine to get the dimension of the fits file

	Parameters
    ==========
    fitspath: str
        The input fits path

    Returns
    =======
    N_dim: int
    	The number of dimensions of the fits file

	"""
	hdul = fits.open(fitspath) #Primary hdu list
	primary_table = hdul['PRIMARY']
	primary_header = primary_table.header

	N_dim = int(primary_header['NAXIS'])

	hdul.close()

	return N_dim

def get_fits_cube_params(fitspath):
	"""Function to read out the dimension and axis information to a dictionary

	The axes are read from the PRIMARY header and then cosiders only the following
	axes (if present): 

	['RA--SIN','DEC--SIN','FREQ','STOKES']

	A dictionary is returned, for each axis key the following values in an array:

	[array lenght, reference pixel value, reference pixel index, pixel size, pixel unit]

	NOTE that the units are not converted within this script!
    
    Parameters
    ==========
    fitspath: str
        The input fits path

    Returns
    =======
	cube_params_dict: dict
		A dictionary with all the axis info

	"""
	hdul = fits.open(fitspath) #Primary hdu list
	primary_table = hdul['PRIMARY']
	primary_header = primary_table.header

	N_dim = int(primary_header['NAXIS'])

	axis_name_array = []
	axis_unit_array = []
	axis_reference_val_array = []
	axis_reference_pix_array = []
	axis_increment_array = []
	axis_lenght_array = []

	#Get all the axis name dimension and unit
	for d in range(1,N_dim+1):
		axis_name_array.append(primary_header['CTYPE{0:d}'.format(d)])
		axis_unit_array.append(primary_header['CUNIT{0:d}'.format(d)])
		axis_reference_val_array.append(primary_header['CRVAL{0:d}'.format(d)])
		axis_increment_array.append(primary_header['CDELT{0:d}'.format(d)])
		axis_reference_pix_array.append(int(primary_header['CRPIX{0:d}'.format(d)]))
		axis_lenght_array.append(primary_header['NAXIS{0:d}'.format(d)])

	hdul.close()

	#Need to convert to numpy arrays for the cross-matching to work
	axis_name_array = np.array(axis_name_array)
	axis_lenght_array = np.array(axis_lenght_array)
	axis_reference_val_array = np.array(axis_reference_val_array)
	axis_reference_pix_array = np.array(axis_reference_pix_array)
	axis_increment_array = np.array(axis_increment_array)
	axis_unit_array = np.array(axis_unit_array)

	#Re arrange the arrays to the shape [RA, Dec, Freq Stokes]
	cube_params_dict = {}

	for ax_name in ['RA---SIN','DEC--SIN','FREQ','STOKES']:
		try:
			cube_params_dict[ax_name] = [axis_lenght_array[axis_name_array == ax_name][0],
									axis_reference_val_array[axis_name_array == ax_name][0],
									axis_reference_pix_array[axis_name_array == ax_name][0],
									axis_increment_array[axis_name_array == ax_name][0],
									axis_unit_array[axis_name_array == ax_name][0]]

		except:
			log.warning('Axis {0:s} is not in the fits header!'.format(ax_name))

	return cube_params_dict

def create_empty_fits_mask_from_file(fitspath, maskpath):
	"""Simple routine to create an empty fits file that can be used as an input
	mask for e.g. SoFiA runs

	The created fits will have the same header as the input (primary hdu only!)
	and it will be an empty image cube.

    Parameters
    ==========
    fitspath: str
        The input fits path

	maskpath: str
		The output fits path


    Returns
    =======
		Creates an empty fits image that can be used as a mask for SoFiA

	"""
	hdul = fits.open(fitspath) #Primary hdu list

	#Get the primary header
	primary_table = hdul['PRIMARY']
	primary_header = primary_table.header

	#Get the data
	mask_data = np.zeros(np.shape(primary_table.data))

	hdul.close()

	#Write the output file
	fits.writeto(maskpath, data=mask_data, header=primary_header, overwrite=True)

	return True

def fill_fits_mask_from_pixel_position_list(maskpath, px_position_list,
											flood=False,
											xy_hflood_list=[7],
											z_hflood_list=[5]):
	"""The routine to fill up a mask file. The mask pixels should be provided
	as a list of tuples, where the touples are the *indices* in the fits file.

	The coordinates has to be provided for all pixes!

	The idea is that a single pixel is provided for a source as an input mask for
	SoFiA, that could find the rest of the source

	However, since SoFiA cannot do source-finding based on input pixels (only linking)
	I provide an option of 'flooding' a caviry around a pixel, that can be provided
	as a weights cube to SoFiA: the data is multiplied with the sqrt (!) of the 
	weights cube before source-finding. Essentially, if the weights are zero,
	the pixels are ignored and weight 1 pixels are considered.

	The flood option currently floods cubes around the sources, however, these
	cubes can be different shapes.

	NOTE if a list of flood values with only one element is provided that will be
	used uniformly across the sources.

	NOTE that the input data is overwritten and the output will consist of zeros,
	except at the specified positions

	TO DO: allow for different x (Ra) and y (Dec) filling ranges

	TO DO: allow for the spatial flood range to be set by both pixel and srcsec units

	NOTE that the flooding option only works for image cubes with 4 axis!

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

	flood: bool, optional
		If True, cubes around the sources will be flooded with ones

	xy_hflood_list: list of ints, optional
		The list of the *half* flooding range in both x (RA) and y (Dec) directions

		The units are in pixels !

		Either the same size as the `px_position_list` or the first element is used

	z_hflood_list: list of ints, optional
		The list of the *half* flooding width in the spectral (freq) direction,
		working similarly to `xy_hflood_list`

    Returns
    =======
		Fills up the input data with ones at the given pixel positions

	"""
	hdul = fits.open(maskpath)

	primary_header = hdul['PRIMARY'].header
	original_mask_data = hdul['PRIMARY'].data 
	#Note that we know that only the PRIMARY header exists

	#A sanity check for dimensions and indices
	N_dim = int(primary_header['NAXIS'])

	if len(px_position_list[0]) != N_dim:
		raise ValueError('The input pixel coordinate shape does not match with the image shape!')

	new_mask_data = np.zeros(np.shape(original_mask_data),dtype=int)

	#If only single pixels need to be masked
	if flood == False:
		for pos in px_position_list:
			new_mask_data[pos] = 1

	#If flooding is used
	else:
		#Check for image dim
		if N_dim != 4:
			raise ValueError('Flooding only supports images cubes with 4 dimensions!')

		#Check for flood ranges
		if np.size(xy_hflood_list) != np.size(px_position_list):
			xy_hflood_range = np.multiply(np.ones(np.size(px_position_list)),xy_hflood_list)
		else:
			xy_hflood_range = np.array(xy_hflood_list)

		#NOTE that the two ranges treathed differently!
		if np.size(z_hflood_list) != np.size(px_position_list):
			z_hflood_range = np.multiply(np.ones(np.size(px_position_list)),z_hflood_list)
		else:
			z_hflood_range = np.array(z_hflood_list)

		#Loop trough the input pixel positions
		for i, pos in zip(range(0, len(px_position_list)),px_position_list):
			new_mask_data[pos[0], #Pol axis
						slice(int(pos[1]-z_hflood_range[i]),int(pos[1]+z_hflood_range[i])), #Spectral axis
						slice(int(pos[2]-xy_hflood_range[i]),int(pos[2]+xy_hflood_range[i])), #RA axis
						slice(int(pos[3]-xy_hflood_range[i]),int(pos[3]+xy_hflood_range[i]))] = 1

		#Weight cube input for SoFia should be float type
		new_mask_data = new_mask_data.astype('float64')

	fits.writeto(maskpath, data=new_mask_data, overwrite=True)	

	hdul.close()

	return True

#=== MAIN ===
if __name__ == "__main__":
    pass
