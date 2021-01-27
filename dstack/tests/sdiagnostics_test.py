"""
Unit testing for the sdiagnositics module using the unittest module
The test libraries are not part of the module!
Hence, they needs to be handled separately for now.

Testing this module is a little bit tricky, as majority of the
functions are focusing on creating diagnostics plots.

So only some functions will be tested, some does not need an actual
SoFiA output, but some does, thus the user has to provide that.

This testing module only checks the core functions, but NOT the
plot functions. It would be too slow to test all the plot functions.
The important things are the core funtions anyway. These gets the
sources and derive the source parameters (arrays imaged).
"""

import os, sys
import unittest
import configparser
import warnings
import logging

import dstack as ds
import numpy as np
from astropy.wcs import WCS


#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Disable fits warnings ===
#In fact this ignores all Warnings, so comment this line for debugging!
#This does not work with the unittest package so all the stupid .fits warnings will appear in STD..
warnings.filterwarnings('ignore', category=Warning, append=True)

#Setup the parset file for the unittest
global _PARSET
global _TEST_DIR

_PARSET = './unittest_all.in'
#Working on UNIX systems as it creates a stacked grid at /var/tmp
_TEST_DIR = '/var/tmp'

def setup_sdiagnostics_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual SoFiA output to test the
    dstack.sdiagnostics functions against. Thus, any SoFiA output can be used for unittesting provided by the user.
    
    The code does NOT check the plot functions, but only the core functions. Thus the images created needs to be
    checked 'manually'. However, the functions generating the arrays for the plots are testes at some level.

    The code uses the configparser package to read the config file

    The config section has to be [Sdiagnostics]

    The parset has to contain all variavbles and respective values this function returns.

    A source provided by the user of the SoFiA output will be used for testing.
    Therefore the source ID has to be selected carefully (i.e. not out of range).i

    For simplicity each variable returned by this function has to be defined using the
    same variable name in the parset file!

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    source_ID: int
        The ID of the source in the SoFiA catalog, which will be used for unittesting

    sofia_dir_path: str
        Full path to the directory where the SoFiA output used for unittesting is located. Has to end with a slash (/)!

    name_base: str
      The `output.filename` variable defined in the SoFiA template .par when the SoFiA test directory generated.
      Basically the base of all file names. However, it has to end with a lower dash (?): _ !

    N_sources: int
        Number of sources in the test SoFiA directory.

    freq: float
        Frequency of the test source defined in the SoFiA catalog.
        Copy from catalog to parset file.

    z: float
        The redshift derived from the test source's frequency. 
    
    rms: float
        The measured RMS in the SoFiA catalog. Copy the value to the parset file.

    dnu: float
        The channel width in [HZ] of the SoFiA output cubes.
    """
    assert os.path.exists(parset_path), 'Test parset does not exist!'

    config = configparser.ConfigParser()
    config.read(parset_path)

    source_ID = int(config.get('Sdiagnostics','source_ID'))
    sofia_dir_path = str(config.get('Sdiagnostics','sofia_dir_path'))
    assert os.path.isdir(sofia_dir_path) == True, 'Invalid input SoFiA test directory is specified!'
    name_base = str(config.get('Sdiagnostics','name_base'))
    N_sources = int(config.get('Sdiagnostics','N_sources'))
    assert N_sources >= source_ID, 'Invalid source ID or too small N_sources given in the parset file!'
    freq = float(config.get('Sdiagnostics','freq'))
    z = float(config.get('Sdiagnostics','z'))
    rms = float(config.get('Sdiagnostics','rms'))
    dnu = float(config.get('Sdiagnostics','dnu'))

    return source_ID, sofia_dir_path, name_base, N_sources, freq, z, rms, dnu

class TestSdiagnostics(unittest.TestCase):
    source_ID, sofia_dir_path, name_base, N_sources, freq, z, rms, dnu = setup_sdiagnostics_unittest(_PARSET)

    def test_get_source_files(self):
        source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(self.source_ID, self.sofia_dir_path, self.name_base)
        assert source_index == (self.source_ID - 1), 'Source ID and index are missmatching!'
        assert os.path.isfile(catalog_path) == True, 'Invalid catalog path created!'
        for cubelet in cubelet_path_dict:
            assert os.path.isfile(cubelet_path_dict[cubelet]) == True, 'Invalid path to cubelet: {0:s} created!'.format(cubelet) 
        assert os.path.isfile(spectra_path) == True, 'Invalid spectra path created!'

        return source_index, catalog_path, cubelet_path_dict, spectra_path

    def test_get_N_sources(self):
        N = ds.sdiagnostics.get_N_sources(self.sofia_dir_path, self.name_base)
        assert N == self.N_sources, 'Invalid number of sources foud by sdiagnostics! (given: {0:d}; found: {1:d}'.format(self.N_sources,N)

    def test_get_freq_and_redshift_from_catalog(self):
        source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(self.source_ID, self.sofia_dir_path, self.name_base)
        freq, z = ds.sdiagnostics.get_freq_and_redshift_from_catalog(catalog_path, source_index)

        assert freq == self.freq, 'Input frequency is different from what read from the SoFiA catalog!'
        
        #This test also tests the get_z_from_freq() function!
        assert np.isclose(z, self.z, rtol=1e-4), 'The derived z is different from the given z value!'

    def test_get_RMS_from_catalog(self):
        source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(self.source_ID, self.sofia_dir_path, self.name_base)
        rms = ds.sdiagnostics.get_RMS_from_catalog(catalog_path, source_index)

        assert rms == self.rms, 'Input RMS is different from what read from the SoFiA catalog!'
 
    def test_freq_to_velocity_conversions(self):
        #Tests the following functions using the global HI velocity variable defined in sdiagnostics
        # - get_velocity_from_freq()
        # - get_velocity_dispersion_from_freq()

        f = ds.sdiagnostics._HI_RESTFREQ #Get the global HI restfrequency

        assert ds.sdiagnostics.get_velocity_from_freq(f) == 0., 'Wrong velocity (optical) conversion is used in sdiagnostics!'
        assert ds.sdiagnostics.get_velocity_from_freq(f, v_frame='optical') == ds.sdiagnostics.get_velocity_from_freq(f, v_frame='radio'), \
                'Wrong velocity (radio) conversion is used in sdiagnostics!'

        assert ds.sdiagnostics.get_velocity_dispersion_from_freq(f,f) == ds.sdiagnostics._C, 'Wrong velocity dispersion calculation is used!'

    def test_column_density_and_sensitivity_conversion(self):
        #This test uses unity input values at z=0to the column density and column density sensitivity functions
        #in which case the output values are known. The following functions are tested:
        # - get_column_density()
        # - get_column_density_sensitivity()
        #
        # Both output should be 2.33 (scaling is set to e-20)

        #But I check if the code returns a WCS object!

        assert ds.sdiagnostics.get_column_density(S=1,z=0,b_maj=1,b_min=1,scaling=1e-20) == 2.33, \
                'Wrong derivation of column density is used in sdiagnostics!'

        assert ds.sdiagnostics.get_column_density_sensitivity(S_rms=1,z=0,b_maj=1,b_min=1,dnu=1,sigma_S=1,scaling=1e-20) == 2.33, \
                'Wrong derivation of column density sensitivity is used in sdiagnostics!' 

    def test_fget_wcs(self):
        #Test if a valid WCS object is returned from the input fits
        source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(self.source_ID, self.sofia_dir_path, self.name_base)
        
        mom0_wcs = ds.sdiagnostics.fget_wcs(cubelet_path_dict['mom0'])
        assert type(mom0_wcs) == type(WCS()), 'Invalid worls coordinate system info retrieved from fits!'

    def test_fget_channel_width(self):
        source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(self.source_ID, self.sofia_dir_path, self.name_base)
 
        dnu = ds.sdiagnostics.fget_channel_width(cubelet_path_dict['cube'])
        assert dnu == self.dnu, 'The channelwidth provided differs from the channel width gained from the fits cube!' 


    def test_get_optical_image_ndarray(self):
        #This function tests the creation of the optical background image. However, it only tests if the requested size
        #background image is returned regardless of the values, the phase centre and the survey used.
        
        #Note, thate this test slows down the whole testing process!
        opt_array, opt_im_wcs, opt_survey  = ds.sdiagnostics.get_optical_image_ndarray(self.source_ID, self.sofia_dir_path, self.name_base, N_optical_pixels=25)
        
        assert np.shape(opt_array) == ((25,25)), 'The created  optical background image have wrong shape!'

        assert type(opt_im_wcs) == type(WCS()), 'Not astropy.wcs.wcs.WCS type world coordinate system is returned!'

        assert opt_survey in ['DSS2 Red', 'DSS1 Red', 'DSS2 Blue', 'DSS1 Blue', 'DSS2 IR', 'None'], \
                'Invalid survey option ({0:s}) for the background optical image is returned!'.format(opt_survey)

    def test_get_momN_ndarray(self):
        #This is even a vaguer test. It only checks if masked numpy.ndarray moment maps are returned.
        #Also checks if the WCS and the sensitivity returned is rhe right type

        for m in range(0,3):
            mom_map_array, mom_map_wcs, sensitivity_lim = ds.sdiagnostics.get_momN_ndarray(m, self.source_ID,self.sofia_dir_path,self.name_base)
    
            assert type(mom_map_array) == type(np.ma.array(np.zeros((2,2)), mask=np.array([[0,1],[1,0]]))), \
                    'Unmasked numpy array is returned for moment map {0:d}! (unexpected behaviour)'.format(m) 

            assert type(mom_map_wcs) == type(WCS()), \
                    'Not astropy.wcs.wcs.WCS type world coordinate system is returned form mom map {0:d}!'.format(m)

            assert type(sensitivity_lim) != float or type(sensitivity_lim) != None, \
                    'Invalid sensitivity limit (type) is returned!'

    def test_get_spectra_array(self):
        #Test if same lenght numpy arrays are returned
        f, v = ds.sdiagnostics.get_spectra_array(self.source_ID,self.sofia_dir_path,self.name_base)
        assert np.shape(f) == np.shape(v), 'Not matching flux and velocity arrays are returned!'

if __name__ == "__main__":
    unittest.main()
