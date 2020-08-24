"""
Unit testing for the msutils module using the unittest module
The trest libraries are not part of the module!
Hence, they needs to be handeled separately for now.
"""

import os
import unittest
import configparser
import ast

from astropy.coordinates import SkyCoord
from astropy import units as u

import dstack as ds

#Setup the parset file for the unittest
global PARSET
PARSET = './msunittest.in'

def setup_MS_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual MS to test the
    msutil functions against. Thus, any MS can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [MS]

    The parset can contain the following lines, but only the relevant functions will be unittested:

    - MSpath: full path to the MS e.g. /home/user/example.ms
    - PhaseCentre: RA and Dec coordinates of the PhaseCentre from the MS in a list,
                    both given in radians e.g. [1.2576,-0.23497]
    - IDs: Field and Direction ID of the PhaseCentre in the MS. Not the reference values,
                    but the 1nd and 0th indices of the FIELD tables PHASE_DIR column
                    e.g. [0,1]
    - NChannels: Number of channels of the MS e.g. 11

    Parameters
    ==========
    parset_path: string
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    MSpath: string
        Full path to a test MS, that is used for unittesting of the utilms functions

    PhaseCentre: Astropy SkyCoord
        The reference phase centre of the MS given as an Astropy SkyCoord object
        Needs to be given in frame='icrs', equinox='J2000' for now

    IDs: list
        The ID of the field and direction of the reference PhaseCentre
    
    NPol: int
        Number of polarisations in the CASAImage
    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    MSpath = config.get('MS','Mspath')

    PhaseCentre = SkyCoord(ra=ast.literal_eval(config.get('MS','PhaseCentre'))[0] * u.rad, 
                dec=ast.literal_eval(config.get('MS','PhaseCentre'))[1] * u.rad,
                frame='icrs', equinox='J2000')
    
    IDs = ast.literal_eval(config.get('MS','IDs'))

    NChan = int(config.get('MS','NChannels'))

    return MSpath, PhaseCentre, IDs, NChan


class TestMS(unittest.TestCase):
    MSpath, PhaseCentre, IDs, NChan = setup_MS_unittest(PARSET)

    def test_get_MS_phasecentre_all(self):
        PhaseCentres = ds.msutil.get_MS_phasecentre_all(self.MSpath)
        assert PhaseCentres[0][0].separation(self.PhaseCentre).arcsec < 1,'Reference PhaseCentre and MS PhaseCentre has >1 arcsec separation!'

    def test_get_single_phasecentre_from_MS(self):
        PhaseCentre = ds.msutil.get_single_phasecentre_from_MS(self.MSpath,field_ID=self.IDs[0],dd_ID=self.IDs[1])
        assert PhaseCentre.separation(self.PhaseCentre).arcsec < 1,'Reference PhaseCentre and MS PhaseCentre has >1 arcsec separation!'

    def test_check_phaseref_in_MS(self):
        found_IDs = ds.msutil.check_phaseref_in_MS(self.MSpath,self.PhaseCentre)
        assert found_IDs[0][0] == self.IDs[0], 'No matching field ID found!'
        assert found_IDs[0][1] == self.IDs[1], 'No matching direction ID found!'

    def test_get_N_chan_from_MS(self):
        C = ds.msutil.get_N_chan_from_MS(self.MSpath)
        assert C == self.NChan, 'Reference and MS channel number is not the same!'

if __name__ == "__main__":
    unittest.main()