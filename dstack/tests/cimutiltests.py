"""
Unit testing for the cimageutil module using the unittest module
The trest libraries are not part of the module!
Hence, they needs to be handeled separately for now.
"""

import os
import unittest
import configparser
import ast

import dstack as ds

#Setup the parset file for the unittest
global PARSET
PARSET = './cimunittest.in'

def setup_CIM_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    cimagutil functions against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [CASAImage]

    The parset can contain the following lines, but only the relevant functions will be unittested:

    - CIMpath: full path to the CASAImage e.g. /home/user/example]
    - NChannels: Number of channels of the MS e.g. 11
    - NPolarisations: Numvber of polarisations e.g. 4

    Parameters
    ==========
    parset_path: string
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    CIMpath: string
        Full path to a test CASAImage, that is used for unittesting of the utilms functions

    NChan: int
        Number of channels in the CASAImage

    NPol: int
        Number of polarisations in the CASAImage
    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMpath = config.get('CASAImage','CIMpath')
    NChan = int(config.get('CASAImage','NChannels'))
    NPol = int(config.get('CASAImage','NPolarisations'))

    return CIMpath, NChan, NPol

class TestMS(unittest.TestCase):
    CIMpath, NChan, NPol = setup_CIM_unittest(PARSET)

    def test_get_N_chan_from_CIM(self):
        C = ds.cimutil.get_N_chan_from_CIM(self.CIMpath)
        assert C == self.NChan, 'Reference and MS channel number is not the same!'

    def test_get_N_pol_from_CIM(self):
        P = ds.cimutil.get_N_pol_from_CIM(self.CIMpath)
        assert P == self.NPol, 'Reference and MS polarisation number is not the same!'

if __name__ == "__main__":
    unittest.main()