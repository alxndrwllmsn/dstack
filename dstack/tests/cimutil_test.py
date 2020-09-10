"""
Unit testing for the cimutil module using the unittest module
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
PARSET = './unittest_all.in'

def setup_CIMutil_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    cimagutil functions against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [CASAImageUtil]

    The parset has to contain all the lines diven in the return parameters.
    - CIMpathA: full path to the CASAImage e.g. /home/user/example
    - CIMpathB: full path to the CASAImage that have the same coordinate system as CIMpathA e.g. /home/user/example
    - NumericalPrecision: The relative precision for which images CIMpathA and CIMpathB are equal. Can be zero.
    - NChannels: Number of channels of the MS e.g. 11
    - NPolarisations: Numvber of polarisations e.g. 4

    Parameters
    ==========
    parset_path: string
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    CIMPathA: str
        A ``casacore.images.image.image`` object given by the full path of a test grid in CASAImage format in the parset for Alice

    CIMPathB: str
        A ``casacore.images.image.image`` object given by the full path of a test grid in CASAImage format in the parset for Bob

    NumPrec: float
        The numerical precision limit for testing CASAImage equity

    NChan: int
        Number of channels in the CASAImage given by CIMPathA

    NPol: int
        Number of polarisations in the CASAImage given by CIMPathA
    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMPathA =  ds.cimutil.create_CIM_object(config.get('CASAImageUtil','CIMpath_A'))
    CIMPathB =  ds.cimutil.create_CIM_object(config.get('CASAImageUtil','CIMpath_B'))
    NumPrec =  float(config.get('CASAImageUtil','NumericalPrecision'))
    assert NumPrec >= 0., 'The NumericalPrecision given is below zero!'
    NChan = int(config.get('CASAImageUtil','NChannels'))
    NPol = int(config.get('CASAImageUtil','NPolarisations'))

    return CIMPathA, CIMPathB, NumPrec, NChan, NPol

class TestCIMUtil(unittest.TestCase):
    CIMPathA, CIMPathB, NumPrec, NChan, NPol = setup_CIMutil_unittest(PARSET)

    def test_check_CIM_equity(self):
        assert ds.cimutil.check_CIM_equity(self.CIMPathA,self.CIMPathB, numprec=self.NumPrec) == True, \
        'The given images differ more than the provided numerical precision tolerance!'

    def test_get_N_chan_from_CIM(self):
        C = ds.cimutil.get_N_chan_from_CIM(self.CIMPathA)
        assert C == self.NChan, 'Reference and CASAImage channel number is not the same!'

    def test_get_N_pol_from_CIM(self):
        P = ds.cimutil.get_N_pol_from_CIM(self.CIMPathA)
        assert P == self.NPol, 'Reference and CASAImage polarisation number is not the same!'

    def test_check_CIM_coordinate_equity(self):
        assert ds.cimutil.check_CIM_coordinate_equity(self.CIMPathA,self.CIMPathB) == True, \
        'The two given CASAImages have different coordinate systems! '

if __name__ == "__main__":
    unittest.main()