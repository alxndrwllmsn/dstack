"""
Unit testing for the cim module using the unittest module
The trest libraries are not part of the module!
Hence, they needs to be handeled separately for now.
"""

import os
import unittest
import configparser
import ast

import dstack as ds
import numpy as np

from casacore import images as casaimage

#Setup the parset file for the unittest
global PARSET
PARSET = './unittest_all.in'


def setup_CIM_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    cimagutil functions against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [CASAImage]

    The parset has to contain the following lines:
    - CIMpath_A: Full path to the CASAImage e.g. /home/user/example]
    - CIMpath_B: Full path to the CASAImage e.g. /home/user/example2]
    - NumericalPrecision: Maximum absolute difference between ASAImages A and B. Have to be >= 0.
    - CIMGridPath: Full path to a grid in CASAImage format
    - Sparseness: Sparseness of the CIMGridPath grid for the first channel and ploarisation in the image cube
    - RMS: the measured RMS of the CIMpath_A image for the first channel and ploarisation in the image cube (the RMS should be measured by using all pixels)

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    CIMPathA: str
        Full path to a test CASAImage, that is used for unittesting of the utilms functions

    CIMPathB: str
        Full path to a second test CASAImage

    NumPrec: float
        The numerical precision limit for testing CASAImage equity

    CIMGridPath: str
        Full path to a test grid in CASAImage format

    Sparseness: float
        Sparseness for the first channel and ploarisation in the image cube of the grid given by CIMGridPath

    RMS: float
        Root Mean Square for the first channel and ploarisation in the image cube of the grid given by CIMPathA

    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMPathA =  config.get('CASAImage','CIMpath_A')
    CIMPathB =  config.get('CASAImage','CIMpath_B')
    NumPrec =  float(config.get('CASAImage','NumericalPrecision'))
    assert NumPrec >= 0., 'The NumericalPrecision given is below zero!'
    CIMGridPath = config.get('CASAImage','CIMGridPath')
    Sparseness = float(config.get('CASAImage','Sparseness'))
    RMS = float(config.get('CASAImage','RMS'))

    return CIMPathA, CIMPathB, NumPrec, CIMGridPath, Sparseness, RMS

class TestCIM(unittest.TestCase):
    CIMPathA, CIMPathB, NumPrec, CIMGridPath, Sparseness, RMS = setup_CIM_unittest(PARSET)

    def test_check_CIM_equity(self):
        assert ds.cim.check_CIM_equity(self.CIMPathA,self.CIMPathB, numprec=self.NumPrec) == True, \
        'The given images differ more than the provided numerical precision tolerance!'

    def test_measure_grid_sparseness(self):
        assert np.isclose(ds.cim.measure_grid_sparseness(self.CIMGridPath),self.Sparseness,rtol=1e-7), \
        'The given sparseness and the sparseness measured on the grid are not matching!'

    def test_create_CIM_diff_array(self):
        assert np.array_equiv(ds.cim.create_CIM_diff_array(self.CIMPathA,self.CIMPathA),
        np.zeros((np.shape(casaimage.image(self.CIMPathA).getdata())[2],np.shape(casaimage.image(self.CIMPathA).getdata())[3]))) == True, \
        'Failed to produce a difference image of zeros using CIM A!'

    def test_measure_CIM_RMS(self):
        assert np.isclose(ds.cim.measure_CIM_RMS(self.CIMPathA),self.RMS,rtol=1e-7), \
        'The given RMS and the RMS measured on the image are not matching!'

if __name__ == "__main__":
    unittest.main()