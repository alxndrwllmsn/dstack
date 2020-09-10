"""
Unit testing for the cim module using the unittest module
The test libraries are not part of the module!
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
    - CIMpath: Full path to the CASAImage e.g. /home/user/example]
    - NumericalPrecision: Maximum absolute difference between ASAImages A and B. Have to be >= 0.
    - RMS: the measured RMS of the CIMpath_A image for the first channel and ploarisation in the image cube (the RMS should be measured by using all pixels)

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    CIMPathA: str
        Full path to a test CASAImage, that is used for unittesting of the cim functions

    RMS: float
        Root Mean Square for the first channel and ploarisation in the image cube given by CIMPath

    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMPath =  config.get('CASAImage','CIMpath')
    RMS = float(config.get('CASAImage','RMS'))

    return CIMPath, RMS

class TestCIM(unittest.TestCase):
    CIMPath, RMS = setup_CIM_unittest(PARSET)

    def test_create_CIM_diff_array(self):
        assert np.array_equiv(ds.cim.create_CIM_diff_array(self.CIMPath,self.CIMPath),
        np.zeros((np.shape(casaimage.image(self.CIMPath).getdata())[2],np.shape(casaimage.image(self.CIMPath).getdata())[3]))) == True, \
        'Failed to produce a difference image of zeros using CIM A!'

    def test_measure_CIM_RMS(self):
        assert np.isclose(ds.cim.measure_CIM_RMS(self.CIMPath),self.RMS,rtol=1e-7), \
        'The given RMS and the RMS measured on the image are not matching!'

if __name__ == "__main__":
    unittest.main()