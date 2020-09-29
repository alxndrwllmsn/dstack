"""
Unit testing for the cgrid module using the unittest module
The test libraries are not part of the module!
Hence, they needs to be handled separately for now.
"""

import os
import unittest
import configparser

import dstack as ds
import numpy as np

from casacore import images as casaimage

#Setup the parset file for the unittest
global _PARSET
_PARSET = './unittest_all.in'

def setup_Cgrid_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    cimagutil functions against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [CGrid]

    The parset has to contain all variavbles and respective values this function returns.

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    GridPath: str
        A ``casacore.images.image.image`` object given by the full path of a test grid in CASAImage format in the parset

    Sparseness: float
        Sparseness for the first channel and polarization in the image cube of the grid given by CIMGridPath

    """
    assert os.path.exists(parset_path), 'Test parset does not exist!'

    config = configparser.ConfigParser()
    config.read(parset_path)

    GridPath = ds.cim.create_CIM_object(config.get('CGrid','GridPath'))
    Sparseness = float(config.get('CGrid','Sparseness'))
    
    return GridPath, Sparseness

class TestCgrid(unittest.TestCase):
    GridPath, Sparseness = setup_Cgrid_unittest(_PARSET)

    def test_measure_grid_sparseness(self):
        assert np.isclose(ds.cgrid.measure_grid_sparseness(self.GridPath), self.Sparseness, rtol=1e-7), \
        'The given sparseness and the sparseness measured on the grid are not matching!'

if __name__ == "__main__":
    unittest.main()