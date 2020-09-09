"""
Unit testing for the cgrid module using the unittest module
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

    The config section has to be [CGrid]

    The parset has to contain the following lines:
    - GridPath: Full path to a grid in CASAImage format (should be a small grid)
    - Sparseness: Sparseness of the CIMGridPath grid for the first channel and ploarisation in the image cube

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    GridPath: str
        Full path to a test grid in CASAImage format

    Sparseness: float
        Sparseness for the first channel and ploarisation in the image cube of the grid given by CIMGridPath

    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    GridPath = config.get('CGrid','GridPath')
    Sparseness = float(config.get('CGrid','Sparseness'))
    
    return GridPath, Sparseness

class TestCIM(unittest.TestCase):
    GridPath, Sparseness = setup_CIM_unittest(PARSET)

    def test_measure_grid_sparseness(self):
        assert np.isclose(ds.cgrid.measure_grid_sparseness(self.GridPath),self.Sparseness,rtol=1e-7), \
        'The given sparseness and the sparseness measured on the grid are not matching!'

    def test_grid_stacking_base(self):
        #Working on UNIX systems as it creates a stacked grid at /var/tmp
        ds.cgrid.grid_stacking_base([self.GridPath,self.GridPath],'/var/tmp','test_grid_stacking_base',overwrite=True)

        assert np.array_equiv(np.multiply(casaimage.image(self.GridPath).getdata(),2),casaimage.image('/var/tmp/test_grid_stacking_base').getdata()), \
        'Stacking the same grid not equivalent with multiplying with two!'

if __name__ == "__main__":
    unittest.main()