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

#Setup the parset file for the unittest
global PARSET
PARSET = './cimtest.in'


def setup_CIM_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    cimagutil functions against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [CASAImage]

    The parset has to contain the following lines:
    - CIMpath_A: full path to the CASAImage e.g. /home/user/example]
    - CIMpath_B: full path to the CASAImage e.g. /home/user/example2]
    - NumericalPrecision: Maximum absolute difference between ASAImages A and B. Have to be >= 0.

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

    """
    assert os.path.exists(parset_path)

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMPathA =  config.get('CASAImage','CIMpath_A')
    CIMPathB =  config.get('CASAImage','CIMpath_B')
    NumPrec =  float(config.get('CASAImage','NumericalPrecision'))

    assert NumPrec >= 0., 'The NumericalPrecision given is below zero!'

    return CIMPathA, CIMPathB, NumPrec

class TestCIM(unittest.TestCase):
    CIMPathA, CIMPathB, NumPrec = setup_CIM_unittest(PARSET)

    def test_check_CIM_equity(self):
        assert ds.cim.check_CIM_equity(self.CIMPathA,self.CIMPathB, numprec=self.NumPrec) == True, \
        'The given images differ more than the provided numerical precision tolerance!'

if __name__ == "__main__":
    unittest.main()