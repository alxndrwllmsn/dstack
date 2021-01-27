"""
Unit testing for the applications which are not modularised and 
tested well enough, using the unittest module
The test libraries are not part of the module!
Hence, they needs to be handled separately for now.
"""

import os
import unittest
import configparser
import subprocess

import dstack as ds
import numpy as np

from casacore import images as casaimage

#Setup the parset file for the unittest
global _PARSET
global _TEST_DIR

_PARSET = './unittest_all.in'
#Working on UNIX systems as it creates a stacked grid at /var/tmp
_TEST_DIR = '/var/tmp'

def setup_APP_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    applications against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [APP]

    The parset has to contain all variavbles and respective values this function returns.

    The apps are unittested by running in the command line using the subprocess module.

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    CIMPath: str
        A ``casacore.images.image.image`` object given by the full path of a test grid in CASAImage format in the parset
    """

    assert os.path.exists(parset_path), 'Test parset does not exist!'

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMPath =  config.get('APP','CIMPath')
    assert os.path.isdir(CIMPath) == True, 'Invalid input image is given {0:s}'.format(CIMPath)
    
    return CIMPath

class TestAPP(unittest.TestCase):
    CIMPath = setup_APP_unittest(_PARSET)

    def test_cim2fits(self):
        output_fits = _TEST_DIR + "/test.fits"

        #Run the app and pipe output and err to STDOUT
        p = subprocess.Popen(['cim2fits', "-i", self.CIMPath, "-o", output_fits],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output, errors = p.communicate()

        #Chehck output file extension and if it is nonzero size
        assert os.path.exists(output_fits) == True, "Output .fits not created!"
        assert os.stat(output_fits).st_size != 0, "The created .fits file is zero bytes!"

if __name__ == "__main__":
    unittest.main()
