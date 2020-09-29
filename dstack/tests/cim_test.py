"""
Unit testing for the cim module using the unittest module
The test libraries are not part of the module!
Hence, they needs to be handled separately for now.
"""

import os
import unittest
import configparser
import warnings

import dstack as ds
import numpy as np

from casacore import images as casaimage

#Setup the parset file for the unittest
global _PARSET
global _TEST_DIR

_PARSET = './unittest_all.in'
#Working on UNIX systems as it creates a stacked grid at /var/tmp
_TEST_DIR = '/var/tmp'

def setup_CIM_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual CASAImage to test the
    dstack.cim functions against. Thus, any CASAImage can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [CImage]

    The parset has to contain all variavbles and respective values this function returns.

    Parameters
    ==========
    parset_path: str
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
        Number of polarizations in the CASAImage given by CIMPathA

    RMS: float
        Root Mean Square for the first channel and polarization in the image cube given by CIMPath
    """
    assert os.path.exists(parset_path), 'Test parset does not exist!'

    config = configparser.ConfigParser()
    config.read(parset_path)

    CIMPathA =  ds.cim.create_CIM_object(config.get('CImage','CIMpath_A'))
    CIMPathB =  ds.cim.create_CIM_object(config.get('CImage','CIMpath_B'))
    NumPrec =  float(config.get('CImage','NumericalPrecision'))
    assert NumPrec >= 0., 'The NumericalPrecision given is below zero!'
    NChan = int(config.get('CImage','NChannels'))
    NPol = int(config.get('CImage','NPolarisations'))
    RMS = float(config.get('CImage','RMS'))

    return CIMPathA, CIMPathB, NumPrec, NChan, NPol, RMS

class TestCIM(unittest.TestCase):
    CIMPathA, CIMPathB, NumPrec, NChan, NPol, RMS = setup_CIM_unittest(_PARSET)

    def test_check_CIM_axes(self):
        ds.cim.check_CIM_axes(self.CIMPathA)

    def test_CIM_dim_equity_check(self):
        ds.cim.CIM_dim_equity_check(self.CIMPathA,self.CIMPathB)

    def test_CIM_unit_equity_check(self):
        ds.cim.CIM_unit_equity_check(self.CIMPathA,self.CIMPathB)

    def test_check_CIM_equity(self):
        assert ds.cim.check_CIM_equity(self.CIMPathA,self.CIMPathB, numprec=self.NumPrec) == True, \
        'The given images differ more than the provided numerical precision tolerance!'

    def test_get_N_chan_from_CIM(self):
        C = ds.cim.get_N_chan_from_CIM(self.CIMPathA)
        assert C == self.NChan, 'Reference and CASAImage channel number is not the same!'

    def test_get_N_pol_from_CIM(self):
        P = ds.cim.get_N_pol_from_CIM(self.CIMPathA)
        assert P == self.NPol, 'Reference and CASAImage polarisation number is not the same!'

    def test_check_CIM_coordinate_equity(self):
        assert ds.cim.check_CIM_coordinate_equity(self.CIMPathA,self.CIMPathB) == True, \
        'The two given CASAImages have different coordinate systems! '

    def test_create_CIM_diff_array(self):
        assert np.array_equiv(ds.cim.create_CIM_diff_array(self.CIMPathA,self.CIMPathA),
        np.zeros((np.shape(casaimage.image(self.CIMPathA).getdata())[2],np.shape(casaimage.image(self.CIMPathA).getdata())[3]))) == True, \
        'Failed to produce a difference image of zeros using CIM A!'

    def test_measure_CIM_RMS(self):
        assert np.isclose(ds.cim.measure_CIM_RMS(self.CIMPathA), self.RMS, rtol=1e-7), \
        'The given RMS and the RMS measured on the image are not matching!'

    def test_CIM_stacking_base(self):
        ds.cim.CIM_stacking_base([self.CIMPathA,self.CIMPathA],_TEST_DIR,'test_CIM_stacking_base', overwrite=True)

        assert np.array_equiv(np.multiply(casaimage.image(self.CIMPathA).getdata(),2), casaimage.image('{0:s}/test_CIM_stacking_base'.format(_TEST_DIR)).getdata()), \
        'Stacking the same image not equivalent with multiplying with two!'

    def test_set_CIM_unit(self):
        test_cim_name = '{0:s}/test_set_CIM_unit'.format(_TEST_DIR)
        template_cim = ds.cim.create_CIM_object(self.CIMPathA)
        coordsys = template_cim.coordinates()

        #If shape is given, the data type is automatically set to float!
        test_cim = casaimage.image(test_cim_name,
                        coordsys=coordsys,
                        values=template_cim.getdata(),
                        overwrite=True)

        #Need to give a unit that is known to casacore
        ds.cim.set_CIM_unit(test_cim_name,'Jy')
        CIM_with_unit = ds.cim.create_CIM_object('{0:s}/test_set_CIM_unit'.format(_TEST_DIR))
        assert CIM_with_unit.unit() == 'Jy', 'Unable to add unit to newly created CASAImage!'

if __name__ == "__main__":
    unittest.main()