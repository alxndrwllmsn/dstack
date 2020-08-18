"""
Unit testing for the utils module using the unittest module
"""
import unittest

from astropy.coordinates import SkyCoord
from astropy import units as u

import dstack as ds

class TestMS(unittest.TestCase):
    #Define a test MS path, and the relevanta ttiributes for unittesting
    MSPATH = '/home/krozgonyi/Desktop/sandbox/scienceData_SB10991_G23_T0_B_06.beam17_SL_C_100_110.ms'
    PHASECENTRE = SkyCoord(ra=5.961288415470099 * u.rad, dec= -0.5630396775534987 * u.rad, frame='icrs', equinox='J2000')
    IDS = [0,0]

    def test_get_MS_phasecentre_all(self):
        phasecentres = ds.util.ms.get_MS_phasecentre_all(self.MSPATH)

        assert phasecentres[0][0].ra.deg == self.PHASECENTRE.ra.deg,'Right Ascension mismatch in get_MS_phasecentre()!'
        assert phasecentres[0][0].dec.rad == self.PHASECENTRE.dec.rad, 'Declination mismatch in get_MS_phasecentre()!'

    def test_get_single_phasecentre_from_MS(self):
        phasecentre = ds.util.ms.get_single_phasecentre_from_MS(self.MSPATH,field_ID=self.IDS[0],dd_ID=self.IDS[1])

        assert phasecentre.ra.deg == self.PHASECENTRE.ra.deg,'Right Ascension mismatch in get_MS_phasecentre()!'
        assert phasecentre.dec.rad == self.PHASECENTRE.dec.rad, 'Declination mismatch in get_MS_phasecentre()!'

    def test_check_phaseref_in_MS(self):
        found_IDs = ds.util.ms.check_phaseref_in_MS(self.MSPATH,self.PHASECENTRE)

        assert found_IDs[0][0] == self.IDS[0], 'No matching field ID found!'
        assert found_IDs[0][1] == self.IDS[1], 'No matching direction ID found!'

if __name__ == "__main__":
    unittest.main()