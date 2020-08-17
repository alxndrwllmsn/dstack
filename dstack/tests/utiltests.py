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

    def test_get_MS_phasecentre(self):
        phasecentres = ds.util.ms.get_MS_phasecentre(self.MSPATH)

        assert phasecentres[0][0].ra.deg == self.PHASECENTRE.ra.deg,'Right Ascension mismatch in get_MS_phasecentre()!'
        assert phasecentres[0][0].dec.rad == self.PHASECENTRE.dec.rad, 'Declination mismatch in get_MS_phasecentre()!'


if __name__ == "__main__":
    unittest.main()