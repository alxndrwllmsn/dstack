"""Utility functions for handeling sources and in broader context some hack for 
sorce finding and characterisation.

NOTE some quick & dirty code is used to deal with SoFiA in an automated way using
templetes... this is not as sophisticated as the interctions with imager...
"""

__all__ = []

import numpy as np
import logging

from astropy.io import fits

from astropy.coordinates import SkyCoord
from astropy import units as u

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def skycords_from_ra_dec_list(ra_list,dec_list,frame='icrs'):
	"""Simple code to convert a list of RA and Dec values provided in degrees to
	astropy sky coordinates
	"""
	return SkyCoord(ra_list, dec_list, unit="deg", frame=frame)

#=== MAIN ===
if __name__ == "__main__":
    pass