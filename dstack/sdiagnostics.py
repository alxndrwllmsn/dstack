"""
Functions used to generate diagnostics plots for the pipeline output.
Namely to plot the SoFiA source finding detections. Note that SoFiA
generates .fits output images, and so this module also includes functions
to handle .fits images. The code should be called within a ``RadioPadre`` notebook!

The formulas used in this module to generate the analytics plots are from
Meyer et al. 2017 (https://arxiv.org/abs/1705.04210)

Therefore for the detailed explanations for the computations, dee Meyer et al. 2017

Some simple functions defined here only for the source analytics. Nd so
these scripts are designed to handle the SoFiA output of HI sources.
"""

__all__ = ['get_z_from_freq', 'fget_wcs']

import os
import shutil
import numpy as np
import logging

from astropy.io import fits
from astropy.wcs import WCS

import dstack as ds

#=== Globals ===
_HI_RESTFREQ = 1420405751.786 #[Hz]

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
#= Conversions & formulae
def get_z_from_freq(obs_freq):
    """ Convert observed frequency to (measured) redshift

    Parameters
    ==========
    obs_freq: float
        Observation (central) frequency in units of Hz !

    Return
    ======
    z: float
        Measured redshift
    """
    z = ( _HI_RESTFREQ / obs_freq ) - 1.

    return z

def get_column_density(S, z, beam_solid_angle):
    """Compute the column density by using eq. 72 from Meyer et al. 2017 

    It is needed to derive the moment0 maps.

    Parameters
    ==========
    S: float
        Measured absolute flux [Jy Hz]

        Why is the formula different in Jonghwhan's code???
        2.33 instead of 2.64

    """

    #N_HI = 2.33 * np.power(10,20) * np.power((1 + z),4) * ( S / beam_solid_angle )
    N_HI = 2.64 * np.power(10,20) * np.power((1 + z),4) * ( S / beam_solid_angle )

    return N_HI



#= Fits
def fget_wcs(fitsfile_path):
    """Get the World coordinate system (wcs) from the fits header for plotting.
    
    Parameters
    ==========
    fitsfile_path: str
        The input fits path

    Return
    ======
    wcs: `astropy.wcs.wcs.WCS`
        Image coordinates in WCS format
    """
    hdu = fits.getheader(fitsfile_path)
    wcs = WCS(hdu).celestial #RA Dec format
    
    return wcs


#=== MAIN ===
if __name__ == "__main__":
    #pass

    print(get_z_from_freq(1412500000))

    exit()

    ff = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/conventional_imaging/beam17_all_cubelets/beam17_all_2_mom0.fits'

    wcs = fget_wcs(ff)

    print(type(wcs))