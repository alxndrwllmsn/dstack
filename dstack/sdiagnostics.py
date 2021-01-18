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

__all__ = ['get_z_from_freq', 'fget_wcs', 'get_column_density']

import os
import shutil
import numpy as np
import logging

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
#from astroquery.utils import download_list_of_fitsfiles

import matplotlib.pyplot as plt

import dstack as ds

#=== Globals ===
_HI_RESTFREQ = 1420405751.786 #[Hz]
_C = 299792.458 #[km/s] speed of light

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


def get_velocity_from_freq(obs_freq, v_frame='optical'):
    """ Convert observed frequency to either optical or radio velocity

    Parameters
    ==========
    obs_freq: float
        Observation frequency in [Hz]

    v_frame: str
        Velocity frame to transform into. Can be `optical` or `radio`

    Return
    ======
    vel: float
        New velocity frmae velocity values in [km/s]
    """
    if v_frame == 'optical':
        vel = _C * (_HI_RESTFREQ / (obs_freq - 1.))
    elif v_frame == 'radio':
        vel = _C * ((1. - obs_freq) / _HI_RESTFREQ)
    else:
        raise ValueError('Not supported velocity frame!')
    
    return vel


def get_column_density(S, z, b_maj, b_min):
    """Compute the column density by using eq. 74 from Meyer et al. 2017 

    It is needed to derive the moment0 maps.

    Parameters
    ==========
    S: float
        Measured absolute flux [Jy Hz]

    z: float
        Measured redshift of the source (use central frequency)

    b_maj: float
        Angular major axis of the beam [arcsec]
    
    b_min: float
        Angular minor axis of the beam [arcsec]

    Return
    ======
    N_HI: float
        Column density
    """
    N_HI = 2.33 * np.power(10,20) * np.power((1 + z),4) * ( S / (b_maj * b_min) )

    return N_HI

def get_column_density_sensitivity(S_rms, z, b_maj, b_min, dnu, sigma_S=1):
    """Compute the column density sensitivity by using eq. 135 from Meyer et al. 2017 

    It is needed to derive a moasking of the low SNR regions in the  moment0 maps.

    Parameters
    ==========
    S_rms: float
        Measured RMS around the source [Jy Hz]

    z: float
        Measured redshift of the source (use central frequency)

    b_maj: float
        Angular major axis of the beam [arcsec]
    
    b_min: float
        Angular minor axis of the beam [arcsec]

    dnu: float
        The frequency width of interest [Hz]

    sigma_S: float
        ??? set to 1 by default for now

    Return
    ======
    N_HI_sensitivity: float
        Column density sensitivity
    """
    S_sensitivity = S_rms * dnu * sigma_S

    N_HI_sensitivity = get_column_density(S_sensitivity, z, b_maj, b_min)
 
    return N_HI_sensitivity

#= Sofia catalog
def get_freq_and_redshift_from_catalog(catalog_path, source_index, rest_frame='frequency'):
    """ Extract the central (?) frequency and corresponding redshift from 
    the SoFiA catalog (VO-table in xml format). Works on a given source defined by
    the source index, which is the row index (!) of the source not the catalog index.

    Note, that these should be identical though.

    Parameters
    ==========
    catalog_path: str
        Full path and file name of the SoFiA catalog

    source_index: int
        The index of the source in the catalog. (row index)

    rest_frame: str
        The rest frame used. Currently only 'frequency' frame supported. 

    Return
    ======
    freq: str
        Central frequency in [Hz]
    z: str
        Measured redshift
    """

    assert rest_frame == 'frequency', 'Currently only the frequency rest frame supported!'
  
    catalog = parse_single_table(catalog_path).to_table(use_names_over_ids=True)

    freq = Column(catalog['freq'], name='freq')[source_index]
    z = Column(get_z_from_freq(catalog['freq']), name='z')[source_index]

    return freq, z

def get_RMS_from_catalog(catalog_path, source_index, rest_frame='frequency'):
    """Get the RMS value measured by SoFiA for the given source.

    Parameters
    ==========
    catalog_path: str
        Full path and file name of the SoFiA catalog

    source_index: int
        The index of the source in the catalog. (row index)

    rest_frame: str
        The rest frame used. Currently only 'frequency' frame supported. 

    Return
    ======
    rms: float
        Measured RMS
    """
    assert rest_frame == 'frequency', 'Currently only the frequency rest frame supported!'
  
    catalog = parse_single_table(catalog_path).to_table(use_names_over_ids=True)

    rms = catalog['rms'][source_index]
 
    return rms



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

def fget_beam(fitsfile_path):
    """Get synthesised beam parameters from the
    input fits header

    Parameters
    ==========
    fitsfile_path: str
        The input fits path

    Returns
    =======
    b_maj: float
        Angular major axis of the beam [arcsec]
    
    b_min: float
        Angular minor axis of the beam [arcsec]

    b_pa: float
        Position angle of the beam [deg]
    """
    b_maj = fits.getheader(fitsfile_path)['BMAJ'] * 3600     # [arcsec]
    b_min = fits.getheader(fitsfile_path)['BMIN'] * 3600     # [arcsec]
    b_pa = fits.getheader(fitsfile_path)['BPA']              # [deg]
    
    return b_maj, b_min, b_pa

def fget_channel_width(fitsfile_path):
    """


    """
    
    dnu = fits.getheader(fitsfile_path)['CDELT3'] #[Hz]
    
    return dnu


#= Get optical image
def get_optical_image(catalog_path, source_index, survey, pixel_size):
    """ Download optical image cutsouts of sources from SkyVeiw via astroquery
    """
    
    catalog = parse_single_table(catalog_path).to_table(use_names_over_ids=True)

    ra = catalog['ra'][source_index]
    dec = catalog['dec'][source_index]
    pos = SkyCoord(ra=ra, dec=dec, unit='deg')

    # get url link to images
    optical_fits = SkyView.get_images(position=pos, survey=survey, projection='Sin', pixels=pixel_size)[0]

    return optical_fits


#=== MAIN ===
if __name__ == "__main__":
    #pass

    #catalog = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/conventional_imaging/beam17_all_cat.xml'
    catalog = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/stacked_grids/beam17_all_cat.xml'

    #optical_im = get_optical_image(catalog,1,'DSS2 Red',max(wcs.array_shape) * 5 )

    #exit()

    freq, z = get_freq_and_redshift_from_catalog(catalog,1)

    #reference_image = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/conventional_imaging/beam17_all_cubelets/beam17_all_2_cube.fits'

    #mom0 =  '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/conventional_imaging/beam17_all_cubelets/beam17_all_2_mom0.fits'
    #mom1 = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/conventional_imaging/beam17_all_cubelets/beam17_all_2_mom1.fits'
    
    reference_image = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/stacked_grids/beam17_all_cubelets/beam17_all_1_cube.fits'

    mom0 =  '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/stacked_grids/beam17_all_cubelets/beam17_all_1_mom0.fits'
    mom1 = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/stacked_grids/beam17_all_cubelets/beam17_all_1_mom1.fits'


    wcs = fget_wcs(mom0)

    optical_im = get_optical_image(catalog,0,'DSS2 Red', max(wcs.array_shape) * 8 )
   
    wcs_opt = WCS(optical_im[0].header)

    mom0_map = fits.getdata(mom0)
    mom1_map = get_velocity_from_freq(fits.getdata(mom1))

    #b_maj, b_min, b_pa = fget_beam(reference_image) #Not working with the grid stack output!
    b_maj = 5
    b_min = 5


    col_den_map = get_column_density(mom0_map, z, b_maj, b_min) * 1e-19

    rms = get_RMS_from_catalog(catalog,1)

    dnu = fget_channel_width(reference_image)

    sen_lim = get_column_density_sensitivity(rms, z, b_maj, b_min, dnu, 1) * 1e-19

    #Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs_opt)

    #mom0
    ax.imshow(optical_im[0].data,origin='lower',cmap='Greys')
    #ax.imshow(col_den_map, origin='lower')
    ax.contour(col_den_map, levels=[3*sen_lim, 5*sen_lim, 7*sen_lim, 9*sen_lim, 11*sen_lim],transform=ax.get_transform(wcs))

    #mom1
    #ax.imshow(mom1_map, origin='lower',cmap='coolwarm')

    plt.show()
