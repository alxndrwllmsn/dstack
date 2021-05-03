"""
Collection of functions operating on images in CASAImageformat.
The functions defined here are expected to work on grids as well, as grids are currently dumped in casaimage format,
hence functions defined in this module works on both images and grids.
The image I/O management is kinda manually at this point, but hopefully will be handled on higher level applications in the future.
"""

__all__ = ['create_CIM_object', 'check_CIM_axes', 'CIM_dim_equity_check', 'CIM_unit_equity_check',
            'get_N_chan_from_CIM', 'get_CIM_spectral_axis_array', 'get_N_pol_from_CIM',
            'check_CIM_coordinate_equity', 'check_CIM_equity', 'normalise_CIM', 'measure_CIM_RMS',
            'create_CIM_diff_array','measure_CIM_max' ,'CIM_stacking_base']

import os
import shutil
import numpy as np
import logging

import copy

from casacore import images as casaimage
from casacore import tables as casatables

#Parallel solutions using multiprocessing
import multiprocessing as mp

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Globals ===
_DEFAULT_REQUIRED_AXES = 4 #Number of axes in a custom image: [freq, Stokes, x, y]

#=== Functions ===
def create_CIM_object(cimpath):
    """This function aims to speed up other bits of this and ``cgrid``
    modules, by returning a ``casacore.images.image.image`` object.

    The trick is, that the ``cimpath`` argument can be either a string i.e. the path
    to the CASAImage wich will be read in and returned, **or** it can be already an
    in-memory ``casacore.images.image.image`` object.

    This might not be the best solution, but I hope overall a check in a lot of cases will
    speed up code, rather than reading in the same CASAImage again-and again. So ideally, only
    one reading in happens for each CASAImage and all inside this function!

    Parameters
    ==========
    cimpath: str
        The input CASAImage path or a ``casacore.images.image.image`` object

    Returns
    =======
    cim: ``casacore.images.image.image`` object
        The in-memory CASAImage

    """
    #create an empty image in-memory to check the object type
    if type(cimpath) == type(casaimage.image(imagename='',shape=np.ones(1))):
    #if type(cimpath) == 'casacore.images.image.image': #this does not seems to work
        return cimpath
    else:
        # We could simply return, no need to assign the return value of
        # `casaimage.image(cimpath)` to a new variable.
        log.debug('Open image: {0:s}'.format(str(cimpath))) #We know it is a string in this case
        return casaimage.image(cimpath)

def check_CIM_axes(cim, required_axes=_DEFAULT_REQUIRED_AXES):
    """Checks if the ``cim`` image object has the correct number of
    axes or dimensions. Raises
    
    Notes
    =====
    I would refactor this safety check into this separate function, even
    though it is only used in two functions. Couple of reasons:
    
    If later it is decided that a different number of axes needs
    to be used, this function can simply be updated so that
    ``required_axes`` will have a different default value and there will
    be no need to copy and paste changes in different parts of the code.
    
    This function can be used in different modules if the need arises,
    and there will be no need to copy-paste the code segment.
    
    Parameters
    ==========
    cim: ``casacore.images.image.image`` object
        In-memory CASAImage
    
    required_axes: int, optional
        Number of required axes or dimensions for the ``cim`` image
        object.
    
    Raises
    ======
    ValueError
        If the number of axes is does not equal the required
        numbers.
    """
    if cim.ndim() != required_axes:
        raise ValueError('The image has more or less than the required {0:d} axes!'.format(int(required_axes)))


def CIM_dim_equity_check(cimpath_a, cimpath_b):
    """Checks if the ``cimpath_a`` and ``cimpath_b`` image objects have the same number
    of dimensions.
    
    Parameters
    ==========
    cimpath_a: ``casacore.images.image.image`` object
        In-memory CASAImage
    cimpath_b: ``casacore.images.image.image`` object
        In-memory CASAImage
    
    Raises
    ======
    ValueError
        If the number of dimensions is not equal for ``cimpath_a`` and ``cimpath_b``.
    
    """
    if cimpath_a.ndim() != cimpath_b.ndim():
        raise ValueError('The dimension of the two input CASAImage are not equal!')

def CIM_unit_equity_check(cimpath_a, cimpath_b):
    """Checks if the ``cim_a`` and ``cimpath_b`` image objects have the same pixel units.
    
    Parameters
    ==========
    cimpath_a: ``casacore.images.image.image`` object
        In-memory CASAImage
    cimpath_b: ``casacore.images.image.image`` object
        In-memory CASAImage
    
    Raises
    ======
    ValueError
        If the number of dimensions is not equal for ``cimpath_a`` and ``cimpath_b``.
    
    """
    if cimpath_a.unit() != cimpath_b.unit():
        raise ValueError('The pixel units of the two input CASAImage are not equal!')

def get_N_chan_from_CIM(cimpath, close=False, required_axes=_DEFAULT_REQUIRED_AXES):
    """Get the number of channels from a CASAImage

    CASAImage indices: [freq, Stokes, x, y]

    Parameters
    ==========
    cimpath: str
        The input CASAImage path or a ``casacore.images.image.image`` object
    
    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    required_axes: int, optional
        Number of required axes or dimensions for the ``cim`` image
        object.

    Returns
    =======
    N_chan: int
        Number of channels in the CASAImage
    """
    cim = ds.cim.create_CIM_object(cimpath)
    check_CIM_axes(cim,required_axes=required_axes)

    N_chan = np.shape(cim.getdata())[0]

    if close:
        log.debug('Closing image: {0:s}'.format(cim.name()))
        del cim

    return N_chan

def get_CIM_spectral_axis_array(cimpath, chan=None, chan_max=None, close=False):
    """Get the spectral coordinate values for a given channel range and returns it as an array.
    By default returns the whole spectral axis as an array and the unit. Nevertheless, the user
    can define a single channel or a channel range using channel indices, in which case, only
    those channels frequency values are returned.

    Note, that the spectral values returned are given in the image's reference frame! (i.e topocentric, barycentric...etc.)

    Parameters
    ==========
    cimpath: str
        The input CASAImage path or a ``casacore.images.image.image`` object

    chan: int, optional
        The first channel of the given spectral window as a channel index. None by default.
        If not defined, the function returns an array containing all channel frequency values.

    chan_max: int, optional
        The index of the last + 1 channel of the selected spectral window. None by default.
        If not given, but the `chan` parameter is defined only a single channel frequency value is returned. 

    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    Return
    ======
    spectral_array: numpy.ndarray
        The frequency values of the given spectral window.

    spectral_unit: str
        The unit in which the spectral values are given
    """
    cim = ds.cim.create_CIM_object(cimpath)

    coords = cim.coordinates()
    coords_axis = 'spectral'

    #Get spectral axis parameters
    spectral_reference_frame = coords[coords_axis].get_frame()
    spectral_unit = coords[coords_axis].get_unit()
    spectral_increment = coords[coords_axis].get_increment()
    spectral_reference_pixel = int(coords[coords_axis].get_referencepixel())
    spectral_reference_value = coords[coords_axis].get_referencevalue()
    spectral_restfrequency = coords[coords_axis].get_restfrequency()

    log.debug('Spectral axis defined in {0:s} frame with unit {1:s} with rest frequency {2:.2f}; \
reference frequency {3:.2f} of channel {4:d} with channel width of {5:.2f}.'.format(
            spectral_reference_frame,spectral_unit, spectral_restfrequency, spectral_reference_value,
            spectral_reference_pixel, spectral_increment))

    #Select spectral window
    if chan == None:
        chan = 0
        chan_max = get_N_chan_from_CIM(cim)

    elif chan != None and chan_max != None:
        if chan_max <= chan and chan_max > 0:
            raise ValueError('Invalid lower ({0:d}) and upper ({1:d}) channel indices are given!'.format(chan,chan_max))

        if np.fabs(chan_max) > get_N_chan_from_CIM(cim):
            raise ValueError('Upper channel index {0:d} is out from the channel range of the image!'.format(chan_max))
    else:
        chan_max = chan + 1 

    #Get subband size in channels
    window_size = chan_max - chan
    spectral_array = np.zeros(window_size)

    #Map axis to selected spectral window array
    #Should vectorise this...
    for i in range(0,window_size):
        spectral_array[i] = spectral_reference_value - ((spectral_reference_pixel - (chan + i)) * spectral_increment)

    if close:
        log.debug('Closing image: {0:s}'.format(cim.name()))
        del cim

    return spectral_array, spectral_unit

def get_N_pol_from_CIM(cimpath, close=False, required_axes=_DEFAULT_REQUIRED_AXES):
    """Get the number of polarizations from a CASAImage. Note, that the
    polarization type is not returned!

    CASAImage indices: [freq, Stokes, x, y]

    Parameters
    ==========
    cimpath: str
        The input CASAImage path or a ``casacore.images.image.image`` object
    
    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    required_axes: int, optional
        Number of required axes or dimensions for the ``cim`` image
        object.

    Returns
    =======
    N_pol: int
        Number of polarizations in the CASAImage
    """
    cim = ds.cim.create_CIM_object(cimpath)
    check_CIM_axes(cim,required_axes=required_axes)

    N_pol = np.shape(cim.getdata())[1]

    if close:
        log.debug('Closing image: {0:s}'.format(cim.name()))
        del cim

    return N_pol

# one space should follow each comma (,)
def check_CIM_equity(cimpath_a, cimpath_b, numprec=1e-8, close=False):
    """Check if two CASAImages are identical or not up to a defined numerical precision
    This function is used to test certain pipeline features.

    Note: NaN-s are treated as equals, and the ``numprec`` parameter only sets the relative difference limit.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage path of Alice or a ``casacore.images.image.image`` object

    cimpath_b: str
        The input CASAImage path of Bob or a ``casacore.images.image.image`` object

    numprec: float, optional
        The numerical precision limit of the maximum allowed relative
        difference between CASAImages Alice and Bob.
        If set to zero, equity is checked.

    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    Returns
    =======
    equity: bool
        True or False, base on the equity of Alice and Bob
    """

    cimA = ds.cim.create_CIM_object(cimpath_a)
    cimB = ds.cim.create_CIM_object(cimpath_b)

    CIM_dim_equity_check(cimA,cimB)

    if numprec == 0.:
        equviv = np.array_equiv(cimA.getdata(),cimB.getdata())
    else:
        equviv = np.allclose(cimA.getdata(),cimB.getdata(),atol=0,rtol=numprec,equal_nan=True)

    if close:
        log.debug('Closing image: {0:s}'.format(cimA.name()))
        log.debug('Closing image: {0:s}'.format(cimB.name()))
        del cimA
        del cimB

    return equviv

def check_CIM_coordinate_equity(cimpath_a, cimpath_b, iclude_pixel_unit=True, close=False):
    """Basic check if the associated coordinate information of two images are somewhat equal.
    This is **not** an equity check for all coordinate values, as the reference pixels can be different,
    even for images (grids) with the same coordinate system. Hence, the rigorous part of the check is
    the increment between channels/pixels. The main idea behind this function is to check if
    images (grids) can be stacked together.

    Note, that the two CASAImages have to be the same dimension!

    The code actually never returns False but fails due to assertion...
    In the future the option to return False instead of assertion can be added to handle
    some weird special cases.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage path of Alice or a ``casacore.images.image.image`` object

    cimpath_b: str
        The input CASAImage path of Bob or a ``casacore.images.image.image`` object

    iclude_pixel_unit: bool, optional
        When comparing the coordinate axis of an image and it's corresponding PSF, the pixel units can be different.
        Thus, this potion allows the user to skip the check of the pixel unit equity of the two input CIM

    close: bool, optional
        If True the in-memory CASAIMages are deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    Returns
    =======
    equity: bool
        True or False, base on the coordinate equity of Alice and Bob
    """
    cimA = ds.cim.create_CIM_object(cimpath_a)
    cimB = ds.cim.create_CIM_object(cimpath_b)
    
    CIM_dim_equity_check(cimA, cimB)
    if iclude_pixel_unit:
        CIM_unit_equity_check(cimA, cimB)

    coordsA = cimA.coordinates()
    coordsB = cimA.coordinates()

    #Spectral coordinates
    coords_axis = 'spectral'
    log.debug('Checking {0:s} axis equity'.format(coords_axis))
    
    if coordsA[coords_axis].get_frame() != coordsB[coords_axis].get_frame():
        raise ValueError('The given images {0:s} and {1:s} have different frames!'.format(cimA.name(),cimB.name()))

    if coordsA[coords_axis].get_unit() == coordsB[coords_axis].get_unit():
        if coordsA[coords_axis].get_increment() != coordsB[coords_axis].get_increment():
            raise ValueError('The increment of the two spectral coordinates are different for images {0:s} and {1:s}'.format(
                            cimA.name(),cimB.name()))

        if coordsA[coords_axis].get_restfrequency() != coordsB[coords_axis].get_restfrequency():
            raise ValueError('The rest frame frequency of the two spectral coordinates are different for images {0:s} and {1:s}'.format(
                            cimA.name(),cimB.name()))
        
        if coordsA[coords_axis].get_referencepixel() == coordsB[coords_axis].get_referencepixel():
            if coordsA[coords_axis].get_referencevalue() != coordsB[coords_axis].get_referencevalue():
                raise ValueError('The reference values of the spectral corrdinates are different for images {0:s} and {1:s}'.format(
                                cimA.name(),cimB.name()))
        else:
            log.warning('The input images {0:s} and {1:s} have different spectral coordinate reference pixel!'.format(
                    cimA.name(),cimB.name()))
    else:
        log.warning('The input images {0:s} and {1:s} have different spectral coordinate units!'.format(
                    cimA.name(),cimB.name()))

    #Polarization coordinates
    coords_axis = 'stokes'
    log.debug('Checking {0:s} axis equity'.format(coords_axis))

    if coordsA[coords_axis].get_stokes() != coordsB[coords_axis].get_stokes():
        raise ValueError('The polarization frame is different for images {0:s} and {1:s}!'.format(
                        cimA.name(),cimB.name()))

    #Direction coordinates if images and linear coordinates if grids
    coords_axis = 'direction'
    try:
        assert coordsA[coords_axis].get_frame() == coordsB[coords_axis].get_frame(), \
        'The given images {0:s} and {1:s} have different frames!'.format(cimA.name(),cimB.name())

        assert coordsA[coords_axis].get_projection() == coordsB[coords_axis].get_projection(), \
        'The given images {0:s} and {1:s} have different projections!'.format(cimA.name(),cimB.name())

        log.debug("Image axis are: 'diretion'")

    except AssertionError:
        if coordsA[coords_axis].get_frame() != coordsB[coords_axis].get_frame():
            raise ValueError('The given images {0:s} and {1:s} have different frames!'.format(cimA.name(),cimB.name()))

        if coordsA[coords_axis].get_projection() != coordsB[coords_axis].get_projection():
            raise ValueError('The given images {0:s} and {1:s} have different projections!'.format(cimA.name(),cimB.name()))
        
        log.debug("Image axis are: 'direction'")
    
    except:
        #Change to linear coord as the given CASAimage is a grid!
        coords_axis = 'linear'
        log.debug("Image axis are: 'linear'")

    if np.all(np.array(coordsA[coords_axis].get_unit()) == np.array(coordsB[coords_axis].get_unit())):
        if np.all(np.array(coordsA[coords_axis].get_increment()) != np.array(coordsB[coords_axis].get_increment())):
            raise ValueError('The increment of the (x,y) direction coordinates are different for the input images {0:s} and {1:s}'.format(
                            cimA.name(),cimB.name()))

        if np.all(np.array(coordsA[coords_axis].get_referencepixel()) == np.array(coordsB[coords_axis].get_referencepixel())):
            if np.all(np.array(coordsA[coords_axis].get_referencevalue()) != np.array(coordsB[coords_axis].get_referencevalue())):
                raise ValueError('The reference values of the (x,y) direction corrdinates are different for images {0:s} and {1:s}'.format(
                                cimA.name(),cimB.name()))
        else:
            log.warning('The input images {0:s} and {1:s} have different (x,y) direction coordinate reference pixels!'.format(
                    cimA.name(),cimB.name()))
    else:
        log.warning('The input images {0:s} and {1:s} have different (x,y) direction coordinate units!'.format(
                    cimA.name(),cimB.name()))

    if close:
        log.debug('Closing image: {0:s}'.format(cimA.name()))
        log.debug('Closing image: {0:s}'.format(cimB.name()))
        del cimA
        del cimB

    return True

def set_CIM_unit(cimpath, unit, overwrite=False):
    """When a CASAImage is created using the ``casaimage.image()`` routine, the pixel unit of the image is empty by default.
    There is no way to set the unit by using the ``casacore.images`` module. However, we can workaround this by opening the
    image as a CASATable. Hooray. When no unit is defined the keyword *units* will be missing, hence we need to add this
    together with the unit value.

    Parameters
    ==========
    cimpath: str
        The input CASAImage path

    unit: str
        The unit of the image pixels e.g. Jy/Beam 

    overwrite: bool, optional
        If True, the existing unit is overwritten with the input ``unit`` parameter

    Returns
    ======= 
    Saves the image with the pixel unit included
    """
    CIMTable = ds.msutil.create_MS_object(cimpath,readonly=False)
    log.debug('Open image as CASATable: {0:s}'.format(CIMTable.name()))

    try:
        CIM_unit = CIMTable.getkeyword('units')
        if CIM_unit != unit and overwrite == False:
            log.warning('The image {0:s} already has a pixel unit: {1:s} that is different from the given unit: {2:s}!'.format(cimpath,CIM_unit,unit))
        else:
            CIMTable.putkeyword('units', unit)
    except:
        CIMTable.putkeyword('units', unit)

    log.debug('Close CASATable image: {0:s}'.format(CIMTable.name()))
    CIMTable.close()

def normalise_CIM(cimpath, output_name=None ,all_dim=True, chan=0, pol=0, overwrite=False):
    """This code can be used to normalize stacked non-normalized PSF CASAImages.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage path

    output_name: str, optional
        The full path and name of the output CASAImage created. If not given, the input image
        is being overwritten!

    all_dim: bool, optional
        If True, all channels and polarizations will be normalized. It is True by default!

    chan: int, optional
        Index of the channel in the image cube

    pol: int, optional
        Index of the polarization in the image cube

    overwrite: bool, optional
        If True, the normalized image will be created regardless if another image exist
        in the same name. Note, that in this case the existing image will be deleted!


    Returns
    ========
    Normalized image: CASAImage
        Create the stacked image at ``output_name``
    """
    cim = ds.cim.create_CIM_object(cimpath)

    #Get the peak
    cim_data_array = cim.getdata()

    if all_dim == True:
        for chan_i in range(0,np.shape(cim_data_array)[0]):
            for pol_j in range(0,np.shape(cim_data_array)[1]):
                peak_val = np.amax(cim_data_array[chan_i,pol_j,...])
                if peak_val <= 0:
                    raise ValueError('Peak value is equals or less than zero!')

                else:
                    cim_data_array[chan_i,pol_j,...] = np.divide(cim_data_array[chan_i,pol_j,...],peak_val)

    else:
        peak_val = np.amax(cim_data_array[chan,pol_j,...])
        if peak_val <= 0:
            raise ValueError('Peak value is equals or less than zero!')

        else:
            cim_data_array[chan_i,pol_j,...] = np.divide(cim_data_array[chan_i,pol_j,...],peak_val)

    #Write back to file
    if output_name != None:
        if overwrite == True and os.path.isdir(output_name):
            log.info('Delete old normalised image and create a new one at {0:s}'.format(output_name))
            shutil.rmtree(output_name)
        elif overwrite == False and os.path.isdir(output_name):
            raise ValueError('Overwrite is set to false, but the outpud directory exist at {0:s}'.format(output_name))

        shutil.copytree(cim.name(),output_name)

        #Have to delete to close the image and hence actually write into
        del cim

        cim = ds.cim.create_CIM_object(output_name)

    cim.putdata(cim_data_array)

    #Have to delete to close the image and hence actually write into
    del cim
    del cim_data_array

    return True

def create_CIM_diff_array(cimpath_a, cimpath_b, rel_diff=False, all_dim=False, chan=0, pol=0, close=False):
    """Compute the difference of two CASAImage, and return it as a numpy array.
    Either the entire difference cube, or only the difference of a selected channel 
    and polarization slice is returned.

    The code computes the first minus second image given, and normalizes with the
    second one if the rel_diff parameter is set to True.

    This function makes sense mostly on images not on grids!

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage path of Alice

    cimpath_b: str
        The input CASAImage path of Bob
    
    rel_diff: bool, optional
        If True, the relative difference is returned. The code uses Bob to normalize.

    all_dim: bool, optional
        If True, the difference across all channels and polarizations will be computed.
        Note taht it can be **very slow and memory heavy**!

    chan: int, optional
        Index of the channel in the image cube

    pol: int, optional
        Index of the polarization in the image cube

    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    Returns
    =======
    diff_array: numpy ndarray
        Either a single channel and polarization slice difference,
        or the difference cube of the two input CASAImages
    """
    cimA = ds.cim.create_CIM_object(cimpath_a)
    cimB = ds.cim.create_CIM_object(cimpath_b)
    
    CIM_dim_equity_check(cimA, cimB)

    if all_dim:
        if rel_diff:
            diff_array = np.divide(np.subtract(cimA.getdata(),cimB.getdata()),cimB.getdata())
        else:
            diff_array = np.subtract(cimA.getdata(),cimB.getdata())
    else:
        if rel_diff:
            diff_array = np.divide(np.subtract(cimA.getdata()[chan,pol,...],cimB.getdata()[chan,pol,...]),cimB.getdata()[chan,pol,...])
        else:
            diff_array = np.subtract(cimA.getdata()[chan,pol,...],cimB.getdata()[chan,pol,...])

    if close:
        log.debug('Closing image: {0:s}'.format(cimA.name()))
        log.debug('Closing image: {0:s}'.format(cimB.name()))
        del cimA
        del cimB            

    return diff_array

def process_CIM_chunk_RMS(image_slice,
                        pol,
                        robust,
                        percentile_cut,
                        window_halfsize=None,
                        window_centre=None):
    """Compute the RMS on a numpy ndarray with shape [N_chan,N_pol,N_x,N_y]
    It could be used to measure the RMS across all channels, but the input is 
    a numpy ndarray, and so it is used to compute the RMS on the subset of a
    CASAImage cube. I.e. this function called by each subprocess when computing
    the RMS parallely.

    This function computes the RMS across all channels of the input cube!

    Note, that a sclaing correction is missing from the robust RMS estimation!
    => this can cause erroreous results for large percentile cuts!

    Parameters
    ==========
    image_slice: numpy.ndarray
        A numpy array with size [N_chan,N_pol,N_x,N_y] where `N_chan` is 
        the channel number of the subset cube.

    pol: int
        The index of the polarisation axis
    
    robust: bool, optional
        RMS is really sensitive to utliers, so a robust way to measure RMS is to use only data between some percentilke valuesű
        e.g. ignore the top and bottom 10% of the data. If this value is set to True, this robust method is used to compute the RMS

    percentile_cut: int, optional
        If the RMS is computed via a robust method, this value gives the (upper) percentile value which below/above the data is being ignored.
        A small number <10% is recommended, as the code does not take scaling due to the cut into account

    window_halfsize: int, optional
        If not None, the RMS is only computed in a window defined in pixels, for all channels.
        This parameter is the halfsize of the window, that is the winow is
        2*window_halfsize + 1 in both x and y direction 

    window_centre: list, optional
        The [x,y] coordinates of the window centre.
        If None, the centre coordinates of the image are used.

    Return
    ======
    slice_RMS: numpy.array
        An array containing the RMS of each channel
    """

    slice_RMS = np.zeros((np.shape(image_slice)[0]))

    for i in range(0,len(slice_RMS)):
        if window_halfsize == None:
            image_data = image_slice[i,pol,...].flatten()
        else:
            window_halfsize = int(window_halfsize)
            if window_centre == None:
                window_centre = [np.floor(np.shape(image_slice)[2]/2),
                                np.floor(np.shape(image_slice)[3]/2)]

            image_data = image_slice[i,pol,
                int(window_centre[0]) - window_halfsize : int(window_centre[0]) + window_halfsize,
                int(window_centre[1]) - window_halfsize : int(window_centre[1]) + window_halfsize].flatten()

        if robust == True:
            lower_cut = np.percentile(image_data,percentile_cut)
            upper_cut = np.percentile(image_data,100 - percentile_cut)

            image_data = image_data[(image_data > lower_cut) & (image_data < upper_cut)]

        slice_RMS[i] = np.sqrt(np.mean(np.square(image_data)))

    return slice_RMS

def measure_CIM_RMS(cimpath,
                    all_dim=False,
                    chan=0,
                    chan_max=None,
                    pol=0,
                    robust=False,
                    percentile_cut=1,
                    window_halfsize=None,
                    window_centre=None,
                    return_dim=False,
                    close=False):
    """Measure the RMS on a CASAImage either for a given channel and polarization,
    or for ALL channels and polarizations. This could be very slow though. Also, currently
    works only for a single polarisation, however that can be selected. Menaing that the
    code works perfectly for SotokesI polarisation, but the user has to choose a polarisation
    axis if the image is polarised.

    The code runs parallely on all vailable nodes if more channels than nodes are selected.

    Parameters
    ==========
    cimgpath: str
        The input CASAImage path

    all_dim: bool, optional
        If True, the RMS will be computed for all channels and polarizations in the image cube
        Note that, this can be **very slow** even when parallelised!
        If set to True the `chan_max` param will be ignored!

    chan: int, optional
        Index of the channel in the image cube

    chan_max: int, optional
        Index of the upper channel limit if the RMS is computed only on subset of the imagechannels.
        Has to be larger than the :chan: variable, and smaéller than the number of channels.
        Can be negative using python indexing scheme, but it can result in an indexing error if specified poorly!

    pol: int, optional
        Index of the polarization in the image cube

    robust: bool, optional
        RMS is really sensitive to utliers, so a robust way to measure RMS is to use only data between some percentilke valuesű
        e.g. ignore the top and bottom 10% of the data. If this value is set to True, this robust method is used to compute the RMS

    percentile_cut: int, optional
        If the RMS is computed via a robust method, this value gives the (upper) percentile value which below/above the data is being ignored.
        A small number <10% is recommended, as the code does not take scaling due to the cut into account

    window_halfsize: int, optional
        If not None, the RMS is only computed in a window defined in pixels, for all channels.
        This parameter is the halfsize of the window, that is the winow is
        2*window_halfsize + 1 in both x and y direction 

    window_centre: list, optional
        The [x,y] coordinates of the window centre.
        If None, the centre coordinates of the image are used.

    return_dim: bool, optional
        If true, a second variable: a string defining the image pixel units, i.e. the RMS units is returned

    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    Returns
    =======
    rms_array: list of floats
        The RMS value for the given channel or a numpy ndarray
        containing the RMS for the corresponding channel and polarization
    
    """
    #parallel solutions explanation here: https://datascience.blog.wzb.eu/2018/02/02/vectorization-and-parallelization-in-python-with-numpy-and-pandas/

    cim = ds.cim.create_CIM_object(cimpath)

    rms_dim = cim.unit()

    pol = int(pol) #If the input pol is a float... this is a bad bug fix...

    if chan_max == None and all_dim == False:
        #Single channel mode but using the same syntax (does not work for last channel I think)
        rms = np.array([process_CIM_chunk_RMS(cim.getdata()[chan:chan+1,...],
                                            pol=pol,
                                            robust=robust,
                                            percentile_cut=percentile_cut,
                                            window_halfsize = window_halfsize,
                                            window_centre = window_centre)])
        if close:
            log.debug('Closing image: {0:s}'.format(cim.name()))
            del cim

        if return_dim:
            return rms, rms_dim
        else:
            return rms

    else:
        #Get the data array as a variable, because I think this is more memory effective.
        #By this the data is read only once and shared across the procresses (?)

        cim_data = cim.getdata()

        if all_dim == True:
            chan = 0
            chan_max = -1
            window_size = np.shape(cim_data)[0] #Number of channels
        
        else:
            if chan_max <= chan and chan_max > 0:
                raise ValueError('Invalid lower ({0:d}) and upper ({1:d}) channel indices are given!'.format(chan,chan_max))

            if np.fabs(chan_max) > get_N_chan_from_CIM(cim):
                raise ValueError('Upper channel index {0:d} is out from the channel range of the image!'.format(chan_max))

            #Get subband size in channels
            window_size = chan_max - chan
    
        #Setup the output array
        rms_array = np.zeros((window_size))

        #Decide if worth processing parallely:
        n_proc = mp.cpu_count()

        if n_proc <= window_size:
            log.info('Running RMS calculations parallel on {0:d} nodes'.format(n_proc))

            chunksize = window_size // n_proc #There will be a remainder        
            log.debug('Computing RMS using {0:d} processes: {1:d} channels (out of {2:d} channels) for each subprocress'.format(n_proc,chunksize,window_size))

            proc_chunks = []
            for i_proc in range(n_proc):
                chunkstart = i_proc * chunksize
                # make sure to include the division remainder for the last process
                chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else window_size

                #Here numpy only passes a view instead of the whole object: https://numpy.org/doc/stable/reference/arrays.indexing.html
                #=> this is why I read in the whole cube to memory in the beginning
                proc_chunks.append(cim_data[chan + chunkstart : chan + chunkend,...])

            with mp.Pool(processes=n_proc) as pool:
                # starts the sub-processes without blocking
                # pass the chunk to each worker process
                proc_results = [pool.apply_async(process_CIM_chunk_RMS,
                                                 args=(chunk,pol,robust,percentile_cut))
                                for chunk in proc_chunks]

                # blocks until all results are fetched
                result_chunks = [r.get() for r in proc_results]

            #Result chunks is a list containing numpy arrays => need to flatten
            rms_array = copy.deepcopy(np.concatenate(result_chunks).ravel())#ravel returns a view so need to copy

            #clean up
            del result_chunks
            del proc_chunks
            del proc_results

        else:
            #Process channels serially
            rms_array = process_CIM_chunk_RMS(cim_data[chan:chan_max,...],
                                            pol=pol,
                                            robust=robust,
                                            percentile_cut=percentile_cut,
                                            window_halfsize = window_halfsize,
                                            window_centre = window_centre)

    if close:
        log.debug('Closing image: {0:s}'.format(cim.name()))
        del cim

    if return_dim:
        return rms_array, rms_dim
    else:
        return rms_array

def measure_CIM_max(cimpath, save_to_file=False, outputfile_path=None, ID=0, all_dim=False, chan=0, pol=0, close=False):
    """Function to measure the peak of a CASAIMage, usually an image of the synthesized beam (PSF).

    For averaging of images, it is needed to get the RMS or the PSF peak to weight the images. The latter
    does not take potential RFI into account, i.e. it only results in sensitivity (integration time) weighted averaging.

    This function is primarily designed to support such an averaging.


    Parameters
    ==========
    cimgpath: str
        The input CASAImage path

    save_to_file: bool
        If True, the results will be saved to a file specified by ``outputfile_path``

    outputfile_path: str, optional
        Abspath, (path+name) of the output file where the max values are saved. It only works for a single polarization at the moment!
        The file is appended if already exist. So multiple day's PSF max can be written into one master file for stacking. Each row is
        a day and each column is a channel. This code creates a row.

    ID: int, optional
        An ID which can be used to identify each CIM added to the ``outputfile_path`` file.

    all_dim: bool, optional
        If True, the peak value will be measured for all channels and polarizations in the image cube

    chan: int, optional
        Index of the channel in the image cube

    pol: int, optional
        Index of the polarization in the image cube

    close: bool, optional
        If True the in-memory CASAIMage is deleted, and the optional write-lock releases
        Set to true if this is the last operation on the image, but False if other functions
        called that operation on the same image. This avoids multiple read-in of the image.

    Returns
    =======
    peak: float or list of floats
        The peak value for the given channel or a numpy ndarray
        containing the peak for the corresponding channel and polarization
    
    peak_in_a-file: file
        Append a file with a row of max values
    """
    cim = ds.cim.create_CIM_object(cimpath)

    if save_to_file == True and outputfile_path == None:
        raise ValueError('No output file path is defined, can not save peak(s) of {0:s}'.format(cim.name()))


    if all_dim:
        peak_matrix = np.zeros((cim.shape()[0],cim.shape()[1]))

        for chan_i in range(0,cim.shape()[0]):
            for pol_j in range(0,cim.shape()[1]):
                peak_matrix[chan_i,pol_j] = np.amax(cim.getdata()[chan_i,pol_j,...])

        if close:
            log.debug('Closing image: {0:s}'.format(cim.name()))
            del cim
        
        if save_to_file == True:
            if cim.shape()[1] > 1:
                raise ValueError('Image polarization is > 1, i.e. polarized instead of intensity only image {0:s}'.format(cim.name()))

            else:
                #Append a file with a line of the following format: ID, peak1, peak2, .. peakN \n
                peak_matrix_string = '{0:f}, '.format(ID) + ', '.join([str(peak_matrix[p,0]) for p in range(0,np.shape(peak_matrix)[0])])
                peak_output = open(outputfile_path, 'a')
                peak_output.write(peak_matrix_string + '\n')
                peak_output.close()

                return True

        else:
            return peak_matrix

    else:
        peak = np.amax(cim.getdata()[chan,pol,...])
        
        if close:
            log.debug('Closing image: {0:s}'.format(cim.name()))
            del cim
        
        if save_to_file == True:
            if cim.shape()[1] > 1:
                raise ValueError('Image polarization is > 1, i.e. polarized instead of intensity only image {0:s}'.format(cim.name()))

            else:
                #Append a file with a line of the following format: ID, peak1 \n
                peak_string = '{0:f}, {1:f}'.format(ID, peak)
                peak_output = open(outputfile_path, 'a')
                peak_output.write(peak_string + '\n')
                peak_output.close()

                return True

        else:
            return peak

def CIM_stacking_base(cimpath_list, cim_output_path, cim_outputh_name, normalise=False, weight_with_psf=False, psfpath_list=None, psf_peaks_log_path=None, overwrite=False, close=False):
    """This function is one of the core functions of the image stacking stacking deep spectral line pipelines.

    This function takes a list of CASAImages and creates the stacked CASAIMage.
    The resultant image can be a simple sum or an average.

    The given images have to have the same:
        - shape
        - coordinates
        - pixel value unit (e.g. Jy/beam)

    NOTE, that there are better tools in YadaSoft and casacore to create stacked images,
    but no option to stack and modify grids.

    This function currently supports the following stacking options:
        - simple sum (for grid stacking)
        - simple average
        - weighted sum using a set of PSF peaks (the max of the given PSF images for each channel)
    
    TO DO:
        - add an RMS-weighted stacking option
        - write the stacked numpy data array parallel where each node reads in an MS \
    and append the stacked data.
        - same as above, but for the weighting subroutine
        - modularize the stack functions
        - use dask for parallelisation and distrubution

    Currently this function is no bottleneck and in the pipelines the corresponding rule can be further modularised
    (see beam17 pipeline). However, parallelisation and distributed processing would be a huge speedup for a 
    pipeline where wide-band grid cubes are created.

    Parameters
    ==========
    cimpath_list: list
        A list of the full paths of the images to be stacked
    
    cim_output_path: str
        Full path to the folder in which the stacked image will be saved

    cim_outputh_name: str
        Name of the stacked image

    normalise: bool, optional
        If True, the images will be averaged instead of just summing them, if ``weight_with_psf`` is set to True,
        that overwrites this option!
    
    weight_with_psf: bool, optional
        If True, a list of PSF are used for weighting. The PSF should be un-normalised i.e. the peak should be the 
        weighted visibility sum. Otherwise, the results are equivalent to use the ``normalise`` parameter, as the
        normalised PSF' are used. If the non-normalised PSF' are used, it corresponds to normalise with sensitivity
        of each data set. NOTE, that this way of normalisation does not include the effect of RFI for example!
    
    psfpath_list: list, optional
        A list of the full pats of the PSF' used for weighting. The order of the PSF' in the list has to be consistent
        with the images in ``cimpath_list``. I.e. the Nth image will be weighted by the Nth PSF. Each image have to have
        a corresponding PSF.

    psf_peaks_log_path: str, optional
        Full path (including filename), of a potential logfile. If given, two logfile is created when weighting the stacked
        images by the corresponding PSF. The logfile at ``psf_peaks_log_path`` contains an ID and the PSF peaks for each channel
        that are used for weighting. The ``psf_peaks_log_path.indices`` is the second file created, and it
        contains an ID and the full path of the image and psf used in conjunction during weighted stacking.

    overwrite: bool, optional
        If True, the stacked image will be created regardless if another image exist
        in the same name. Note, that in this case the existing image will be deleted!
    
    close: bool, optional
        If True the in-memory CASAIMages given by ``cimpath_list`` are deleted,
        and the optional write-lock releases.
        Set to true if this is the last operation on the images, but False if other functions
        called that operation on the same images. This avoids multiple read-in of the image.
    
    Returns
    ========
    Stacked image: CASAImage
        Create the stacked image at ``cim_output_path/cim_outputh_name``
    """
    if len(cimpath_list) < 2:
        raise ValueError('Less than two image given for stacking!')

    log.info('Stacking {0:d} images together.'.format(len(cimpath_list)))

    if weight_with_psf == True and psfpath_list == None or weight_with_psf and len(cimpath_list) != len(psfpath_list):
        raise ValueError('The psfpath_list is not given or not the right amount of psf are provided!')

    if weight_with_psf == True and normalise == True:
        normalise = False
        log.info('Use the provided PSF list for weighting (i.e. weighting by sensitivity), set weight_with_psf to False. \
                 If you want to normalize only by the number of nights combined!')
    elif weight_with_psf == False and normalise == True:
        log.info('Use averaging in stacking.')

    elif weight_with_psf == False and normalise == False:
        log.info('Sum the images in stacking.')

    if weight_with_psf == True and psf_peaks_log_path != None:
        log.info('Weighting images by sensitivity (peak of the PSF). The logfiles are created with base name at {0:s}'.format(psf_peaks_log_path))
    else:
        log.info('Weighting images by sensitivity (peak of the PSF). No logfiles are created!')

    output_cim = '{0:s}/{1:s}'.format(cim_output_path,cim_outputh_name)

    if os.path.isdir(output_cim) and overwrite == False: 
        raise TypeError('Stacked image already exist, and the overwrite parameters is set to False!') 

    base_cim = ds.cim.create_CIM_object(cimpath_list[0])

    #Coordinate system is initialized by the first CASAImages coordinate system
    coordsys = base_cim.coordinates()

    check_attrgroup_empty = lambda x: None if x.attrgroupnames() == [] else log.warning('Input image {0:s} has a non-empty attribute list!'.format(x.name()))
    check_history_empty = lambda x: None if x.history() == [] else log.warning('Input image {0:s} has a non-empty history field!'.format(x.name()))

    check_attrgroup_empty(base_cim)
    check_history_empty(base_cim)

    #If shape is given, the data type is automatically set to float!
    log.debug('Create stacked image: {0:s}'.format(output_cim))
    stacked_cim = casaimage.image(output_cim,
                    coordsys=coordsys,
                    values=base_cim.getdata(),
                    overwrite=overwrite)

    #=== Weighting with PSF ===
    #Define a closure function to weight with the PSF
    def use_psf_weight(input_cim,input_psf,data_ID):
        """A closure function which computes the weighted image data array and the PSF peak list.

        This code checks if the given image and PSF have the same coordinates. Furthermore,
        if given, a logfile is created at ``psf_peaks_log_path`` were each row starts with a dataset ID
        followed by the PSF peak values for each frequency channel.

        A mapping file also created at ``psf_peaks_log_path.indices`` that contains for each row the
        dataset ID and the image and psf paths, which are corresponding to that ID at ``psf_peaks_log_path``

        Parameters
        ==========
        input_cim: str
            The input CASAImage path that needs to be weighted for stacking
        
        input_psf: str
            The input PSF path that is used for the weighting

        data_ID: float
            A unique ID used fin logging if ``psf_peaks_log_path`` is defined
        
        Returns
        ========
        weighted_input_cim_data: :numpy.ndarray:
            The image matrix, which channels are multiplied by the peak of the corresponding channels of the PSF
    
        psf_peak_matrix: :numpy.ndarray:
            A matrix with the shape of [N_chan,1]. Each entry contains the peak (max) value of the PSF
        
        psf_peaks_log_path: file
            A file containing the data_ID and the PSF peaks for each channel in every row. This function only appends the logfile
            with the corresponding row. Only created if the ``psf_peaks_log_path`` variable is defined in the parent function.

        psf_peaks_log_path.indices: file
            A file containing the data_ID and the full path of the image and PSF used in stacking.  
            This function only appends the logfile with the corresponding row. 
            Only created if the ``psf_peaks_log_path`` variable is defined in the parent function.
        """
        if ds.cim.check_CIM_coordinate_equity(input_cim,input_psf,iclude_pixel_unit=False) == False:
            raise ValueError('The input image image and the corresponding psf ({0:s} and {1:s}) have different coordinate systems!'.format(input_cim.name(),input_psf.name()))

        #Get the peak matrix
        psf_peak_matrix = measure_CIM_max(cimpath=input_psf, all_dim=True)

        weighted_input_cim_data = input_cim.getdata()

        #Should make this bit parallel as well
        #Also, easy to add a polarization axis in the future if needed
        #Also, should define outside as a separate function
        for i in range(0,np.shape(weighted_input_cim_data)[0]):
            weighted_input_cim_data[i,0,...] = np.multiply(weighted_input_cim_data[i,0,...],psf_peak_matrix[i,0])

        #Create two logfiles 
        #This variable is defined in the parent function and accessed here directly and not passed as an argument!
        if psf_peaks_log_path != None:
            measure_CIM_max(cimpath=input_psf, all_dim=True, save_to_file=True, ID=data_ID, outputfile_path=psf_peaks_log_path, close=False)

            #Format: ID, image_path, psf_path
            peak_output = open(psf_peaks_log_path + '.indices', 'a')
            peak_output.write('{0:f}, {1:s}, {2:s}'.format(data_ID,input_cim.name(),input_psf.name()) + '\n')
            peak_output.close()

        del input_psf

        return weighted_input_cim_data, psf_peak_matrix

    #Keep the data in memory before deleting the first cim and psf
    if weight_with_psf:
        base_psf = ds.cim.create_CIM_object(psfpath_list[0])
        
        weighted_cim_data, psf_peak_matrix = use_psf_weight(base_cim,base_psf, data_ID=0.)

        stacked_cim_data = copy.deepcopy(weighted_cim_data)
        stacked_psf_peak_matrix = copy.deepcopy(psf_peak_matrix)
    else:
        stacked_cim_data = base_cim.getdata()

    #Close the image so the unit can be set
    del stacked_cim

    #Set the unit of the resultant image based on the first image
    ds.cim.set_CIM_unit(output_cim, base_cim.unit())

    #Read back the stacked image
    stacked_cim = ds.cim.create_CIM_object(output_cim)

    #=== Stacking ===
    for i in range(1,len(cimpath_list)):
        cim = ds.cim.create_CIM_object(cimpath_list[i])

        if base_cim.datatype() != cim.datatype():
            raise TypeError('The data type of the two input images ({0:s} and {1:s}) are not equal!'.format(base_cim.name(),cim.name()))
        if ds.cim.check_CIM_coordinate_equity(cim,stacked_cim) == False:
            raise ValueError('The created stacked image and the image {0:s} have different coordinate systems!'.format(cim.name()))
        
        check_attrgroup_empty(cim)
        check_history_empty(cim)

        if weight_with_psf:
            psf = ds.cim.create_CIM_object(psfpath_list[i])
            weighted_cim_data, psf_peak_matrix = use_psf_weight(cim, psf, data_ID=float(i))

            stacked_cim_data = np.add(stacked_cim_data, weighted_cim_data)
            stacked_psf_peak_matrix = np.add(stacked_psf_peak_matrix, psf_peak_matrix)

        else:
            stacked_cim_data = np.add(stacked_cim_data, cim.getdata())

        if close:
            log.debug('Closing image: {0:s}'.format(cim.name()))
            del cim

    #=== Normalize ===
    if normalise:
        stacked_cim_data = np.divide(stacked_cim_data,len(cimpath_list))

    if weight_with_psf:
        for i in range(0,np.shape(stacked_cim_data)[0]):
            stacked_cim_data[i,0,...] = np.divide(stacked_cim_data[i,0,...],stacked_psf_peak_matrix[i,0])

    log.debug('Write the stacked data to {0:s}'.format(output_cim))
    stacked_cim.putdata(stacked_cim_data)

    #Deleting the CIM variable closes the image, which release the lock
    log.debug('Closing image: {0:s}'.format(stacked_cim.name()))
    log.debug('Closing image: {0:s}'.format(output_cim))
    
    del stacked_cim
    del output_cim

    if close:
        log.debug('Closing image: {0:s}'.format(base_cim.name()))
        del base_cim

if __name__ == "__main__":
    pass
