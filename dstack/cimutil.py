"""
Collection of utility functions to interact with CASA images. 
Grids are currently dumped in casaimage format, hence the utilities works at both images and grids.
"""

__all__ = ['create_CIM_object', 'get_N_chan_from_CIM', 'get_N_pol_from_CIM',
            'check_CIM_equity', 'check_CIM_coordinate_equity']

import numpy as np
import warnings

from casacore import images as casaimage

import dstack as ds

def create_CIM_object(cimpath):
    """This function aims to speed up other bits of the ``cimutil``, ``cim`` and ``cgrid``
    modules, by returning a ``casacore.images.image.image`` object.

    The trick is, that the ``cimpath`` argument can be either a string i.e. the path
    to the CASAImage wich will be read in and returned, **or** it can be already an
    in-memory ``casacore.images.image.image`` object.

    This might not be the best solution, but I hope overall a check in a lot of cases will
    speed up code, rather than readiung in the same CASAImage again-and again. So ideally, only
    one reading in happends for each CASAImage and all inside this function!

    Parameters
    ==========
    cimpath: str
        The input CASAImage parth or a ``casacore.images.image.image`` object

    Returns
    =======
    cim: ``casacore.images.image.image`` object
        The in-memory CASAImage

    """
    #create an empty image in-memory to check the object type
    if type(cimpath) == type(casaimage.image(imagename='',shape=np.ones(1))):
        return cimpath
    else:
        cim = casaimage.image(cimpath)
        return cim

def get_N_chan_from_CIM(cimpath):
    """Get the number of channels from a CASAImage

    CASAImage indices: [freq, Stokes, x, y]

    Parameters
    ==========
    cimpath: str
        The input CASAImage parth or a ``casacore.images.image.image`` object
    
    Returns
    =======
    N_chan: int
        Number of channels in the CASAImage
    """
    cim = ds.cimutil.create_CIM_object(cimpath)
    assert cim.ndim() == 4, 'The image has more than 4 axes!'

    N_chan = np.shape(cim.getdata())[0]

    return N_chan

def get_N_pol_from_CIM(cimpath):
    """Get the number of polarisations from a CASAImage. Note, that the
    polarisation type is not returned!

    CASAImage indices: [freq, Stokes, x, y]

    Parameters
    ==========
    cimpath: str
        The input CASAImage parth or a ``casacore.images.image.image`` object
    
    Returns
    =======
    N_pol: int
        Number of polarisations in the CASAImage
    """
    cim = ds.cimutil.create_CIM_object(cimpath)
    assert cim.ndim() == 4, 'The image has more than 4 axes!'

    N_pol = np.shape(cim.getdata())[1]

    return N_pol

def check_CIM_equity(cimpath_a,cimpath_b,numprec=1e-8):
    """Check if two CASAImages are identical or not up to a defined numerical precision
    This function is used to test certain piepline features.

    Note: NaN-s are trethed as equals, and the ``numprec`` parameter only sets the relative difference limit.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage path of Alice or a ``casacore.images.image.image`` object

    cimpath_b: str
        The input CASAImage path of Bob or a ``casacore.images.image.image`` object

    numprec: float
        The numerical precision limit of the maximum allowed relative
        difference between CASAImages Alice and Bob.
        If set to zero, equity is checked.

    Returns
    =======
    equity: bool
        True or False, base on the equity of Alice and Bob
    """

    cimA = ds.cimutil.create_CIM_object(cimpath_a)
    cimB = ds.cimutil.create_CIM_object(cimpath_b)

    assert cimA.ndim() == cimB.ndim(), 'The dimension of the two input CASAImage is not equal!'

    if numprec == 0.:
        return np.array_equiv(cimA.getdata(),cimB.getdata())
    else:
        return np.allclose(cimA.getdata(),cimB.getdata(),atol=0,rtol=numprec,equal_nan=True)

def check_CIM_coordinate_equity(cimpath_a,cimpath_b):
    """Basic cehck if the associated coordinate information of two images are somewhat equal.
    This is **not** an equity check for all coordinate values, as the reference pixels can be differnt,
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

    Returns
    =======
    equity: bool
        True or False, base on the coordinate equity of Alice and Bob
    """
    cimA = ds.cimutil.create_CIM_object(cimpath_a)
    cimB = ds.cimutil.create_CIM_object(cimpath_b)

    assert cimA.ndim() == cimB.ndim(), 'The dimension of the two input CASAImage is not equal!'

    coordsA = cimA.coordinates()
    coordsB = cimA.coordinates()

    #Spectral coordinates
    coords_axis = 'spectral'

    assert coordsA[coords_axis].get_frame() == coordsB[coords_axis].get_frame(), \
    'The given images {0:s} and {1:s} have different frames!'.format(cimA.name(),cimB.name())

    if coordsA[coords_axis].get_unit() == coordsB[coords_axis].get_unit():
        assert coordsA[coords_axis].get_increment() == coordsB[coords_axis].get_increment(), \
        'The increment of the two spectral coordinates are different for images {0:s} and {1:s}'.format(
        cimA.name(),cimB.name())
        
        assert coordsA[coords_axis].get_restfrequency() == coordsB[coords_axis].get_restfrequency(), \
        'The rest frame frequency of the two spectral coordinates are different for images {0:s} and {1:s}'.format(
        cimA.name(),cimB.name())

        if coordsA[coords_axis].get_referencepixel() == coordsB[coords_axis].get_referencepixel():
            assert coordsA[coords_axis].get_referencevalue() == coordsB[coords_axis].get_referencevalue(), \
            'The reference values of the spectral corrdinates are different for images {0:s} and {1:s}'.format(
            cimA.name(),cimB.name())
        else:
            warnings.warn('The input images {0:s} and {1:s} have different spectral coordinate reference pixel!'.format(
                    cimA.name(),cimB.name()))
    else:
        warnings.warn('The input images {0:s} and {1:s} have different spectral coordinate units!'.format(
                    cimA.name(),cimB.name()))

    #Polarisation coordinates
    coords_axis = 'stokes'

    assert coordsA[coords_axis].get_stokes() == coordsB[coords_axis].get_stokes(), \
    'The polarisation frame is different for images {0:s} and {1:s}!'.format(cimA.name(),cimB.name())

    #Direction coordinates if images and linear coordinates if grids
    coords_axis = 'direction'
    try:
        assert coordsA[coords_axis].get_frame() == coordsB[coords_axis].get_frame(), \
        'The given images {0:s} and {1:s} have different frames!'.format(cimA.name(),cimB.name())

        assert coordsA[coords_axis].get_projection() == coordsB[coords_axis].get_projection(), \
        'The given images {0:s} and {1:s} have different projections!'.format(cimA.name(),cimB.name())

    except AssertionError:
        #re-run the assertion to actually fail the code
        assert coordsA[coords_axis].get_frame() == coordsB[coords_axis].get_frame(), \
        'The given images {0:s} and {1:s} have different frames!'.format(cimA.name(),cimB.name())

        assert coordsA[coords_axis].get_projection() == coordsB[coords_axis].get_projection(), \
        'The given images {0:s} and {1:s} have different projections!'.format(cimA.name(),cimB.name())
    
    except:
        #Change to linear coord as the given CASAimage is a grid!
        coords_axis = 'linear'

    if np.all(np.array(coordsA[coords_axis].get_unit()) == np.array(coordsB[coords_axis].get_unit())):
        assert np.all(np.array(coordsA[coords_axis].get_increment()) == np.array(coordsB[coords_axis].get_increment())), \
        'The increment of the (x,y) direction coordinates are different for the input images {0:s} and {1:s}'.format(
        cimA.name(),cimB.name())

        if np.all(np.array(coordsA[coords_axis].get_referencepixel()) == np.array(coordsB[coords_axis].get_referencepixel())):
            assert np.all(np.array(coordsA[coords_axis].get_referencevalue()) == np.array(coordsB[coords_axis].get_referencevalue())), \
            'The reference values of the (x,y) direction corrdinates are different for images {0:s} and {1:s}'.format(
            cimA.name(),cimB.name())
        else:
            warnings.warn('The input images {0:s} and {1:s} have different (x,y) direction coordinate reference pixels!'.format(
                    cimA.name(),cimB.name()))
    else:
        warnings.warn('The input images {0:s} and {1:s} have different (x,y) direction coordinate units!'.format(
                    cimA.name(),cimB.name()))

    return True

if __name__ == "__main__":
    CIMpath_A = '/home/krozgonyi/Desktop/first_pass/image.restored.wr.1.sim_PC'
    CIMpath_B = '/home/krozgonyi/Desktop/first_pass/image.restored.wr.1.sim_PC'

    create_CIM_object(CIMpath_A)

    #exit()

    check_CIM_coordinate_equity(CIMpath_A,CIMpath_B)