"""
Collection of functions operating on images in CASAImageformat.
The functions defined here are expected to work on grids as well, as grids are currently dumped in casaimage format,
hence functions defined in this module works on both images and grids.
"""

__all__ = ['create_CIM_object', 'get_N_chan_from_CIM', 'get_N_pol_from_CIM',
            'measure_CIM_RMS', 'check_CIM_equity', 'check_CIM_coordinate_equity', 
            'create_CIM_diff_array', 'CIM_stacking_base']

import os
import shutil
import numpy as np
import warnings

from casacore import images as casaimage

import dstack as ds

def create_CIM_object(cimpath):
    """This function aims to speed up other bits of this and ``cgrid``
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
    cim = ds.cim.create_CIM_object(cimpath)
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
    cim = ds.cim.create_CIM_object(cimpath)
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

    cimA = ds.cim.create_CIM_object(cimpath_a)
    cimB = ds.cim.create_CIM_object(cimpath_b)

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
    cimA = ds.cim.create_CIM_object(cimpath_a)
    cimB = ds.cim.create_CIM_object(cimpath_b)

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

def create_CIM_diff_array(cimpath_a,cimpath_b,rel_diff=False,all_dim=False,chan=0,pol=0):
    """Compute the difference of two CASAImage, and return it as a numpy array.
    Either the entire difference cube, or only the difference of a selected channel 
    and polarisation slice is returned.

    The code computes the first minus second image given, and normalises with the
    second one if the rel_diff parameter is set to True.

    This function maskes sense mostly on images not on grids, that is why I put it under this module.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage parth of Alice

    cimpath_b: str
        The input CASAImage parth of Bob
    
    rel_diff: bool
        If True, the relative difference is returned. The code uses Bob to normalise.

    all_dim: bool
        If True, the difference across all channels and polarisations will be computed.
        Note taht it can be **very slow and memory heavy**!

    chan: int
        Index of the channel in the image cube

    pol: int
        Index of the polarisation in the image cube

    Returns
    =======
    diff_array: numpy ndarray
        Either a single channel and polarisation slice difference,
        or the difference cube of the two input CASAImages
    """
    cimA = ds.cim.create_CIM_object(cimpath_a)
    cimB = ds.cim.create_CIM_object(cimpath_b)

    assert cimA.ndim() == cimB.ndim(), 'The dimension of the two input CASAImage is not equal!'

    if all_dim:
        if rel_diff:
            return np.divide(np.subtract(cimA.getdata(),cimB.getdata()),cimB.getdata())
        else:
            return np.subtract(cimA.getdata(),cimB.getdata())
    else:
        if rel_diff:
            return np.divide(np.subtract(cimA.getdata()[chan,pol,...],cimB.getdata()[chan,pol,...]),cimB.getdata()[chan,pol,...])
        else:
            return np.subtract(cimA.getdata()[chan,pol,...],cimB.getdata()[chan,pol,...])


def measure_CIM_RMS(cimpath,all_dim=False,chan=0,pol=0):
    """Measure the RMS on a CASAImage either for a given channel and polarisation,
    or for ALL channels and polarisations. This could be very slow though.

    Parameters
    ==========
    cimgpath: str
        The input CASAImage parth

    all_dim: bool
        If True, the RMS will be computed for all channels and polarisations in the image cube
        Note that, this can be **very slow**!

    chan: int
        Index of the channel in the image cube

    pol: int
        Index of the polarisation in the image cube

    Returns
    =======
    rms: float or list of floats
        The RMS value for the given channel or a numpy ndarray
        containing the RMS for the corresponding channel and ploarisation
    
    """
    cim = ds.cim.create_CIM_object(cimpath)

    if all_dim:
        rms_matrix = np.zeros((cim.shape()[0],cim.shape()[1]))

        for chan_i in range(0,cim.shape()[0]):
            for pol_j in range(0,cim.shape()[1]):
                rms_matrix[i,j] = np.sqrt(np.mean(np.square(cim.getdata()[chan_i,pol_j,...])))

        return rms_matrix

    else:
        return np.sqrt(np.mean(np.square(cim.getdata()[chan,pol,...])))

def CIM_stacking_base(cimpath_list,cim_output_path,cim_outputh_name,normalise=False,overwrite=False):
    """This function is one of the core functions of the imge stacking stacking deep spectral line pipelines.

    This function takes a list of CASAImages and creates the stacked CASAIMage.
    The resultant image can be a simple sum or an average.

    The given imaes have to have the same:
        - shape
        - coordinates

    NOTE, that there are better tools in YadaSoft and casacore to cretate stacked images.

    Parameters
    ==========
    cimpath_list: list
        A list of the full paths of the images to be stacked
    
    cim_output_path: str
        The full path to the folder in which the stacked image will be saved

    cim_outputh_name: str
        The name of the stacked imae

    normalise: bool
        If True, the images will be averaged instead of just summing them
    
    overwrite: bool
        If True, the stacked image will be created regardless if another image exist
        in the same name. Note, that in this case the existing grid will be deleted!

    Returns
    ========
    Stacked image: CASAImage
        Create the stacked image at ``cim_output_path/cim_outputh_name``

    """
    assert len(cimpath_list) >= 2, 'Less than two image given for stacking!'

    output_cim = '{0:s}/{1:s}'.format(cim_output_path,cim_outputh_name)

    if os.path.isdir(output_cim): assert overwrite, 'Stacked image already exist, and the overwrite parameters is set to False!'

    base_cim = ds.cim.create_CIM_object(cimpath_list[0])

    #Coordinate system is initialised by the first CASAImages coordinate system
    coordsys = base_cim.coordinates()

    check_attrgroup_empty = lambda x: None if x.attrgroupnames() == [] else warnings.warn('Input image {0:s} has a non-empty attribute list!'.format(x.name()))
    check_history_empty = lambda x: None if x.history() == [] else warnings.warn('Input image {0:s} has a non-empty history field!'.format(x.name()))

    check_attrgroup_empty(base_cim)
    check_history_empty(base_cim)

    #If shape is given, the data type is automatically set to float!
    stacked_cim = casaimage.image(output_cim,
                    coordsys=coordsys,
                    values=base_cim.getdata(),
                    overwrite=overwrite)

    stacked_cim_data = base_cim.getdata()

    for i in range(1,len(cimpath_list)):
        cim = ds.cim.create_CIM_object(cimpath_list[i])

        assert base_cim.datatype() == cim.datatype(), 'The data type of the two input images ({0:s} and {1:s}) are not equal!'.format(base_cim.name(),cim.name())
        assert base_cim.ndim() == cim.ndim(), 'The dimension of the two input images ({0:s} and {1:s}) are not equal!'.format(base_cim.name(),cim.name())

        #This is slow as it reads in the grids again!
        assert ds.cim.check_CIM_coordinate_equity(cim,stacked_cim), \
        'The created stacked grid and the grid {0:s} have different coordinate systems!'.format(cim.name())

        check_attrgroup_empty(cim)
        check_history_empty(cim)

        stacked_cim_data = np.add(stacked_cim_data, cim.getdata())

    if normalise:
        stacked_cim_data = np.divide(stacked_cim_data,len(cimpath_list))

    stacked_cim.putdata(stacked_cim_data)


if __name__ == "__main__":
    CIM_stacking_base(['/home/krozgonyi/Desktop/list_imaging_test/dumpgrid_first_night/image.restored.test',
                    '/home/krozgonyi/Desktop/list_imaging_test/dumpgrid_second_night/image.restored.test'],
                    '/home/krozgonyi/Desktop','a.image', normalise=True,overwrite=True)
    
    import logging;
    import sys
    log = logging.getLogger();

    log.setLevel(logging.INFO);
    log.addHandler(logging.StreamHandler(sys.stdout));

    def log_image_diff_RMS(type_string_one,type_string_two,imager_string_one,imager_string_two,image_one,image_two,string_one,string_two):
        rms = np.sqrt(np.mean(np.square(ds.cim.create_CIM_diff_array('{0:s}{1:s}/{2:s}'.format(type_string_one,imager_string_one,image_one),
                                '{0:s}{1:s}/{2:s}'.format(type_string_two,imager_string_two,image_two),
                                rel_diff=False,all_dim=False,chan=0,pol=0))))

        log.info("{0:s} -- {1:s}: {2:.4e}".format('{0:s}: {1:s}'.format(type_string_one,string_one),string_two,rms))

    def log_image_equity(rprec,type_string_one,type_string_two,imager_string_one,imager_string_two,image_one,image_two,string_one,string_two):
        log.info("{0:s} -- {1:s}: {2:s}".format('{0:s}: {1:s}'.format(type_string_one,string_one),string_two,
            str(ds.cim.check_CIM_equity('{0:s}{1:s}/{2:s}'.format(type_string_one,imager_string_one,image_one),
                                '{0:s}{1:s}/{2:s}'.format(type_string_two,imager_string_two,image_two),numprec=rprec))))

        if ds.cim.check_CIM_equity('{0:s}{1:s}/{2:s}'.format(type_string_one,imager_string_one,image_one),'{0:s}{1:s}/{2:s}'.format(type_string_two,imager_string_two,image_two),numprec=rprec) == False:
            log_image_diff_RMS(type_string_one,type_string_two,imager_string_one,imager_string_two,image_one,image_two,string_one,string_two)

    import matplotlib.pyplot as plt
    def show_difference_slice(ima,imb,chan=0,pol=0):
        diffmap = np.real(ds.cim.create_CIM_diff_array(ima,imb,rel_diff=False,all_dim=False,chan=chan,pol=pol))
        plt.matshow(diffmap,cmap='viridis')
        plt.colorbar()
        plt.show()
        plt.close()


    #log_image_equity(0,'/home/krozgonyi/','/home/krozgonyi/','Desktop','Desktop/list_imaging_test/dumpgrid_list_ms','a.image','image.restored.test','stacked','list')
    #log_image_equity(0,'/home/krozgonyi/','/home/krozgonyi/','Desktop','Desktop/list_imaging_test/dumpgrid_concentrated_ms','a.image','image.restored.test','stacked','concentrated')

    for i in range(0,11):
        show_difference_slice('/home/krozgonyi/Desktop/a.image','/home/krozgonyi/Desktop/dstacked_grids/image.test.restored',chan=i)

    #pass