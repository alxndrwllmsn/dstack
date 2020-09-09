"""
Collection of the base functions operating on images in CASAImageformat. The functions defined here *should* work on grids as well,
but for several reasons specific functions for grids are implemented in a separate module called ``cgrid``.
"""

__all__ = ['check_CIM_equity', 'measure_CIM_RMS', 'create_CIM_diff_array']

import numpy as np

from casacore import images as casaimage

def check_CIM_equity(cimpath_a,cimpath_b,numprec=1e-8):
    """Check if two CASAImages are identical or not up to a defined numerical precision
    This function is used to test certain piepline features.

    Note: NaN-s are trethed as equals, and the :numprec: parameter only sets the relative difference limit.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage parth of Alice

    cimpath_b: str
        The input CASAImage parth of Bob

    numprec: float
        The numerical precision limit of the maximum allowed relative
        difference between CASAImages Alice and Bob.
        If set to zero, equity is checked.

    Returns
    =======
    equity: bool
        True or False, base on the equity of Alice and Bob
    """

    cimA = casaimage.image(cimpath_a)
    cimB = casaimage.image(cimpath_b)

    assert cimA.ndim() == cimB.ndim(), 'The dimension of the two input CASAImage is not equal!'

    if numprec == 0.:
        return np.array_equiv(cimA.getdata(),cimB.getdata())
    else:
        return np.allclose(cimA.getdata(),cimB.getdata(),atol=0,rtol=numprec,equal_nan=True)

def create_CIM_diff_array(cimpath_a,cimpath_b,rel_diff=False,all_dim=False,chan=0,pol=0):
    """Compute the difference of two CASAImage, and return it as a numpy array.
    Either the entire difference cube, or only the difference of a selected channel 
    and polarisation slice is returned.

    The code computes the first minus second image given, and normalises with the
    second one if the rel_diff parameter is set to True.

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
    cimA = casaimage.image(cimpath_a)
    cimB = casaimage.image(cimpath_b)

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
    cim = casaimage.image(cimpath)

    if all_dim:
        rms_matrix = np.zeros((cim.shape()[0],cim.shape()[1]))

        for chan_i in range(0,cim.shape()[0]):
            for pol_j in range(0,cim.shape()[1]):
                rms_matrix[i,j] = np.sqrt(np.mean(np.square(cim.getdata()[chan_i,pol_j,...])))

        return rms_matrix

    else:
        return np.sqrt(np.mean(np.square(cim.getdata()[chan,pol,...])))


if __name__ == "__main__":
    pass