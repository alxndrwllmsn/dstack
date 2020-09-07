"""
Collection of the base functions operating on CASAImages that can be both grids and images.
"""

__all__ = ['check_CIM_equity', 'measure_grid_sparseness', 'measure_CIM_RMS', 'create_CIM_diff_array']

import numpy as np

from casacore import images as casaimage

def check_CIM_equity(cimpath_a,cimpath_b,numprec=1e-8):
    """Check if two CASAImages are identical or not up to a defined numerical precision
    This function is used to test certain piepline features.

    Note: NaN-s are treted as equals, and the :numprec: parameter only sets the relative difference limit.

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

def measure_grid_sparseness(cimgrid_path,chan=0,pol=0):
    """Measure the sparseness of the input grid, given in a complex CASAImage format

    The sparseness is defined as:

    .. math:: S = G_0 / G
    
    where :math:`G` representing all the grid cells and :math:`G_0` he empty grid cells.
    However due to optimisation the code actually computes:

    .. math:: S = (G - G_N) / G

    where, :math:`G_N` is the number of non-zero grid cells.

    The sparseness is counted over the complex entries, thus only grid cells with
    both zero real and imaginary part are counted as empty cells.

    Parameters
    ==========
    cimgrid_path: str
        The input grid parth (CASAImage format)

    chan: int
        Index of the channel in the grid cube

    pol: int
        Index of the polarisation in the grid cube

    Returns
    =======
    sparseness: float
        Sparseness of the grid
    """
    cimgrid = casaimage.image(cimgrid_path)

    assert cimgrid.datatype() == 'Complex', 'Input CASAImage is not complex, and grids are axpected to be complex!'

    gird_size = cimgrid.shape()[2] * cimgrid.shape()[3]

    sparseness = (gird_size - np.count_nonzero(cimgrid.getdata()[chan,pol,...])) / gird_size

    return sparseness

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
        Note taht it can be *very slow and memory heavy*!

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
        Note that, this can be *very slow*!

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
    if all_dim:
        cimgrid = casaimage.image(cimgrid_path)
        rms_matrix = np.zeros((cimgrid.shape()[0],cimgrid.shape()[1]))

        for chan_i in range(0,cimgrid.shape()[0]):
            for pol_j in range(0,cimgrid.shape()[1]):
                rms_matrix[i,j] = np.sqrt(np.mean(np.square(cimgrid.getdata()[chan_i,pol_j,...])))

        return rms_matrix

    else:
        return np.sqrt(np.mean(np.square(cimgrid.getdata()[chan,pol,...])))

if __name__ == "__main__":
    #CIMGRID_PATH = '/home/krozgonyi/Desktop/ASKAP_high_res_grid_example/grid.wr.1.sim_PC'
    CIMGRID_PATH = '/home/krozgonyi/Desktop/ASKAP_high_res_grid_example/pcf.wr.1.sim_PC'

    S = measure_grid_sparseness(CIMGRID_PATH)

    print(S)

    exit()

    CIMPATH_A = '/home/krozgonyi/Desktop/first_pass/grid.wr.1.sim_PC'
    CIMPATH_B = '/home/krozgonyi/Desktop/first_pass/psfgrid.wr.1.sim_PC'


    cimA = casaimage.image(CIMPATH_A)
    cimB = casaimage.image(CIMPATH_B)  


    print(np.shape(np.unique(cimA.getdata())))
    print(np.shape(np.unique(cimB.getdata())))



    print(check_CIM_equity(CIMPATH_A,CIMPATH_B,numprec=1e2))