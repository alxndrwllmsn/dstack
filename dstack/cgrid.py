"""
Collection of specific functions operating on grids, which are stored in CASAImageformat. The main reason for a separate module,
is to logically separate the grid-specific functions. The ``cim`` module functions are working on grids as well, however the grids
data array is complex that can complicate things...
"""

__all__ = ['measure_grid_sparseness']

import os
import shutil
import numpy as np
import logging

from casacore import images as casaimage

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def measure_grid_sparseness(cimgrid_path, chan=0,pol=0):
    """Measure the sparseness of the input grid, given in a complex CASAImage format

    The sparseness is defined as:

    .. math:: S = G_0 / G
    
    where :math:`G` representing all the grid cells and :math:`G_0` he empty grid cells.
    However due to optimization the code actually computes:

    .. math:: S = (G - G_N) / G

    where, :math:`G_N` is the number of non-zero grid cells.

    The sparseness is counted over the complex entries, thus only grid cells with
    both zero real and imaginary part are counted as empty cells.

    Parameters
    ==========
    cimgrid_path: str
        The input grid path (CASAImage format)

    chan: int, optional
        Index of the channel in the grid cube

    pol: int, optional
        Index of the polarization in the grid cube

    Returns
    =======
    sparseness: float
        Sparseness of the grid
    """
    cgrid = ds.cim.create_CIM_object(cimgrid_path)
    
    if cgrid.datatype() != 'Complex':
        raise TypeError('Input CASAImage is not complex, and grids are axpected to be complex!')

    gird_size = cgrid.shape()[2] * cgrid.shape()[3]

    sparseness = (gird_size - np.count_nonzero(cgrid.getdata()[chan,pol,...])) / gird_size

    return sparseness

if __name__ == "__main__":
    pass
