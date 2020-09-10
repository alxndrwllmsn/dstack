"""
Collection of the base functions operating on grids, which are stored in CASAImageformat. The main reason for a separate module,
is to logicaly separate the grid-specific functions. The ``cim`` module functions should be working for grids as well, however the grids
data array is complex that can complicate things. The grid utility functions can be found in the ``cmutil`` module. 
"""

__all__ = ['measure_grid_sparseness', 'grid_stacking_base']

import os
import shutil
import numpy as np
import warnings

from casacore import images as casaimage

import dstack as ds

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
    cgrid = ds.cimutil.create_CIM_object(cimgrid_path)

    assert cgrid.datatype() == 'Complex', 'Input CASAImage is not complex, and grids are axpected to be complex!'

    gird_size = cgrid.shape()[2] * cgrid.shape()[3]

    sparseness = (gird_size - np.count_nonzero(cgrid.getdata()[chan,pol,...])) / gird_size

    return sparseness


def grid_stacking_base(cgridpath_list,cgrid_output_path,cgrid_outputh_name,coordsys=None,normalise=False,overwrite=False):
    """This function is one of the core functions of grid stacking deep spectral line pipelines.

    This function takes a list of grids (CASAImage format) and create the stacked CASAIMage grid.
    The resultant grid can be a simple sum or a simple average.

    The given grids have to have the same:
        - shape
        - coordinates (except if the ``coordsys`` parameter defines the soordinarte system)

    The grids dumped from YandaSoft have none info assiciated. Hence, the stacked grid 
    have no associated attribute group to the image data! If the input grids 
    have associated attribute groups or history, a warning is raised.

    Furthermore, when creatiung a new image with ``python-casacore`` I can't define
    the unit of the pixels. So the created grid will have no associated units.

    Parameters
    ==========
    cgridpath_list: list
        A list of the full paths of the grids stacked
    
    cgrid_output_path: str
        The full path to the folder in which the stacked grid will be saved

    cgrid_output_name: str
        The name of the stacked grid
    
    coordsys: None or ``casacore.images.coordinates.coordinatesystem``
        The coordinate system used for the stacked grid. Use this functionality wisely,
        as this allows the (unphysical) stacking of grids with different
        coordinate system information!

        If not given, then the first grids coordinate system info 
        will be used from the ``cgridpath_list`` list.

    normalise: bool
        If True, the grids will be averaged instead of just summing them
    
    overwrite: bool
        If True, the stacked grid will be created regardless if another grid exist
        in the same name. Note, that in this case the existing grid will be deleted!

    Returns
    ========
    Stacked grid: CASAImage
        Create the stacked grid at ``cgrid_output_path/cgrid_output_name``
    """
    assert len(cgridpath_list) >= 2, 'Less than two grid is given for stacking!'

    output_cgrid = '{0:s}/{1:s}'.format(cgrid_output_path,cgrid_outputh_name)

    if os.path.isdir(output_cgrid): assert overwrite, 'Stacked grid already exist, and the overwrite parameters is set to False!'

    base_cgrid = ds.cimutil.create_CIM_object(cgridpath_list[0])

    assert base_cgrid.datatype() == 'Complex', 'Input grid {0:s} is not complex, and grids are axpected to be complex!'.format(cgrid.name())

    if coordsys == None:
        coordsys = base_cgrid.coordinates()
    else:
        assert type(coordsys) == type(base_cgrid.coordinates()), \
        'The given coordsys is not an casacore.images.coordinates.coordinatesystem object!'

    check_attrgroup_empty = lambda x: None if x.attrgroupnames() == [] else warnings.warn('Input grid {0:s} has a non-empty attribute list!'.format(x.name()))
    check_history_empty = lambda x: None if x.history() == [] else warnings.warn('Input grid {0:s} has a non-empty history field!'.format(x.name()))

    check_attrgroup_empty(base_cgrid)
    check_history_empty(base_cgrid)

    #If shape is given, the data type is automatically set to float!
    stacked_cgrid = casaimage.image(output_cgrid,
                    coordsys=coordsys,
                    values=base_cgrid.getdata(),
                    overwrite=overwrite)

    stacked_cgrid_data = base_cgrid.getdata()

    for i in range(1,len(cgridpath_list)):
        cgrid = ds.cimutil.create_CIM_object(cgridpath_list[i])

        assert cgrid.datatype() == 'Complex', 'Input grid {0:s} is not complex, and grids are axpected to be complex!'.format(cgrid.name())
        assert base_cgrid.ndim() == cgrid.ndim(), 'The dimension of the two input grids ({0:s} and {1:s}) are not equal!'.format(base_cgrid.name(),cgrid.name())

        #This is slow as it reads in the grids again!
        assert ds.cimutil.check_CIM_coordinate_equity(cgrid,stacked_cgrid), \
        'The created stacked grid and the grid {0:s} have different coordinate systems!'.format(cgrid.name())

        check_attrgroup_empty(base_cgrid)
        check_history_empty(base_cgrid)

        stacked_cgrid_data = np.add(stacked_cgrid_data, cgrid.getdata())

    if normalise:
        stacked_cgrid_data = np.divide(stacked_cgrid_data,len(cgridpath_list))

    stacked_cgrid.putdata(stacked_cgrid_data)

if __name__ == "__main__":
    grid_stacking_base(['/home/krozgonyi/Desktop/first_pass/grid.wr.1.sim_PC','/home/krozgonyi/Desktop/first_pass/grid.wr.1.sim_PC'],
                        '/home/krozgonyi/Desktop','a.grid',overwrite=True)