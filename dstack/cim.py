"""
Collection of the base functions operating on CASAImages that can be both grids and images.
"""

__all__ = ['check_CIM_equity']

import numpy as np

from casacore import images as casaimage

def check_CIM_equity(cimpath_a,cimpath_b,numprec=1e-8):
    """Check if two CASAImages are identical or not up to a defined numerical precision
    This function is used to test certain piepline features.

    Note: NaN-s are treted as equals, and the :numprec: parameter only sets the absolute difference limit.
    The relative difference is not used.

    Parameters
    ==========
    cimpath_a: str
        The input CASAImage parth of Alice

    cimpath_b: str
        The input CASAImage parth of Bob

    numprec: float
        The numerical precision limit of the maximum allowed absolute
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
        return np.allclose(cimA.getdata(),cimB.getdata(),atol=numprec,rtol=0,equal_nan=True)


if __name__ == "__main__":
    CIMPATH_A = '/home/krozgonyi/Desktop/first_pass/grid.wr.1.sim_PC'
    CIMPATH_B = '/home/krozgonyi/Desktop/first_pass/psfgrid.wr.1.sim_PC'


    cimA = casaimage.image(CIMPATH_A)
    cimB = casaimage.image(CIMPATH_B)  


    print(np.shape(np.unique(cimA.getdata())))
    print(np.shape(np.unique(cimB.getdata())))



    print(check_CIM_equity(CIMPATH_A,CIMPATH_B,numprec=1e2))