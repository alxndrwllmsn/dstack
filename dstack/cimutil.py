"""
Collection of utility functions to interact with CASA images. 
Grids are currently dumped in casaimage format, hence the utilities works at both images and grids.
"""

__all__ = ['get_N_chan_from_CIM', 'get_N_pol_from_CIM']

import numpy as np

from casacore import images as casaimage


def get_N_chan_from_CIM(cimpath):
    """Get the number of channels from a CASAImage

    CASAImage indices: [freq, Stokes, x, y]

    Parameters
    ==========
    cimpath: str
        The input CASAImage parth
    
    Returns
    =======
    N_chan: int
        Number of channels in the CASAImage
    """
    cim = casaimage.image(cimpath)
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
        The input CASAImage parth
    
    Returns
    =======
    N_pol: int
        Number of polarisations in the CASAImage
    """
    cim = casaimage.image(cimpath)
    assert cim.ndim() == 4, 'The image has more than 4 axes!'

    N_pol = np.shape(cim.getdata())[1]

    return N_pol

if __name__ == "__main__":
    CIMPATH = '/home/krozgonyi/Desktop/first_pass/grid.wr.1.sim_PC'

    N_chan = get_N_chan_from_CIM(CIMPATH)
    N_pol = get_N_pol_from_CIM(CIMPATH)

    print(N_pol)