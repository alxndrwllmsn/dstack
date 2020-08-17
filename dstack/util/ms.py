"""
Collection of utility functions to interact with Measurement Sets
"""

__all__ = ['get_MS_phasecentre']

import numpy as np

from casacore import tables as casatables

from astropy.coordinates import SkyCoord
from astropy import units as u

def get_MS_phasecentre(mspath, frame='icrs', ack=False):
    """Get the list of the phase centres for each field and direction of the MS
    and return a list of astropy skycoord values

    Both field and direcrion IDs are expected to increment from zero, and the maximum
    ID can be the number of unique fields/dds. However, less than the maxumum number of
    valid IDs can occure and this code can handle that.

    e.g. one field and one direction ID, but in the PHASE_DIR table, 
    phase centre for two directions are existing, the code chooses the valid one

    Parameters
    ==========

    mspath: str
        The input MS path

    frame: str, optional
        Reference frame used to calculate the Astropy phase centre. Default: 'icrs'

    ack: bool, optional
        Enabling messages of successful interaction with the MS
        e.g. successful opening of a table
    
    Returns
    =======
    phasecentres: list of lists containing Astropy skycoords
        A list of the phasecentres for each field and direction in the MS as a list of lists
        i.e. each element is a list

    """
    MS = casatables.table(mspath, ack=ack)

    #Get the number of unique fields and data descriptoions (e.g. footprints)
    fields = np.unique(MS.getcol('FIELD_ID'))
    dds = np.unique(MS.getcol('DATA_DESC_ID'))

    fields_table = casatables.table(mspath + '/FIELD', ack=ack)    

    phasecentres = []

    #Get the reference equinox from the table keywords
    equinox = fields_table.getcolkeyword('PHASE_DIR','MEASINFO')['Ref'] 

    #Only can convert from radians
    assert fields_table.getcolkeyword('PHASE_DIR','QuantumUnits')[0] == 'rad', 'Phase centre direction is not in radians!'

    i = 0
    j = 0
    for field in range(0,np.size(fields)):
        #The number and referencing of fields can be messy
        if np.shape(fields_table.getcol('PHASE_DIR'))[1] > np.size(fields):
            field_ID = fields[i]
        else:
            field_ID = field

        directions = []

        for dd in range(0,np.size(dds)):
            #Same for the DDs as the fields
            if np.shape(fields_table.getcol('PHASE_DIR'))[0] > np.size(dds):
                dd_ID = dds[i]
            else:
                dd_ID = dd

            pc = fields_table.getcol('PHASE_DIR')[dd_ID,field_ID, :]

            #Convert to astropy coordinates
            directions.append(SkyCoord(ra=pc[0] * u.rad, dec=pc[1] * u.rad, frame=frame, equinox=equinox))

            j += 1
    
    phasecentres.append(directions)

    i += 1

    return phasecentres
