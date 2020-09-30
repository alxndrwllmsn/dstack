"""Applications that can be directly used in imaging pipelines.
For example the stacking step in imaging is implemented both as
and application and a function in dstack. However, more complicated
imaging steps are implemented only as applications.
Parameters of the applications (e.g. input file names) are passed
as arguments.
"""

__all__ = ['dstacking']

import argparse
import logging

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def dstacking():
    """Stacks series of CASAImages using ``CIM_stacking_base()``

    

    """    
    print('Hello world, I am an application!')



if __name__ == "__main__":
    pass