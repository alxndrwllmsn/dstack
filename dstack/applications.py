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
    parser = argparse.ArgumentParser(description='This is program to stack CASAImages')

    #=== Required arguments ===
    #The -cl option creates a list every time it is called, thus the args.cimpath_list is a list containing lists
    #Therefore, I need to create a single list from the potential sub-lists
    parser.add_argument('-cl', '--cimpath_list', 
                        help='A list of the full paths of the images to be stacked', 
                        required=True, action="append", nargs='+', type=str)

    parser.add_argument('-cp', '--cim_output_path', 
                        help='Full path to the folder in which the stacked image will be saved', 
                        required=True, action="store", type=str)

    parser.add_argument('-cn', '--cim_outputh_name', 
                        help='Name of the stacked image', 
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    #The boolean type arguments are set to True if given, otherwise they set to False
    parser.add_argument('-n', '--normalise', 
                        help='If True, the images will be averaged instead of just summing them', 
                        required=False, action="store_true")

    parser.add_argument('-o', '--overwrite', 
                        help='If True, the stacked image will be created regardless if another image exist by the same name.', 
                        required=False, action="store_true")

    parser.add_argument('-c', '--close', 
                        help='If True the in-memory CASAIMages given by ``cimpath_list`` are deleted, and the optional write-lock releases.', 
                        required=False, action="store_true")

    #=== Application MAIN ===
    args = parser.parse_args()
    #Flatten out the cimpat_list argument, which currently a list of lists 
    args.cimpath_list = [cimpath for sublist in args.cimpath_list for cimpath in sublist]

    ds.cim.CIM_stacking_base(cimpath_list= args.cimpath_list,
                            cim_output_path = args.cim_output_path,
                            cim_outputh_name = args.cim_outputh_name,
                            normalise = args.normalise,
                            overwrite = args.overwrite,
                            close = args.close)

if __name__ == "__main__":
    pass