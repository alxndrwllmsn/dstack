"""Applications that can be directly used in imaging pipelines.
For example the stacking step in imaging is implemented both as
and application and a function in dstack. However, more complicated
imaging steps are implemented only as applications.
Parameters of the applications (e.g. input file names) are passed
as arguments.
"""

__all__ = ['dstacking', 'dparset']

import argparse
import logging

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def dstacking():
    """Stacks series of CASAImages using ``CIM_stacking_base()``

    Keyword Arguments
    =================
    list -cl or --cimpath_list:
        Full path of images to be stacked. Note, that a list of lists is created
        when the -cl argument is called (can happen multiple times), but the code handles it.

    str -cp or --cim_output_path:
        Full path to the folder in which the stacked image will be saved.

    str -cn or --cim_outputh_name:
        Name of the stacked image.
    
    optional -n or --normalise:
        Boolean argument, if not given set to False otherwise True.
        If True, the images will be averaged instead of just summing them.

    optional -o or --overwrite:
        Boolean argument, if not given set to False otherwise True.
        If True, the stacked image will be created regardless if another image exist by the same name.

    optional -c or --close:
        Boolean argument, if not given set to False otherwise True.
        If True the in-memory CASAIMages given by ``cimpath_list`` are deleted, and the optional write-lock releases.

    """
    parser = argparse.ArgumentParser(description='This is an application to stack CASAImages.')

    #=== Required arguments ===
    parser.add_argument('-cl', '--cimpath_list', 
                        help='A list of the full paths of the images to be stacked.', 
                        required=True, action="append", nargs='+', type=str)

    parser.add_argument('-cp', '--cim_output_path', 
                        help='Full path to the folder in which the stacked image will be saved.', 
                        required=True, action="store", type=str)

    parser.add_argument('-cn', '--cim_outputh_name', 
                        help='Name of the stacked image', 
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    #The boolean type arguments are set to True if given, otherwise they set to False
    parser.add_argument('-n', '--normalise', 
                        help='If True, the images will be averaged instead of just summing them.', 
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

def dparset():
    """Creates a ``YandaSoft`` parset file from template and other parameters.

    A flexible application around the :obj:`Parset` class, which instantly creates
    parset files.

    Keyword Arguments
    =================
    



    """
    parser = argparse.ArgumentParser(description='This is an application to create parset files for YandaSoft.')

    #=== Required arguments ===
    parser.add_argument('-i', '--imager', 
                        help='The Imager used in the parset. Have to be supported by dstack.',
                        required=True, action="store", type=str)

    parser.add_argument('-n', '--image_names', 
                        help='The Images.Names parameters used to create the parset file.',
                        required=True, action="store", type=str)

    parser.add_argument('-g', '--gridder_name', 
                        help='The gridder parameter used to create the parset file.',
                        required=True, action="store", type=str)

    parser.add_argument('-op', '--output_path', 
                        help='Full path to the folder in which the parset will be saved.',
                        required=True, action="store", type=str)

    parser.add_argument('-pn', '--parset_name', 
                        help='Name of the parset file created.',
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    parser.add_argument('-t', '--template_path', 
                        help='Full path to a template parset file, which can be used to initialize the parset parameters.', 
                        required=False, action="store", type=str)

    parser.add_argument('-tn', '--template_image_names', 
                        help='The Images.Names parameters used in the parset template. If not given, the ``--imager_names`` argument value used instead.',
                        required=False, action="store", type=str)

    parser.add_argument('-tg', '--teamplate_gridder_name', 
                        help='The gridder parameter used in the parset template. If not given, the ``--gridder_name`` argument value used instead.',
                        required=False, action="store", type=str)

    parser.add_argument('-p', '--preconditioner', 
                        help='A list of the preconditioners used to create the parset. If not given, the preconditioners read from the template if given, otherwise an empty list.', 
                        required=False, action="append", nargs='+', type=str)    


    #=== Application MAIN ===
    args = parser.parse_args()

    #Initialise parset, check template and if given set the gridder_name and image_names parameters
    if args.template_path != None:
        if args.template_image_names == None:
            log.debug('Parset template provided but no specific template_image_names is specified.')

            if args.teamplate_gridder_name == None:
                log.debug('Parset template provided but no specific teamplate_gridder_name is specified.')
                
                print(args.image_names)
                parset = ds.parset.Parset(imager=args.imager, image_names=args.image_names,
                                        gridder_name=args.gridder_name, template_path=args.template_path)

            else:
                parset = ds.parset.Parset(imager=args.imager, image_names=args.image_names,
                                        gridder_name=args.teamplate_gridder_name, template_path=args.template_path)

                parset.update_gridder_name(gridder_name=args.gridder_name)

        else:
            if args.teamplate_gridder_name == None:
                log.debug('Parset template provided but no specific teamplate_gridder_name is specified.')
    
                parset = ds.parset.Parset(imager=args.imager, image_names=args.template_image_names,
                                        gridder_name=args.gridder_name, template_path=args.template_path)

                parset.update_image_names(image_names=args.image_names)

            else:
               parset = ds.parset.Parset(imager=args.imager, image_names=args.template_image_names,
                                        gridder_name=args.teamplate_gridder_name, template_path=args.template_path)

               parset.update_parset_mapping(image_names=args.image_names,gridder_name=args.gridder_name)

    else:
        log.debug('No parset template is given!')
        parset = ds.parset.Parset(imager=args.imager, image_names=args.image_names,gridder_name=args.gridder_name)

    #Preconditioning
    if args.preconditioner != None:
        #Flatten out the preconditioner argument, which currently a list of lists 
        args.preconditioner = [p for sublist in args.preconditioner for p in sublist]

        #Update preconditioner
        #Need to create this method

    else:
        log.debug('Preconditioner set based on the template parset.')


    print(parset)

    parset.save_parset(output_path=args.output_path, parset_name=args.parset_name)

if __name__ == "__main__":
    pass