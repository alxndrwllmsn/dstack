"""Applications that can be directly used in imaging pipelines.
For example the stacking step in imaging is implemented both as
and application and a function in dstack. However, more complicated
imaging steps are implemented only as applications.
Parameters of the applications (e.g. input file names) are passed
as arguments.
"""

__all__ = ['dstacking', 'dparset', 'cim2fits', 'sdplots', 'cimRMS', 'initTSF',
            'dsSoFiAtparset']

import sys, os
import argparse
import logging

import numpy as np

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def argflatten(arg_list):
    """Some list arguments is actually a list of lists.
    A simple routine to faltten list of lists to a simple list

    Parameters
    ==========
    arg_list: list of lists
        Value of a list argument needs to be flatten

    Return
    ======
    arg_as_list: list
        The flattened list of lists

    """
    return [p for sublist in arg_list for p in sublist]


def dstacking():
    """Stacks series of CASAImages using ``CIM_stacking_base()``

    Keyword Arguments
    =================
    list -cl or --cimpath_list:
        List of the full path of images to be stacked. Note, that a list of lists is created
        when the -cl argument is called (can happen multiple times), but the code handles it.

    str -cp or --cim_output_path:
        String, full path to the folder in which the stacked image will be saved.

    str -cn or --cim_outputh_name:
        String, name of the stacked image.
    
    optional -n or --normalise:
        Boolean argument, if not given set to False otherwise True.
        If True, the images will be averaged instead of just summing them.

    optional -psf or --weight_with_psf:
        Boolean argument, if True, the images will be weighted wit a list of PSF provided.

    optional -pl or --psfpath_list:
        List of the full paths of the psf used for weighting.

    optional -l or --psf_peaks_log_path:
        string, full path and name of a lofgile which the PSF peask will be saved.

    optional -o or --overwrite:
        Boolean argument, if not given set to False otherwise True.
        If True, the stacked image will be created regardless if another image exist by the same name.

    optional -c or --close:
        Boolean argument, if not given set to False otherwise True.
        If True the in-memory CASAIMages given by ``cimpath_list`` are deleted, and the optional write-lock releases.

    Return
    ======
    Stacked CASAImage: file
        Creates the stacked CASAImage.
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

    parser.add_argument('-psf', '--weight_with_psf', 
                        help='If True, the images will be weighted wit a list of PSF provided.', 
                        required=False, action="store_true")

    parser.add_argument('-pl', '--psfpath_list', 
                        help='A list of the full paths of the psf used for weighting.', 
                        required=False, action="append", nargs='+', type=str)

    parser.add_argument('-l', '--psf_peaks_log_path', 
                        help='Full path and name of a lofgile which the PSF peask will be saved.', 
                        required=False, action="store", type=str)

    parser.add_argument('-o', '--overwrite', 
                        help='If True, the stacked image will be created regardless if another image exist by the same name.', 
                        required=False, action="store_true")

    parser.add_argument('-c', '--close', 
                        help='If True the in-memory CASAIMages given by ``cimpath_list`` are deleted, and the optional write-lock releases.', 
                        required=False, action="store_true")

    #=== Application MAIN ===
    args = parser.parse_args()
    #Flatten out the cimpat_list argument, which currently a list of lists 
    args.cimpath_list = argflatten(args.cimpath_list)

    if args.weight_with_psf == True:
        args.psfpath_list = argflatten(args.psfpath_list)

        ds.cim.CIM_stacking_base(cimpath_list = args.cimpath_list,
                                cim_output_path = args.cim_output_path,
                                cim_outputh_name = args.cim_outputh_name,
                                weight_with_psf = args.weight_with_psf,
                                psfpath_list = args.psfpath_list,
                                psf_peaks_log_path = args.psf_peaks_log_path,
                                normalise = args.normalise,
                                overwrite = args.overwrite,
                                close = args.close)

    else:
        ds.cim.CIM_stacking_base(cimpath_list = args.cimpath_list,
                                cim_output_path = args.cim_output_path,
                                cim_outputh_name = args.cim_outputh_name,
                                normalise = args.normalise,
                                overwrite = args.overwrite,
                                close = args.close)

def dparset():
    """Creates a ``YandaSoft`` parset file from template and other parameters.

    A flexible application around the :obj:`Parset` class, which instantly creates
    and saves a parset file. Using this application allows the user to read in
    a template parset and save a parset with additional parameters, using a different
    mapping...etc.

    Keyword Arguments
    =================
    str -i or --imager:
        The Imager used in the parset. Has to be supported by dstack.
    
    str -n or --image_names:
        The Images.Names parameters used to create the parset file. Defines the parameter for the output file, 
        thus if a template is used and the input is different specify that by the argument ``--template_image_names``.

    str -g or --gridder_name:
        The gridder parameter used to create the parset file. Defines the parameter for the output file, 
        thus if a template is used and the input is different specify that by the argument ``--teamplate_gridder_name``.

    str -op or --output_path:
        Full path to the folder in which the parset will be saved.
         
    str -pn or --parset_name:
        Name of the parset file created.

    optional -l or --log:
        Boolean. If True the logger level set to INFO. Set to False by default

    optional -t or --template_path:
        String. Full path to a template parset file, which can be used to initialize the parset parameters.

    optional -tn or --template_image_names:
        String. The Images.Names parameters used in the parset template. If not given, the ``--image_names`` argument value used instead.

    optional -tg or --teamplate_gridder_name:
        The gridder parameter used in the parset template. If not given, the ``--gridder_name`` argument value used instead.

    optional -p or --preconditioner:
        String(s). A list of the preconditioners used to create the parset. If not given, 
        the preconditioners read from the template if given. A simple -p with no arguments 
        results in an empty list i.e. no preconditioners defined.
    
    optional -a or --append_preconditioner_settings:
        List of dstack parameter keys and values in the format of  "key=value". The parset parameters are appended with the list elements.

    optional -d or --delete_preconditioner_settings:
        List of dstack parameter keys to remove from the parset.

    optional: -u or --use_image_names:
        Boolean. If True the Images.image_names.param parametes set to be the default parameters for the ambigous imaging parameters.
        Set to False by default, i.e. the default parameters are the Images.parameters for the ambigous parameters.

    Return
    ======
    parset: file
        Creates the parset file defined by the arguments.
    """
    parser = argparse.ArgumentParser(description='This is an application to create parset files for YandaSoft.')

    #=== Required arguments ===
    parser.add_argument('-i', '--imager', 
                        help='The Imager used in the parset. Has to be supported by dstack.',
                        required=True, action="store", type=str)

    parser.add_argument('-n', '--image_names', 
                        help='The Images.Names parameters used to create the parset file. Defines the parameter for the output file,\
                             thus if a template is used and the input is different specify that by the argument --template_image_names.',
                        required=True, action="store", type=str)

    parser.add_argument('-g', '--gridder_name', 
                        help='The gridder parameter used to create the parset file.Defines the parameter for the output file,\
                             thus if a template is used and the input is different specify that by the argument --teamplate_gridder_name.',
                        required=True, action="store", type=str)

    parser.add_argument('-op', '--output_path', 
                        help='Full path to the folder in which the parset will be saved.',
                        required=True, action="store", type=str)

    parser.add_argument('-pn', '--parset_name', 
                        help='Name of the parset file created.',
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    parser.add_argument('-l', '--log', 
                        help='If True the logger level set to INFO. Set to False by default', 
                        required=False, action="store_true")

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
                        help='A list of the preconditioners used to create the parset. If not given, the preconditioners read from the template if given. \
                            A simple -p with no arguments results in an empty list i.e. no preconditioners defined.', 
                        required=False, action="append", nargs='*', type=str)

    parser.add_argument('-a', '--append_preconditioner_settings',
                        help='List of dstack parameter keys and values in the format of  "key=value". The parset parameters are appended with the list elements.',
                        required=False, action="append", nargs='+', type=str)

    parser.add_argument('-d', '--delete_preconditioner_settings',
                        help='List of dstack parameter keys to remove from the parset.',
                        required=False, action="append", nargs='+', type=str)

    parser.add_argument('-u', '--use_image_names', 
                        help='If True the Images.image_names.param parametes set to be the default parameters for the ambigous imaging parameters.\
                            Set to False by default, i.e. the default parameters are the Images.parameters for the ambigous parameters.', 
                        required=False, action="store_true")

    #=== Application MAIN ===
    args = parser.parse_args()

    #Set up logging if needed
    if args.log:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #Initialise parset, check template and if given set the gridder_name and image_names parameters
    if args.template_path != None:
        if args.template_image_names == None:
            log.debug('Parset template provided but no specific template_image_names is specified.')

            if args.teamplate_gridder_name == None:
                log.debug('Parset template provided but no specific teamplate_gridder_name is specified.')
                
                parset = ds.parset.Parset(imager=args.imager, image_names=args.image_names,
                                        gridder_name=args.gridder_name, template_path=args.template_path)

            else:
                parset = ds.parset.Parset(imager=args.imager, image_names=args.image_names,
                                        gridder_name=args.teamplate_gridder_name, template_path=args.template_path)

                parset.update_gridder(gridder_name=args.gridder_name)
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
        args.preconditioner = argflatten(args.preconditioner)
        parset.update_preconditioner(args.preconditioner)

    else:
        log.debug('Preconditioner is set based on the template parset.')

    #Append parset settings
    if args.append_preconditioner_settings != None:
        args.append_preconditioner_settings = argflatten(args.append_preconditioner_settings)
        for new_prec_param in args.append_preconditioner_settings:
            param_key = new_prec_param.split("=")[0]
            param_val = new_prec_param.split("=")[1]
            parset.add_parset_parameter(param_key,param_val)

    #Delete parset settings
    if args.delete_preconditioner_settings != None:
        args.delete_preconditioner_settings = argflatten(args.delete_preconditioner_settings)
        for prec_param_to_remove in args.delete_preconditioner_settings:
            parset.remove_parset_parameter(prec_param_to_remove)

    parset.save_parset(output_path=args.output_path, parset_name=args.parset_name,use_image_names=args.use_image_names)

def cim2fits():
    """Simple command line tool to convert CASAImages to .fits format. It is a wrapper arounf the python-casacore.images.image.tofits() function

    The DINGO pipeline needs to converts the final deep images to .fits format if the output is not .fits by default as SofiA 2.0 requires .fits input.

    The application runs in the working directory by default, but absolute pats can be defined as well.

    Keyword Arguments
    =================
    str -i or --input:
        Input CASAImage either fileneam only or full path.
    
    str -o or --output:
        Output .fits file name or full path.

    Return
    ======
    fits: file
        The input image in a .fits format.
    """
    parser = argparse.ArgumentParser(description='This is an application to convert CASAImages to .fits format')

    #=== Required arguments ===
    parser.add_argument('-i', '--input', 
                        help='Input CASAImage either fileneam only or full path.',
                        required=True, action="store", type=str)

    parser.add_argument('-o', '--output', 
                        help='Output .fits file name or full path.',
                        required=True, action="store", type=str)

    #=== Application MAIN ===
    args = parser.parse_args()

    input_cim = ds.cim.create_CIM_object(args.input)

    input_cim.tofits(args.output)

    del input_cim

def sdplots():
    """Application that creates complementary diagnostics plots for the output of a SoFiA source finder.

    The diagnostics plots are discussed in detail in the `sdiagnostic` module of `dstack`.
    
    This application calls the `ds.sdiagnostic.create_complementary_figures_to_sofia_output()` function.
    
    Keyword Arguments
    =================
    str, -s or --sofia_dir_path:
        Full path to the directory where the output of SoFiA saved/generated. Has to end with a slash (/)!

    str -n or --name_base:
      The `output.filename` variable defined in the SoFiA template .par. Basically the base of all file names.
      However, it has to end with a lower dash (?): _ !
   
    optional -m or --masking:
        Boolean. If True (default), pixel values below a certain sensitivity threshold will be masked out.

    optional -ms or --mask_sigma:
        Float. The masking threshold value. The masking is performed based on the moment0 map.
        The threshold is given in terms of column density sensitivity values (similarly to contour lines)

    optional -c or --contour_levels:
        List. List of contour levels to be drawn. The levels are defined in terms of column-density sensitivity

    optional -N or --N_optical_pixels:
        Int. Number of pixels of the background image. Image size in arcseconds if the background is `DSS2 Red` (default)
     
    optional -j or --b_maj:
        Float. Angular major axis of the beam [arcsec]
    
    optional -i or --b_min:
        Float. Angular minor axis of the beam [arcsec]

    optional -a or --b_pa:
        Float. Angle of the beam [deg]

    optional -f or --v_frame:
        Str. The velocity frame. Can be 'frequency', 'optical' or 'radio'
    
    optional -bc or --beam_correction:
        Boolean. If True, the flux values are corrected for the synthesised beam

    optional -jp or --b_maj_px:
        Float. The major axis of the beam in pixels

    optional -ip or --b_min_px:
        Float. The minor axis of the beam in pixels

    optional -uc or --use_config:
        Bool. If True (False by default) the beam parameters are read from a config file

    optional -cp or --config_path:
        The path to the config file containing the beam parameters

    Return
    ======
    output_images: multiple files
        Create summary plots and in separate folders sub images of mom0 contours + optical background,
        mom0, mom1, mom2 maps and spectra for each SoFiA source.
 
    """
    parser = argparse.ArgumentParser(description='This is an application to create complementary imaes of sources found by the SoFiA source finder.')

    #=== Required arguments ===
    parser.add_argument('-s', '--sofia_dir_path', 
                        help='Full path to the SoFiA output directory. Has to end with a / !',
                        required=True, action="store", type=str)

    parser.add_argument('-n', '--name_base',
                        help='Name base of the output files. The same as the output.filename vaiable with an underscore at the end, as it has to end with _ !',
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    parser.add_argument('-m', '--masking',
                    help='If True (default), masking will be applied to the moment maps.',
                    required=False, action="store_false")

    parser.add_argument('-ms', '--masking_sigma',
                    help='The threshold in terms of column density sensitivity sigmas, which below pixels are masked',
                    required=False, nargs='?', default=3., type=float)

    parser.add_argument('-c', '--contour_levels',
                    help='List of contour levels to be drawn. The levels are defined in terms of column-density sensitivity',
                    required=False, action="append", nargs='*', type=float)

    parser.add_argument('-N', '--N_optical_pixels',
                    help='Number of pixels of the background image. Image size in arcseconds if the background is `DSS2 Red` (default)',
                    required=False, default=600, nargs='?', type=int)

    parser.add_argument('-j', '--b_maj',
                    help='Angular major axis of the beam [arcsec]',
                    required=False, default=30., nargs='?', type=float)

    parser.add_argument('-i', '--b_min',
                    help='Angular minor axis of the beam [arcsec]',
                    required=False, default=30., nargs='?', type=float)

    parser.add_argument('-a', '--b_pa',
                    help='Angle of the beam [deg]',
                    required=False, default=0., nargs='?', type=float)

    parser.add_argument('-f', '--v_frame',
                    help="The velocity frame. Can be 'frequency', 'optical' or 'radio'",
                    required=False, nargs='?', default='optical', type=str)

    parser.add_argument('-bc', '--beam_correction',
                    help='If True, the flux values are corrected for the synthesised beam',
                    required=False, action="store_true")

    parser.add_argument('-jp', '--b_maj_px',
                    help='The major axis of the beam in pixels',
                    required=False, default=5., nargs='?', type=float)

    parser.add_argument('-ip', '--b_min_px',
                    help='The minor axis of the beam in pixels',
                    required=False, default=5., nargs='?', type=float)

    parser.add_argument('-sv', '--survey',
                    help='The optical survey used for the background image',
                    required=False, default=None, nargs='?', type=str)

    parser.add_argument('-uc', '--use_config', 
                        help='If yes, a config file is used for the beam parameters',
                        required=False, action="store_true")

    parser.add_argument('-cp', '--config_path', 
                        help='Full path and name of the input config file',
                        required=False, default=None, type=str)

    #=== Application MAIN ===
    args = parser.parse_args()

    #Set up logging if needed
    #if args.log:
    #    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.contour_levels == None:
        args.contour_levels = [3.0, 5.0, 7.0, 9.0, 11.0]
    else:
        args.contour_levels = argflatten(args.contour_levels)

    if not args.use_config:
        ds.sdiagnostics.create_complementary_figures_to_sofia_output(
                sofia_dir_path = args.sofia_dir_path,
                name_base = args.name_base,
                masking = args.masking,
                contour_levels = args.contour_levels,
                N_optical_pixels = args.N_optical_pixels,
                b_maj = args.b_maj,
                b_min = args.b_min,
                b_pa = args.b_pa,
                v_frame = args.v_frame,
                beam_correction = args.beam_correction,
                b_maj_px = args.b_maj_px,
                b_min_px = args.b_min_px,
                survey = args.survey)
    else:
        #Read beam parameters from config file
        fitspath, dsID, b_maj, b_min, b_pa, d_px, b_maj_px, b_min_px, source_ID, RA, Dec, freq, region_string = \
        ds.pipelineutil.read_TSF_dsconfig(args.config_path)

        ds.sdiagnostics.create_complementary_figures_to_sofia_output(
                sofia_dir_path = args.sofia_dir_path,
                name_base = args.name_base,
                masking = args.masking,
                contour_levels = args.contour_levels,
                N_optical_pixels = args.N_optical_pixels,
                b_maj = b_maj,
                b_min = b_min,
                b_pa = b_pa,
                v_frame = args.v_frame,
                beam_correction = args.beam_correction,
                b_maj_px = b_maj_px,
                b_min_px = b_min_px,
                survey = args.survey)

def cimRMS():
    """Simple application to measure the RMS noise of an image cube along the
    spectral axis. The code cannot deal with polarised images.
    
    It only works on CASA images, but not on .fits files!

    It is a wrapper around `dstcak.cim.measure_CIM_RMS`

    The output is a .dat file with two columns containing the spectral values
    and the respective measured RMS values.

    Keyword Arguments
    =================
    str, -c or --cim_path:
        Full path to the CASAImage file

    str -o or --output:
        Output .dat file name or full path.

    optional -ad or --all_dim:
        Boolean. If True (defalut), the RMS will be computed for all channels if false,
        only the channels defined will be used

    optional -cmin or --chan_min:
        Int. Index of the first spectral channel

    optional -cmax or --chan_max:
        Int. Index of the last spectral channel

    optional -r or --robust:
        Boolean. If True (default), the RMS will be computed using a robust method

    optional -pc or --percentile_cut
        The (upper) percentile value which below/above the data is being ignored
        when computing the robust RMS

    optional -w or --window:
        Boolean. If true (default) the given window will be used for the RMS calculation

    optional -wh or --window_halfsize
        Int. The halfsize of the rectengular window used. The size is 1+2*wh

    optional -wc or --window_centre
        List of ints. The [x,y] coordinates of the window centre. If not given, the centre
        coordinates of the image are used.

    Return
    ======
    output_data_fuiles: multiple files
        Create the dat files with the channel and RMS columns
    """
    parser = argparse.ArgumentParser(description='This is an application to \
measure the RMS of CASAimages for given channels.')

    #=== Required arguments ===
    parser.add_argument('-c', '--casaimage_path', 
                        help='Full path to the CASAImage file',
                        required=True, action="store", type=str)

    parser.add_argument('-o', '--output',
                        help='Output .dat file name or full path.',
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    parser.add_argument('-ad', '--all_dim',
                    help='If True (defalut), the RMS will be computed \
for all channels if false, only the channels defined will be used',
                    required=False, action="store_false")

    parser.add_argument('-cmin', '--channel_min',
                    help='Index of the first spectral channel',
                    required=False, default=0, nargs='?', type=int)

    parser.add_argument('-cmax', '--channel_max',
                    help='Index of the last spectral channel',
                    required=False, default=100, nargs='?', type=int)

    parser.add_argument('-r', '--robust',
                    help='If True (default), the RMS will be \
computed using a robust method',
                    required=False, action="store_false")

    parser.add_argument('-pc', '--percentile_cut',
                    help='The (upper) percentile value which below/above \
the data is being ignored when computing the robust RMS',
                    required=False, default=1., nargs='?', type=float)

    parser.add_argument('-w', '--window',
                    help='If true (default) the given window will be used \
for the RMS calculation',
                    required=False, action="store_false")

    parser.add_argument('-wh', '--window_halfsize',
                    help='The halfsize of the rectengular window used',
                    required=False, default=100, nargs='?', type=int)

    parser.add_argument('-wc', '--window_centre',
                    help='The [x,y] coordinates of the window centre. \
If not given, the centre coordinates of the image are used.',
                    required=False, action="append", nargs='*', type=int)

    #=== Application MAIN ===
    args = parser.parse_args()

    if args.window == False:
        args.window_halfsize = None

    if args.window_centre != None:
        args.window_centre = argflatten(args.window_centre)

    #Read in the CASAImage only once
    cim = ds.cim.create_CIM_object(cimpath = args.casaimage_path)

    #Get spectral axis array
    if args.all_dim == True:
        spectral_array, spectral_dim = ds.cim.get_CIM_spectral_axis_array(
            cimpath = cim,
            chan=None, chan_max=None, close=False)

    else:
        N_chan = ds.cim.get_N_chan_from_CIM(cimpath = cim,
                close = False)

        if args.channel_max > N_chan:
            args.channel_max = N_chan

        spectral_array, spectral_dim = ds.cim.get_CIM_spectral_axis_array(
            cimpath = cim,
            chan=args.channel_min, chan_max=args.channel_max, close=False)

    #Get the RMS array
    rms_array, rms_dim = ds.cim.measure_CIM_RMS(cimpath = cim,
                    all_dim = args.all_dim,
                    chan = args.channel_min,
                    chan_max = args.channel_max,
                    pol = 0,
                    robust = args.robust,
                    percentile_cut = args.percentile_cut,
                    window_halfsize = args.window_halfsize,
                    window_centre = args.window_centre,
                    return_dim = True,
                    close = True)

    np.savetxt(args.output, np.column_stack((spectral_array, rms_array)),
            fmt='%e', delimiter=',',
            header='The measured RMS by the cimRMS app from dstack\n\
Note that the grid stacked RMS units are currently wrong!\n\
The columns are: spectral axis [{0:s}] , RMS [{1:s}]\n'.format(
            spectral_dim, rms_dim))

def initTSF():
    """A command line application that cretes a parameter file that can be used to
    initialise targeted source finding using `SoFiA` and `dstack` as a wrapper.

    The created files are structured `python configparser files`. The file will
    be structured as follows:

    [ENV]
    #Parameters defining the envinroment

    fitspath = #Path to the data used
    dsID = #Identification string

    [IMG]
    #Parameters derived from the image

    b_maj = #Beam major axis [arcsec]
    b_min = #Beam minor axis [arcsec]
    b_pa = #Beam position angle [deg]
    d_px = #Image ixelsize [arcsec]
    b_maj_px = #Beam major axis [pixel]
    b_min_px = #Beam minor axis [pixel]

    [SOURCE]
    #Parameters specifying the source

    ID = #Source ID, equal to dsID
    RA = #Source RA [deg]
    Dec = #Source Dec [deg]
    freq = #Source freq [Hz]
    region_coords = #The region in which the source should be searched for by SoFiA

    The file is created under the name:
    {output_path}/{identification_string}_dsconfig.par
    
    NOTE that I use an identification string which is equal to the source name
    read from the catalog. This is to matcjóh the catalog names to the wildcards
    in Snakemeake. This may be surplus.

    The code generatess config files for all sources in one go.

    TO DO: add feture which allows to generate the config file for only a single
    source from the catalog based on source ID. This would enable different sourceID
    and wildcard matches in Snakemake using some mapping.

    Keyword Arguments
    =================

    str -fp or --fits_path:
        Full path (and name) to the fits file that wil be specified in the parset.

    str -cp or --catalog_path:
        Full path to the input source catalog that will be divided into several
        configuration files, which will be used to call `SoFiA` and `dstack` tasks
        The catalog structrure is described in:
        `ds.sourceutil.get_ID_and_pos_list_from_input_catalog`
    
    str -op or --output_path:
        Full path to the folder in whic a config file for each source will be generated

    optional -l or --log:
        Boolean. If True the logger level set to INFO. Set to False by default

    optional -xf or --x_hflood:
        List of ints. List of RA half flood values used for the targeted region size definitions

    optional -yf or --y_hflood:
        List of ints. List of Dec half flood values used for the targeted region size definitions

    optional -zf or --z_hflood:
        List of ints. List of freq half flood values used for the targeted region size definitions

    Return
    ======
    tsf_config: file
        Creates the configuration file defined by the arguments.

    """

    parser = argparse.ArgumentParser(description='This is an application to\
create configuration files for targeted SoFiA source finding')

    #=== Required arguments ===
    parser.add_argument('-fp', '--fits_path', 
                        help='Full path and name of the input .fits image',
                        required=True, action="store", type=str)

    parser.add_argument('-cp', '--catalog_path', 
                        help='Full path and name of the input catalog text file',
                        required=True, action="store", type=str)

    parser.add_argument('-op', '--output_path', 
                        help='Full path to the folder in whic a config file for each source will be generated',
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    parser.add_argument('-l', '--log', 
                    help='If True the logger level set to INFO. Set to False by default', 
                    required=False, action="store_true")

    parser.add_argument('-xf', '--x_hflood',
                    help='List of RA half flood values used for the targeted region size definitions',
                    required=False, action="append", nargs='*', type=int)

    parser.add_argument('-yf', '--y_hflood',
                    help='List of Dec half flood values used for the targeted region size definitions',
                    required=False, action="append", nargs='*', type=int)

    parser.add_argument('-zf', '--z_hflood',
                    help='List of freq half flood values used for the targeted region size definitions',
                    required=False, action="append", nargs='*', type=int)

    #=== Application MAIN ===
    args = parser.parse_args()

    #Set up logging if needed
    if args.log:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #Get the fits image parameters
    b_maj, b_min, b_pa = ds.fitsutil.get_synthesiseb_beam_params_from_fits_header(args.fits_path,
                        return_beam='mean') #Expected to return [arcsec, arcsec, deg]

    cube_params_dict = ds.fitsutil.get_fits_cube_params(args.fits_path)

    if ds.miscutil.deg2arcsec(np.fabs(cube_params_dict['RA---SIN'][3])) != \
    ds.miscutil.deg2arcsec(np.fabs(cube_params_dict['DEC--SIN'][3])):
        raise ValueError('Fits image RA and Dec pixel sized do not match!')

    d_px = float(ds.miscutil.deg2arcsec(np.fabs(cube_params_dict['RA---SIN'][3])))

    #NOTE that b_px_maj b_px_min are computed later

    #Get the source parameters
    ID_list, ra_list, dec_list, freq_list = \
        ds.sourceutil.get_ID_and_pos_list_from_input_catalog(args.catalog_path)    

    sources_2D_position_list = ds.sourceutil.skycords_from_ra_dec_list(ra_list,dec_list)

    #Now get the fill (half flood) values for the region selection
    if args.x_hflood == None:
        args.x_hflood = [15]
    else:
        args.x_hflood = argflatten(args.x_hflood)

    if args.y_hflood == None:
        args.y_hflood = [15]
    else:
        args.y_hflood = argflatten(args.y_hflood)

    if args.z_hflood == None:
        args.z_hflood = [7]
    else:
        args.z_hflood = argflatten(args.z_hflood)

    subcube_coord_list = ds.sourceutil.get_region_coords_from_skycoord_and_freq(args.fits_path,
            skycoord_list=sources_2D_position_list, freq_list=freq_list,
            x_hflood_list=args.x_hflood, y_hflood_list=args.y_hflood,
            z_hflood_list=args.z_hflood)

    #Now generate the config file
    for i in range(0,len(ID_list)):

        #Generate the file name
        config_path = args.output_path + '/{0:s}_dsconfig.par'.format(ID_list[i])

        with open(config_path, 'w') as dsconfig:
            dsconfig.write('#Configuration file for targeted source-finding genereted by dstack'\
                + os.linesep + os.linesep)

            #Define enviromental variables
            dsconfig.write('[ENV]' + os.linesep + os.linesep)

            dsconfig.write('fitspath = {0:s}'.format(args.fits_path) + os.linesep)
            dsconfig.write('dsID = {0:s}'.format(ID_list[i]) + os.linesep)

            #Define image coordinate variables
            dsconfig.write(os.linesep + '[IMG]' + os.linesep + os.linesep)

            dsconfig.write('b_maj = {0:.8f}'.format(b_maj) + os.linesep)
            dsconfig.write('b_min = {0:.8f}'.format(b_min) + os.linesep)
            dsconfig.write('b_pa = {0:.8f}'.format(b_pa) + os.linesep)
            dsconfig.write('d_px = {0:.8f}'.format(d_px) + os.linesep)
            dsconfig.write('b_maj_px = {0:.8f}'.format(float(b_maj / d_px)) + os.linesep)
            dsconfig.write('b_min_px = {0:.8f}'.format(float(b_min / d_px)) + os.linesep)

            #Define source variables
            dsconfig.write(os.linesep + '[SOURCE]' + os.linesep + os.linesep)

            dsconfig.write('ID = {0:s}'.format(ID_list[i]) + os.linesep)
            dsconfig.write('RA = {0:.8f}'.format(ra_list[i]) + os.linesep)
            dsconfig.write('Dec = {0:.8f}'.format(dec_list[i]) + os.linesep)
            dsconfig.write('Freq = {0:.8f}'.format(freq_list[i]) + os.linesep)
            dsconfig.write('Region = {0:s}'.format(subcube_coord_list[i]) + os.linesep)

        dsconfig.close()

        continue

def dsSoFiAtparset():
    """Creates a ``SoFiA`` parset file from template and other parameters.

    A simple application using some quick and dirty hacks to generate parsets for
    targeted source finding in a fits cube. Using this application allows the
    user to read in a template parset and save a parset with defined target region.

    The target region is computed on-the fly for creating the parsets for each source
    specified. 

    Keyword Arguments
    =================
    str -t or --template_path:
        Full path to a template parset file (including its name), which can be
        used to initialize the parset parameters.

    str -op or --output_path:
        Full path to the folder in which the parset will be saved.

    str -pn or --parset_name:
        Name of the parset file created.

    str -cp or --config_path:
        Full path (and name) to the config file that wil be used to specify some
        of the parset params
    
    str -sd or --sofia_output_dir:
        The utput directory in which SoFiA will write the outputs

    optional -l or --log:
        Boolean. If True the logger level set to INFO. Set to False by default

    Return
    ======
    parset: file
        Creates the parset file defined by the arguments.
    """
    parser = argparse.ArgumentParser(description='This is an application to create parset files for SoFiA targeted source finding.')

    #=== Required arguments ===
    parser.add_argument('-t', '--template_path', 
                        help='Full path to a template parset file, which can be used to initialize the parset parameters.',
                        required=True, action="store", type=str)

    parser.add_argument('-op', '--output_path', 
                        help='Full path to the folder in which the parset will be saved.',
                        required=True, action="store", type=str)

    parser.add_argument('-pn', '--parset_name', 
                        help='Name of the parset file created.',
                        required=True, action="store", type=str)

    parser.add_argument('-cp', '--config_path', 
                        help='Full path and name of the input config file',
                        required=True, action="store", type=str)

    parser.add_argument('-sd', '--sofia_output_dir', 
                        help='Path to the directory where SoFiA will generate the output',
                        required=True, action="store", type=str)

    #=== Optional arguments ===
    parser.add_argument('-l', '--log', 
                        help='If True the logger level set to INFO. Set to False by default', 
                        required=False, action="store_true")

    #=== Application MAIN ===
    args = parser.parse_args()

    #Set up logging if needed
    if args.log:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #Construct the output parset string
    if args.output_path[-1] != '/':
        #NOTE this only works for Unix systems...
        #TO DO use sys path generating
        oparset_path = args.output_path + '/' + args.parset_name
    else:
        oparset_path = args.output_path + args.parset_name

    #Read the specifics from the config file
    fitspath, dsID, b_maj, b_min, b_pa, d_px, b_maj_px, b_min_px, source_ID, RA, Dec, freq, region_string = \
    ds.pipelineutil.read_TSF_dsconfig(args.config_path)

    #Generate the parset file
    ds.sourceutil.create_SoFiA_par_from_template(template_path = args.template_path,
                                    out_par_path = oparset_path,
                                    data_path = fitspath,
                                    region_string = region_string,
                                    output_fname = dsID,
                                    output_dir = args.sofia_output_dir)


if __name__ == "__main__":
    pass
