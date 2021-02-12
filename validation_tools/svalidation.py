"""These functions are used to compare sources found by SoFiA in different pipeline setups.
The code is defined for a single source, as the SoFiA parameters (i.e. source ID) can
vary across the different pipelines. Thus, the user have to select the appropriate sources to compare.

The code compare moment maps, contours and spectra. These validation plots can be
used to evaluate the robustness of different pipeline setups, or to decide between
the quality of the output.

We used these diagnostics to validate the grid stacking pipeline as compared to 
image domain stacking ad gridded visibility combination pipelines.
"""
#=== Imports ===
import os, sys
import shutil
import numpy as np
import logging
import warnings
import random
import copy
from scipy import stats

import astropy #Some functions I call explicitly from astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
#from astroquery.utils import download_list_of_fitsfiles

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

import cmocean

import dstack as ds


#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Globals ===
#RCparams for plotting
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

matplotlib.rcParams['xtick.major.size'] = 9
matplotlib.rcParams['ytick.major.size'] = 9

matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.width'] = 3

matplotlib.rcParams['axes.linewidth'] = 2

plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

#4 sampled colors from viridis
c0 = '#440154';#Purple
c1 = '#30678D';#Blue
c2 = '#35B778';#Greenish
c3 = '#FDE724';#Yellow

outlier_color = 'dimgrey'

#Select the colormap and set outliers
_CMAP = matplotlib.cm.viridis
_CMAP.set_bad(color=outlier_color)

#Set the secondary colormap
#_DIV_CMAP = cmocean.cm.oxy
_CMAP2 = matplotlib.cm.plasma
_CMAP2.set_bad(color=outlier_color)

#Set diverging colormap default
#_DIV_CMAP = cmocean.cm.balance
_DIV_CMAP = cmocean.cm.delta
_DIV_CMAP.set_bad(color=outlier_color)

#=== Disable fits warnings ===
#In fact this ignores all Warnings, so comment this line for debugging!
warnings.filterwarnings('ignore', category=Warning, append=True)

#=== Functions ===
def initialise_argument_list(required_list_length,argument_list):
    """Funtion to recursively append the argument lists if needed.

    Always append with the last element.

    Parameters
    ==========
    required_list_length: int
        Goal lenght to append

    argument_list: list
        List to append if needed

    Return
    ======
    argument_list: list
        Appended list

    """
    while len(argument_list) != required_list_length:
        argument_list.append(argument_list[-1])

    return argument_list

def plot_mom0_contour_triangle_matrix(source_ID_list, sofia_dir_list, name_base_list, output_name, N_optical_pixels=600, contour_levels=[1.6,2.7,5.3,8,13,21], masking_list=[True], mask_sigma_list=[3.0], b_maj_list=[30.], b_min_list=[30.], b_pa_list=[0.], color_list=[None], label_list=[''], col_den_sensitivity_lim_list=[None]):
    """Create a triangle plot from the input SoFiA sources. In the plot diagonal
    the contour plot of each source is shown. The upper triangle is empty,
    while in the lower triangle panels both the i and j panel conbtour plots are
    shown. This is a plot for pair-wise comparision of the input sources.

    In the main diagonal the 3sigma column density sensitivity contours are shown
    in the backround (red), while in the lower triangle only the measured column
    density contours are shown.

    The plot size is determined by the number of outputs.

    The background optical image is generated based on the first source given!

    Parameters
    ==========
    source_ID_list: list of int
        The SoFiA ID of the sources

    sofia_dir_list: list str
        List of the SoFiA directories to compare

    name_base_list: list of str
        List of the name `output.filename` variable defined in the SoFiA template 
        .par. Basically the base of all file names in the rspective SoFiA dir.

    output_name: str
        The name and full path to the output tiangle plot generated.

    N_optical_pixels: int, optional
        Number of pixels for the optical background image.

    contour_levels: list of float, optional
        List containing the column density contours to drawn in units of 10^(20)
        HI atoms / cm^2.

    masking_list: list of bool, optional
        If True, the respective mom0 maps will be msked

    mask_sigma_list: list of float, optional
        If the mom0 map is masked, pixels below this threshold will be masked.
        The values are given in units of column density sensitivity.

    b_maj_list: list of float, optional
        The major axis of the synthesised beam in arcseconds.

    b_min_list: list of float, optional
        The minor axis of the synthesised beam in arcseconds.

    b_pa_list: list of float, optional
        The position angle of the synthesised beam.

    color_list: list of colors, optional
        A list defining each sources color. If None is given a random color is
        generated.

    label_list: list of str, optional
        The name/titile used for each source.
    
    col_den_sensitivity_lim_list: list of float, optional
        A list containing the column density sensitivity for each SoFiA output provided by
        the user, rather than using the by default value computed from the SoFiA RMS value.
        Given in units of 10^20 HI /cm^2
    
    Return
    ======
    output_image: file
        The image created
    """
    #Initialise arguments by recursively appending them
    N_sources = len(source_ID_list)
    assert len(sofia_dir_list) == N_sources, 'More or less sources are given than SoFiA directory paths!'

    masking_list = initialise_argument_list(N_sources,masking_list)
    mask_sigma_list = initialise_argument_list(N_sources, mask_sigma_list)
    b_maj_list = initialise_argument_list(N_sources, b_maj_list)
    b_min_list = initialise_argument_list(N_sources, b_min_list)
    b_pa_list = initialise_argument_list(N_sources, b_pa_list)
    label_list = initialise_argument_list(N_sources,label_list)
    col_den_sensitivity_lim_list = initialise_argument_list(N_sources, col_den_sensitivity_lim_list)

    #The name bases might be the same
    if len(name_base_list) != N_sources:
        name_base_list = initialise_argument_list(N_sources, name_base_list)

    #Generate random colors if needed
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color

    #Get the data arrays and coordinate systems for the plots
    #I only get a single (!) background image: this is the first sources corrsponding background image
    #This will be used as the background for all the contours

    optical_background, optical_wcs, survey = ds.sdiagnostics.get_optical_image_ndarray(source_ID_list[0],
                sofia_dir_list[0], name_base_list[0], N_optical_pixels=N_optical_pixels)

    #Get the moment maps and sensitivities
    mom0_map_list = []
    mom0_wcs_list = []
    mom0_sen_lim_list = []

    for i in range(0,N_sources):

        mom0_map, map_wcs, map_sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
                                        source_ID = source_ID_list[i],
                                        sofia_dir_path = sofia_dir_list[i],
                                        name_base = name_base_list[i],
                                        masking = masking_list[i],
                                        mask_sigma = mask_sigma_list[i],
                                        b_maj = b_maj_list[i],
                                        b_min = b_min_list[i],
                                        col_den_sensitivity = col_den_sensitivity_lim_list[i])

        mom0_map_list.append(mom0_map)
        mom0_wcs_list.append(map_wcs)
        mom0_sen_lim_list.append(map_sen_lim)

    #=== Create the figure
    fig, axes = plt.subplots(figsize=(2 + 4 * N_sources, 2 + 4 * N_sources),
                sharex=True, sharey=True, ncols=N_sources, nrows=N_sources,
                subplot_kw={'projection': optical_wcs})


    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if i < j:
                #Upper triangle blank
                axes[i,j].set_axis_off() #This does not work with projection
               
                #A tricky way to hide the frame around the plots
                axes[i,j].coords.frame.set_color('white')
                axes[i,j].coords.frame.set_linewidth(0)

                ra = axes[i,j].coords[0]
                dec = axes[i,j].coords[1]

                ra.set_ticks_visible(False)
                ra.set_ticklabel_visible(False)
                dec.set_ticks_visible(False)
                dec.set_ticklabel_visible(False)
                ra.set_axislabel('')
                dec.set_axislabel('')


            else:
                axes[i,j].imshow(optical_background, origin='lower', cmap='Greys')
                
                #Plot the contours as function of 10^20 particle / cm^2
                axes[i,j].contour(mom0_map_list[i], levels=contour_levels,
                        transform=axes[i,j].get_transform(mom0_wcs_list[i]),
                        colors=color_list[i], linewidths=1.5, alpha=0.8)

                #Grid
                #axes[i,j].coords.grid(color='white', alpha=0.5, linestyle='solid', linewidth=1)

                ra = axes[i,j].coords[0]
                dec = axes[i,j].coords[1]

                ra.set_ticklabel_visible(False)
                dec.set_ticklabel_visible(False)

                if i == j:
                    #Plot the 3sigma sensitivity limit with red in the bottome
                    axes[i,j].contour(mom0_map_list[i], levels=np.multiply(np.array([3]), mom0_sen_lim_list[i]),
                            transform=axes[i,j].get_transform(mom0_wcs_list[i]),
                            colors='red', linewidths=1.5, alpha=1.)

                if i != j:
                    axes[i,j].contour(mom0_map_list[j], levels=contour_levels,
                            transform=axes[i,j].get_transform(mom0_wcs_list[j]),
                            colors=color_list[j], linewidths=1.5, alpha=0.8)

                
                else:
                    #Add inner title
                    t = ds.sdiagnostics.add_inner_title(axes[i,j], label_list[i], loc=2, 
                            prop=dict(size=16,color=color_list[i]))
                    t.patch.set_ec("none")
                    t.patch.set_alpha(0.5)

                    #Add beam ellipse centre is defined as a fraction of the background image size
                    beam_loc_ra = optical_wcs.array_index_to_world(int(0.03 * N_optical_pixels), int(0.03 * N_optical_pixels)).ra.value
                    beam_loc_dec = optical_wcs.array_index_to_world(int(0.03 * N_optical_pixels), int(0.03 * N_optical_pixels)).dec.value

                    beam_ellip = Ellipse((beam_loc_ra, beam_loc_dec), b_maj_list[i]/3600, b_min_list[i]/3600,
                            b_pa_list[i], fc='black', ec='black', alpha=1., transform=axes[i,j].get_transform('fk5'))
                    axes[i,j].add_patch(beam_ellip)

            if i == (N_sources - 1):
                ra.set_ticklabel_visible(True)
                ra.set_axislabel('RA -- sin', fontsize=18)
            
            if j == 0:
                dec.set_ticklabel_visible(True)
                dec.set_axislabel('Dec --sin', fontsize=18)

    #Some style settings
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    plt.savefig(output_name, bbox_inches='tight')
    plt.close()

def get_common_frame_for_sofia_sources(moment, source_ID, sofia_dir_path, name_base, N_optical_pixels=600, masking=True, mask_sigma=3.0, b_maj=5, b_min=5, col_den_sensitivity_lim=None, temp_fits_path=str(os.getcwd() + '/temp.fits')):
    """This function creates an empty sky image based on a SoFiA source.
    The image centre is defined by the SoFiA cataloge RA and Dec of the source,
    while the image size is given by the user. This function is really similar to
    the function `get_optical_image_ndarray()`, however, the pixel size of the
    image is defined by the pixel size of the SoFiA source cubelet.

    Thus, this function can be used as a frame (background image) to wich the
    SoFiA cubelet moment maps can be projected.

    This is the basis of comparing SoFiA source cubelets of the same sky position,
    but with different cubelet size.

    For more details of the projection see `convert_source_mom_map_to_common_frame()`

    Parameters
    ==========
    moment: int
        The moment map that ndarray is returned (0,1 or 2) 
    
    source_ID: int
        The ID of the selected source. IDs are not pythonic; i.e. the first ID is 1.

    sofia_dir_path: str
        Full path to the directory where the output of SoFiA saved/generated. Has to end with a slash (/)!

    name_base: str
      The `output.filename` variable defined in the SoFiA template .par. Basically the base of all file names.
      However, it has to end with a lower dash (?): _ !
    
    N_optical_pixels: int, optional
        Number of pixels for the optical background image created. The pixel size
        is derived from the SoFiA cubelet provided.

    masking: bool, optional
        If True, pixel values below a certain sensitivity threshold will be masked out.

    mask_sigma: int, optional
        The masking threshold value. The masking is performed based on the moment0 map.
        The threshold is given in terms of column density sensitivity values (similarly to contour lines)

    b_maj: float, optional
        Angular major axis of the beam [arcsec]
    
    b_min: float, optional
        Angular minor axis of the beam [arcsec]
    
    col_den_sensitivity_lim: float, optional
        The column density sensitivity if the user wants to use a different value than
        from what computed from the SoFiA RMS value. None by default, and so the 
        SoFiA RMS value is used. Useful if e.g. the user wants to use an RMS of the cube
        computed differently from SoFiA. In the units of 10^20 HI / cm^2
 
    temp_fits_path: string, optional
        Full path and name for a temprorary .fits file created while generating
        the HDU for the backround (empty) image.
    
    Return
    ======
    data_array: `numpy.ndarray.ndarray`
        2D array of zeros with N_optical_pixels x N_optical_pixels size
    x: astropy.WCS
        World coordinate system info for the background image
    """
    #Create the backround image from scratch => Use the first source
    source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(source_ID, sofia_dir_path, name_base)
    catalog = ds.sdiagnostics.parse_single_table(catalog_path).to_table(use_names_over_ids=True)

    #Get the background image centre from the SoFiA RA, Dec coordinates of the selected source
    ra = catalog['ra'][source_index]
    dec = catalog['dec'][source_index]
    pos = SkyCoord(ra=ra, dec=dec, unit='deg',equinox='J2000')

    #Get the source's moment map for the pixel size
    mom_map, map_wcs, map_sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = moment,
                                        source_ID = source_ID,
                                        sofia_dir_path = sofia_dir_path,
                                        name_base = name_base,
                                        masking = masking,
                                        mask_sigma = mask_sigma,
                                        b_maj = b_maj,
                                        b_min = b_min,
                                        col_den_sensitivity = col_den_sensitivity_lim)

    #Initialise the background image
    data_array = np.zeros((N_optical_pixels,N_optical_pixels))

    #Create the WCS using the first dources moment map projected pixel size
    w = WCS(naxis=2)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = astropy.wcs.utils.proj_plane_pixel_scales(map_wcs) #pixels in arcseconds
    w.wcs.crpix = [N_optical_pixels // 2 + 1, N_optical_pixels // 2 + 1]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [pos.ra.deg, pos.dec.deg]
    w.naxis = 2
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0

    #Create file and read in and then delete it....
    #Remove file if exists
    if os.path.exists(temp_fits_path):
        os.remove(temp_fits_path)
    
    fits.writeto(filename=temp_fits_path, data=data_array, header=w.to_header(), 
            checksum=True, output_verify='ignore', overwrite=False)
    
    optical_fits = fits.open(temp_fits_path)
    
    #Remove for good
    os.remove(temp_fits_path)

    del mom_map, map_wcs, map_sen_lim, optical_fits

    return data_array, w

def convert_source_mom_map_to_common_frame(moment, source_ID, sofia_dir_path, name_base, optical_wcs, optical_data_array, masking=True, mask_sigma=3.0, b_maj=5, b_min=5, col_den_sensitivity_lim=None, sensitivity=False, flux_density=False, beam_correction=False, b_maj_px=5, b_min_px=5):
    """This function converts a given SoFiA source moment map cubelet to a pre-
    defined reference background image. The background image and its wcs should
    vbe created via `get_common_frame_for_sofia_sources()`

    This function project the given moment (or sensitivity) map onto this backround
    image.

    NOTE that the projection is currently sub-optimal. It is because the moment
    maps [0,0] corner pixel is projected only so a flip is necessary. This also
    results in that the background image has to be large enough for the flip
    to happen....

    TO DO: get a better but similarly quick projection.

    See the comments of the code for more details

    Parameters
    ==========
    moment: int
        The moment map that ndarray is returned (0,1 or 2) 
    
    source_ID: int
        The ID of the selected source. IDs are not pythonic; i.e. the first ID is 1.

    sofia_dir_path: str
        Full path to the directory where the output of SoFiA saved/generated. Has to end with a slash (/)!

    name_base: str
      The `output.filename` variable defined in the SoFiA template .par. Basically the base of all file names.
      However, it has to end with a lower dash (?): _ !
    
    optical_wcs: `astropy.WCS`
        The World Coordinate System information of the background image. 

    optical_data array: `numpy.ndarray.ndarray`
        Background image data cube (should contain zeros only)

    masking: bool, optional
        If True, pixel values below a certain sensitivity threshold will be masked out.

    mask_sigma: int, optional
        The masking threshold value. The masking is performed based on the moment0 map.
        The threshold is given in terms of column density sensitivity values (similarly to contour lines)

    b_maj: float, optional
        Angular major axis of the beam [arcsec]
    
    b_min: float, optional
        Angular minor axis of the beam [arcsec]
    
    col_den_sensitivity_lim: float
        The column density sensitivity if the user wants to use a different value than
        from what computed from the SoFiA RMS value. None by default, and so the 
        SoFiA RMS value is used. Useful if e.g. the user wants to use an RMS of the cube
        computed differently from SoFiA. In the units of 10^20 HI / cm^2
 
    sensitivity: bool, optional
        If true, a triangle plot for the sensitivity is generated. Automatically
        change moment to 0 and normalises the moment0 map with the column density
        sensitivity value for each source (either given, or computed form SoFiA RMs)

    Return
    ======
    transformed_map: `numpy.ndarray.ndarray`
        The projected moment map with new size, equal of the background image

    map_sen: float
        The column density snesitivity limit of the source (inherited from functions
        called and I need to rerturn this for some plotting)
    
    """
    #Get the moment map as a SoFiA output
    mom_map, map_wcs, map_sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = moment,
                                        source_ID = source_ID,
                                        sofia_dir_path = sofia_dir_path,
                                        name_base = name_base,
                                        masking = masking,
                                        mask_sigma = mask_sigma,
                                        b_maj = b_maj,
                                        b_min = b_min,
                                        col_den_sensitivity = col_den_sensitivity_lim,
                                        flux_density = flux_density,
                                        beam_correction = beam_correction,
                                        b_maj_px =b_maj_px,
                                        b_min_px = b_min_px)

    #Transform mom0 map to sensitivity if sensitivity set to True:
    if sensitivity:
        mom_map = np.divide(mom_map, map_sen_lim)

    #Add the moment maps to the background image, however
    #NOTE that the moment maps can be different sizes and the projection to
    # the backround image is non-trivial. I put together a quick routine to
    # transform the moment maps onto the right place and orientation, but
    # it uses the corner pixel (numpy) of each moment map as a reference pixel.
    # As a result, I can easily loop through the moment map pixels and transform
    # to the background image. However, I am not sure why (maybe because I
    # used sources on the southern sky), but the moment maps are rotated in
    # the wrong direction in the Dec (x) axis.
    #
    #NOTE that because of this method the background image has to be large
    # enough for the moment map to be (wrongly) mapped onto.

    #Define the data array wich the moment map will be transformed
    transformed_map = copy.deepcopy(optical_data_array)

    #Get the referenc epixels
    x_ref, y_ref = astropy.wcs.utils.skycoord_to_pixel(
            astropy.wcs.utils.pixel_to_skycoord(0, 0, map_wcs, origin=0),
            optical_wcs, origin=0)

    x_ref_index = int(x_ref)
    y_ref_index = int(y_ref)

    #Add the moment map to the background image
    for j in range(0,np.shape(mom_map)[0]):
        for k in range(0,np.shape(mom_map)[1]):

            #This is a slower pixel-to-pixel tranformation
            #x_ref, y_ref = astropy.wcs.utils.skycoord_to_pixel(
            #astropy.wcs.utils.pixel_to_skycoord(k,j, map_wcs, origin=0), w, origin=0)
            #x_ref_index = int(x_ref)
            #y_ref_index = int(y_ref)
            #transformed_map[y_ref_index, x_ref_index] = mom_map[j,k]

            transformed_map[y_ref_index + j, x_ref_index - k] = mom_map[j,k]

    #Flip the result image along the Dec axis
    transformed_map = np.flip(transformed_map,axis=1)

    #Mask zero values as the input moment maps are masked as well.
    mask = (transformed_map == 0.)
    transformed_map = np.ma.array(transformed_map, mask=mask)

    #Mask out Nan values as well (if present)
    transformed_map = np.ma.array(transformed_map, mask=np.isnan(transformed_map))

    del mom_map, mask, map_wcs

    return transformed_map, map_sen_lim

def plot_momN_triangle_matrix(moment, source_ID_list, sofia_dir_list, name_base_list, output_name, N_optical_pixels=600, masking_list=[True], mask_sigma_list=[3.0], b_maj_list=[30.], b_min_list=[30.], b_pa_list=[0.], color_list=[None], label_list=[''], temp_fits_path=str(os.getcwd() + '/temp.fits'), ident_list=['?'], col_den_sensitivity_lim_list=[None], sensitivity=False):
    """This is a big and complex (poorly written) function to generate a triangle
    matrix showing the SoFiA stamps in the diagonal elements and the pixel-by-pixel
    difference in the lower triangle.

    This is a one-fit for all function.
    And so, it is poorly written with waay too many arguments.

    TO DO: modularise


    Parameters
    ==========
    moment: int
        The moment for wich the triangle plot is generated.

    source_ID_list: list of int
        The SoFiA ID of the sources

    sofia_dir_list: list str
        List of the SoFiA directories to compare

    name_base_list: list of str
        List of the name `output.filename` variable defined in the SoFiA template 
        .par. Basically the base of all file names in the rspective SoFiA dir.

    output_name: str
        The name and full path to the output tiangle plot generated.

    N_optical_pixels: int, optional
        Number of pixels for the optical background image.

    masking_list: list of bool, optional
        If True, the respective mom0 maps will be msked

    mask_sigma_list: list of float, optional
        If the mom0 map is masked, pixels below this threshold will be masked.
        The values are given in units of column density sensitivity.

    b_maj_list: list of float, optional
        The major axis of the synthesised beam in arcseconds.

    b_min_list: list of float, optional
        The minor axis of the synthesised beam in arcseconds.

    b_pa_list: list of float, optional
        The position angle of the synthesised beam.

    color_list: list of colors, optional
        A list defining each sources color. If None is given a random color is
        generated.

    label_list: list of str, optional
        The name/titile used for each source.
    
    temp_fits_path: str, optional
        Full path and name for a temprorary .fits file created while generating
        the HDU for the backround (empty) image.
    
    ident_list: list of str, optional
        A list containing only a single character. This will be used in the top
        panel to indicate which spectra is subtracted from which. It also used to
        append the label list with this string in parentheses.
    
    col_den_sensitivity_lim_list: list of float, optional
        A list containing the column density sensitivity for each SoFiA output provided by
        the user, rather than using the by default value computed from the SoFiA RMS value.
        Given in units of 10^20 HI /cm^2
    
    sensitivity: bool, optional
        If true, a triangle plot for the sensitivity is generated. Automatically
        change moment to 0 and normalises the moment0 map with the column density
        sensitivity value for each source (either given, or computed form SoFiA RMs)

    Return
    ======
    output_image: file
        The image created
 
    """
    #Initialise arguments by recursively appending them
    N_sources = len(source_ID_list)
    assert len(sofia_dir_list) == N_sources, 'More or less sources are given than SoFiA directory paths!'

    masking_list = initialise_argument_list(N_sources,masking_list)
    mask_sigma_list = initialise_argument_list(N_sources, mask_sigma_list)
    b_maj_list = initialise_argument_list(N_sources, b_maj_list)
    b_min_list = initialise_argument_list(N_sources, b_min_list)
    b_pa_list = initialise_argument_list(N_sources, b_pa_list)
    label_list = initialise_argument_list(N_sources,label_list)
    ident_list = initialise_argument_list(N_sources, ident_list)

    col_den_sensitivity_lim_list = initialise_argument_list(N_sources, col_den_sensitivity_lim_list)

    #The name bases might be the same
    if len(name_base_list) != N_sources:
        name_base_list = initialise_argument_list(N_sources, name_base_list)

    #Generate random colors if needed
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color
    
    #If the sensitivity maps are returned, then the moment is set to 0!
    if sensitivity:
        if moment != 0:
            raise Warning('Moment selected: {0:d}, but currently, only the moment0 sensitivity maps are supported!'.format(moment))
            moment = 0

    #=== Create background image
    data_array, w = get_common_frame_for_sofia_sources(moment = moment,
                                        source_ID = source_ID_list[0],
                                        sofia_dir_path = sofia_dir_list[0],
                                        name_base = name_base_list[0],
                                        masking = masking_list[0],
                                        mask_sigma = mask_sigma_list[0],
                                        b_maj = b_maj_list[0],
                                        b_min = b_min_list[0],
                                        col_den_sensitivity_lim = col_den_sensitivity_lim_list[i],
                                        N_optical_pixels = N_optical_pixels)

    #Now we have the background image that has the same pixel size as the moment maps

    #NOTE that all mom map has to have the same pixel size!
    
    #Get the moment maps and sensitivities
    transformed_map_list = []

    #Get max min values for setting up the colorbars
    c_min = np.inf
    c_max = -np.inf

    #Get all the moment maps and transform them into the background image coordinate frame
    for i in range(0,N_sources):
        transformed_map, tmap_sen_lim = convert_source_mom_map_to_common_frame(moment = moment,
                                        source_ID = source_ID_list[i],
                                        sofia_dir_path = sofia_dir_list[i],
                                        name_base = name_base_list[i],
                                        optical_wcs = w,
                                        optical_data_array = data_array,
                                        masking = masking_list[i],
                                        mask_sigma = mask_sigma_list[i],
                                        b_maj = b_maj_list[i],
                                        b_min = b_min_list[i],
                                        col_den_sensitivity_lim = col_den_sensitivity_lim_list[i],
                                        sensitivity = sensitivity)

        if np.amax(transformed_map) > c_max:
            c_max = np.amax(transformed_map)

        if np.amin(transformed_map) < c_min:
            c_min = np.amin(transformed_map)
        
        transformed_map_list.append(transformed_map)

    #=== Get the min max values for the difference maps
    c_diff_min = np.inf
    c_diff_max = -np.inf

    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if j>i:
                #mask NaN values
                diff_map = np.subtract(transformed_map_list[i],transformed_map_list[j]) 
                diff_map = np.ma.array(diff_map, mask=np.isnan(diff_map))
        
                if c_diff_min > np.amin(diff_map):
                    c_diff_min = np.amin(diff_map)
                if c_diff_max < diff_map.max():
                    c_diff_max = diff_map.max()

    del diff_map

    #=== Create the figure
    fig, axes = plt.subplots(figsize=(2 + 4 * N_sources, 2 + 4 * N_sources),
                sharex=True, sharey=True, ncols=N_sources, nrows=N_sources,
                subplot_kw={'projection': w})


    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if i<j:
                #Upper triangle blank
                #A tricky way to hide the frame around the plots
                axes[i,j].coords.frame.set_color('white')
                axes[i,j].coords.frame.set_linewidth(0)

                ra = axes[i,j].coords[0]
                dec = axes[i,j].coords[1]

                ra.set_ticks_visible(False)
                ra.set_ticklabel_visible(False)
                dec.set_ticks_visible(False)
                dec.set_ticklabel_visible(False)
                ra.set_axislabel('')
                dec.set_axislabel('')


            else:
                #Unset ticks and labels
                ra = axes[i,j].coords[0]
                dec = axes[i,j].coords[1]

                ra.set_ticklabel_visible(False)
                dec.set_ticklabel_visible(False)

                if i == j:
                    if moment == 1:
                        mom_ax = axes[i,j].imshow(transformed_map_list[i], alpha=1.,
                            origin='lower', vmin=c_min, vmax=c_max, cmap=_DIV_CMAP)
                    else:
                        mom_ax = axes[i,j].imshow(transformed_map_list[i], alpha=1.,
                            origin='lower', vmin=c_min, vmax=c_max, cmap=_CMAP)

                    axes[i,j].coords.grid(color='white', linewidth=0.5, 
                            alpha=0.15, linestyle='dashed')
                    #axes[i,j].set_axisbelow(True)
               
                    #Add colorbar
                    cbaxes = inset_axes(axes[i,j], width="5.5%", height="75%", 
                            loc='lower right') 
                    cb = plt.colorbar(mom_ax, cax=cbaxes, orientation='vertical')

                    #set ticks and label left side
                    cbaxes.yaxis.set_ticks_position('left')
                    #cbaxes.yaxis.set_label_position('left')
                    
                    #cbaxes.spines['top'].set_edgecolor('white')
                    #cbaxes.spines['right'].set_color('white')
                    #cbaxes.spines['left'].set_color('white')
                    #cbaxes.spines['bottom'].set_color('white')

                    cb.ax.tick_params(colors='white')
                   
                    if sensitivity:
                        cb.ax.set_ylabel(r'SNR', color='black',
                            fontsize = 18, labelpad=10)
                    else:
                        if moment == 0:
                            cb.ax.set_ylabel(r'N$_{HI}$ [10$^{20}$cm$^2$]', color='black',
                                fontsize = 18, labelpad=10)
                        else:
                            cb.ax.set_ylabel(r'v$_{opt}$ [km/s]', color='black',
                                fontsize = 18, labelpad=10)
 
                    #Add inner title
                    t = ds.sdiagnostics.add_inner_title(axes[i,j], 
                            str(label_list[i] + ' ({0:s})'.format(ident_list[i])),
                            loc=2, prop=dict(size=16,color=color_list[i]))
                    t.patch.set_ec("none")
                    t.patch.set_alpha(1.)

                    #Add beam ellipse centre is defined as a fraction of the background image size
                    beam_loc_ra = w.array_index_to_world(int(0.05 * N_optical_pixels), int(0.05 * N_optical_pixels)).ra.value
                    beam_loc_dec = w.array_index_to_world(int(0.05 * N_optical_pixels), int(0.05 * N_optical_pixels)).dec.value

                    beam_ellip = Ellipse((beam_loc_ra, beam_loc_dec), b_maj_list[i]/3600, b_min_list[i]/3600,
                            b_pa_list[i], fc='white', ec='white', alpha=1., transform=axes[i,j].get_transform('fk5'))
                    axes[i,j].add_patch(beam_ellip)


                if i != j:
                    diff_map = np.subtract(transformed_map_list[j],transformed_map_list[i])
                    diff_map = np.ma.array(diff_map, mask=np.isnan(diff_map))  
                    
                    mom_ax = axes[i,j].imshow(diff_map,
                            vmin=c_diff_min, vmax=c_diff_max, alpha = 1., origin='lower',
                            cmap=_CMAP2)
                    
                    axes[i,j].coords.grid(color='white', alpha=0.25,
                            linewidth=0.5, linestyle='dashed')
                
                    #Add colorbar
                    cbaxes = inset_axes(axes[i,j], width="5.5%", height="75%", 
                            loc='lower right') 
                    cb = plt.colorbar(mom_ax, cax=cbaxes, orientation='vertical')

                    #set ticks and label left side
                    cbaxes.yaxis.set_ticks_position('left')
                    #cbaxes.yaxis.set_label_position('left')
                    
                    #cbaxes.spines['top'].set_edgecolor('white')
                    #cbaxes.spines['right'].set_color('white')
                    #cbaxes.spines['left'].set_color('white')
                    #cbaxes.spines['bottom'].set_color('white')

                    cb.ax.tick_params(colors='white')
                    
                    #cb.ax.set_ylabel(r'N$_{HI}$ [10$^{20}$cm$^2$]', color='black',
                    #        fontsize = 18, labelpad=10)

                    #Add inner title
                    t = ds.sdiagnostics.add_inner_title(axes[i,j], 
                            str(r'({0:s} - {1:s})'.format(ident_list[j],ident_list[i])),
                            loc=2, prop=dict(size=16,color='black'))
                    t.patch.set_ec("none")
                    t.patch.set_alpha(1.)

            if i == (N_sources - 1):
                ra.set_ticklabel_visible(True)
                ra.set_axislabel('RA -- sin', fontsize=18)
            
            if j == 0:
                dec.set_ticklabel_visible(True)
                dec.set_axislabel('Dec --sin', fontsize=18)

    #Some style settings
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    plt.savefig(output_name, bbox_inches='tight')
    plt.close()

def plot_flux_density_diff_dependience_on_column_density(source_ID_list, sofia_dir_list, name_base_list, output_fname, N_optical_pixels=600, masking_list=[True], mask_sigma_list=[3.0], b_maj_list=[30.], b_min_list=[30.], b_pa_list=[0.], col_den_sensitivity_lim_list=[None], sensitivity=False, ident_string='?', beam_correction=False, b_maj_px_list=[5], b_min_px_list=[5], col_den_binwidth=0.1, diff_binwidth=0.001, col_den_lim=None):
    """

    """
    #Initialise arguments by recursively appending them
    N_sources = len(source_ID_list)
    assert len(sofia_dir_list) == N_sources, 'More or less sources are given than SoFiA directory paths!'

    masking_list = initialise_argument_list(N_sources,masking_list)
    mask_sigma_list = initialise_argument_list(N_sources, mask_sigma_list)
    b_maj_list = initialise_argument_list(N_sources, b_maj_list)
    b_min_list = initialise_argument_list(N_sources, b_min_list)
    b_pa_list = initialise_argument_list(N_sources, b_pa_list)
    col_den_sensitivity_lim_list = initialise_argument_list(N_sources, col_den_sensitivity_lim_list)
    b_maj_px_list = initialise_argument_list(N_sources, b_maj_px_list)
    b_min_px_list = initialise_argument_list(N_sources, b_min_px_list)

    #The name bases might be the same
    if len(name_base_list) != N_sources:
        name_base_list = initialise_argument_list(N_sources, name_base_list)

    #=== Create background image
    data_array, w = get_common_frame_for_sofia_sources(moment = 0,
                                        source_ID = source_ID_list[0],
                                        sofia_dir_path = sofia_dir_list[0],
                                        name_base = name_base_list[0],
                                        masking = masking_list[0],
                                        mask_sigma = mask_sigma_list[0],
                                        b_maj = b_maj_list[0],
                                        b_min = b_min_list[0],
                                        col_den_sensitivity_lim = col_den_sensitivity_lim_list[0],
                                        N_optical_pixels = N_optical_pixels)

    #Now we have the background image that has the same pixel size as the moment maps
    #NOTE that all mom map has to have the same pixel size!
    
    #Get the moment maps and sensitivities
    transformed_map_list = []
    transformed_map_sensitivity_limit_list = []

    #Get all the moment maps and transform them into the background image coordinate frame
    for i in range(0,N_sources):
        transformed_map, tmap_sen_lim = convert_source_mom_map_to_common_frame(moment = 0,
                                        source_ID = source_ID_list[i],
                                        sofia_dir_path = sofia_dir_list[i],
                                        name_base = name_base_list[i],
                                        optical_wcs = w,
                                        optical_data_array = data_array,
                                        masking = masking_list[i],
                                        mask_sigma = mask_sigma_list[i],
                                        b_maj = b_maj_list[i],
                                        b_min = b_min_list[i],
                                        col_den_sensitivity_lim = col_den_sensitivity_lim_list[i],
                                        sensitivity = sensitivity)
        
        transformed_map_list.append(transformed_map)
        transformed_map_sensitivity_limit_list.append(tmap_sen_lim)

    #Get the flux density maps
    transformed_flux_density_map_list = []
    transformed_flux_density_map_sensitivity_limit_list = []

    for i in range(0,N_sources):
        transformed_map, tmap_sen_lim = convert_source_mom_map_to_common_frame(moment = 0,
                                        source_ID = source_ID_list[i],
                                        sofia_dir_path = sofia_dir_list[i],
                                        name_base = name_base_list[i],
                                        optical_wcs = w,
                                        optical_data_array = data_array,
                                        masking = masking_list[i],
                                        mask_sigma = mask_sigma_list[i],
                                        b_maj = b_maj_list[i],
                                        b_min = b_min_list[i],
                                        col_den_sensitivity_lim = col_den_sensitivity_lim_list[i],
                                        sensitivity = False,
                                        flux_density = True,
                                        beam_correction = beam_correction,
                                        b_maj_px = b_maj_px_list[i],
                                        b_min_px = b_min_px_list[i])
        
        transformed_flux_density_map_list.append(transformed_map)
        transformed_flux_density_map_sensitivity_limit_list.append(tmap_sen_lim)
    
    #=== Normalise the density maps and get the difference
    diff_map = np.subtract(transformed_flux_density_map_list[0],
                            transformed_flux_density_map_list[1]) 
   
    diff_map = np.ma.array(diff_map, mask=np.isnan(diff_map))

    #Similar vbut in SNR difference
    snr_sen_lim = 3

    #=== Get the average and std of the measured column densities of the two input map
    column_density_mean = np.divide(np.add(transformed_map_list[0],
                                    transformed_map_list[1]), 2)
    
    #Masking Nan values
    column_density_mean = np.ma.array(column_density_mean, mask=np.isnan(column_density_mean))

    #Only for moment 2 maps
    #diff_map = np.ma.array(diff_map, mask=np.ma.getmask(column_density_mean))

    #explicitly cut the masked values
    # NOTE now I am working with numpy arrays not masked arrays!
    #
    p_col_den = copy.deepcopy(np.ma.compressed(column_density_mean))
    p_diff_map = copy.deepcopy(np.ma.compressed(diff_map))

    #=== Set parameters for histograms and plot limits
    #Set an upper limint in column density
    if col_den_lim != None:
        if col_den_lim[0] != None:
            col_den_lower_lim = col_den_lim[0]
        else:
            col_den_lower_lim = np.amin(column_density_mean)
        
        if col_den_lim[1] != None:
            col_den_upper_lim = col_den_lim[1]
        else:
            col_den_upper_lim = np.amax(column_density_mean) + col_den_binwidth

    else:
        col_den_upper_lim = np.amax(column_density_mean) + col_den_binwidth
        #col_den_upper_lim = 1.
    
        col_den_lower_lim = np.amin(column_density_mean)

    #Limit in flux difference
    diff_upper_lim = np.ceil(np.amax(p_diff_map))
    diff_lower_lim = -1 * np.ceil(np.abs(np.amin(p_diff_map)))
    #I know that the lower lim will be negative...

    #Make a cut in the data (in column density not just in the plot range)
    p_diff_map = p_diff_map[p_col_den <= col_den_upper_lim]
    p_col_den = p_col_den[p_col_den <= col_den_upper_lim]

    #Get rid of the lower limit points if relevant
    p_diff_map = p_diff_map[p_col_den >= col_den_lower_lim]
    p_col_den = p_col_den[p_col_den >= col_den_lower_lim]

    #Get the positive and negative column densities
    pos_def_col_den = p_col_den[p_diff_map >= 0]
    pos_def_diff_map = p_diff_map[p_diff_map >= 0]

    neg_def_col_den = p_col_den[p_diff_map < 0]
    neg_def_diff_map = p_diff_map[p_diff_map < 0]

    #=== Create the plot
    #Following Matplotlibs example:
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/scatter_hist.html
    
    #setting up spacing and define axes
    # main plot x, y; histogram plots width; spacing for axis; distance between
    # main plot and histograms
    # Use fig_width to determine the x axis size, and the tight_layout() will
    # automatically re-scale the plot so the main plot rato is what I defined,
    # and the spacing and histogram plot widths are equal sizes
    #
    fig_x, fig_y, subfig, figspace, figskip = 9, 5, 1.5, 0.4, 0.1 
    figsum = fig_x + fig_y + subfig + figspace + figskip
    fig_width = 10
    
    rect_scatter = np.divide(np.array([figspace, figspace, fig_x, fig_y]), figsum)
    rect_histx = np.divide(np.array([figspace, figspace + fig_y + figskip, fig_x, subfig]), figsum)
    rect_histy = np.divide(np.array([figspace + fig_x + figskip, figspace, subfig, fig_y]), figsum)


    #create the plot with the axes
    fig = plt.figure(figsize=(fig_width,fig_width))
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False) 
     
    #The plots
    #ax_scatter.scatter(p_col_den, p_diff_map, marker='o', s=5, c=c2)

    #Plot the 'positive end' green and the negative 'blue' 
    ax_scatter.scatter(pos_def_col_den, pos_def_diff_map, marker='o', s=5, c=c2, alpha=1.)
    ax_scatter.scatter(neg_def_col_den, neg_def_diff_map, marker='o', s=5, c=c1, alpha=1.)
 
    #Set scatter plot limits
    ax_scatter.set_xlim((col_den_lower_lim - col_den_binwidth,
        col_den_upper_lim + col_den_binwidth))
   
    #ax_scatter.set_ylim((diff_lower_lim - diff_binwidth,
    #    diff_upper_lim + diff_binwidth))
   
    #= Histograms
    #Top
    col_den_bins = np.arange(col_den_lower_lim, col_den_upper_lim + col_den_binwidth,
            col_den_binwidth)

    ax_histx.hist(p_col_den, bins=col_den_bins, histtype='step',
            linewidth=2, color=c0)
    
    ax_histx.hist(pos_def_col_den, bins=col_den_bins, histtype='stepfilled',
            color=c2, alpha=0.75)
    ax_histx.hist(neg_def_col_den, bins=col_den_bins, histtype='stepfilled',
            color=c1, alpha=0.75)
 
    ax_histx.set_xlim(ax_scatter.get_xlim())

    #Right
    diff_map_bins = np.arange(diff_lower_lim, diff_upper_lim + diff_binwidth,
            diff_binwidth)

    #Compute the mean of the scatterplot
    mean_diff = np.mean(p_diff_map)

    ax_histy.hist(pos_def_diff_map, bins=diff_map_bins, orientation='horizontal',
            histtype='stepfilled', color=c2)
    
    ax_histy.hist(neg_def_diff_map, bins=diff_map_bins, orientation='horizontal',
            histtype='stepfilled', color=c1)

    ax_histy.set_ylim(ax_scatter.get_ylim())

    #= Binned statistics
    bin_average, bin_edges, binnumber = stats.binned_statistic(p_col_den, p_diff_map,
            statistic='mean', bins=col_den_bins)
            #statistic='mean', bins=np.size(col_den_bins-2),
            #range=(col_den_lower_lim, col_den_upper_lim + col_den_binwidth))

    bin_std, bin_edges, binnumber = stats.binned_statistic(p_col_den, p_diff_map,
            statistic='std', bins=col_den_bins)
            #statistic='std', bins=np.size(col_den_bins-2),
            #range=(col_den_lower_lim, col_den_upper_lim + col_den_binwidth))

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax_scatter.errorbar(bin_centres, bin_average, yerr=bin_std, alpha=1.,
            fmt='D', capsize=2, elinewidth=1.5, markersize=5, color=c0)

    #= Draw lines
    #Draw x=0 y=0 lines
    ax_scatter.axhline(y=0, ls='--', lw=2., color=outlier_color)
    #ax_scatter.axhline(y=mean_diff, ls='--', lw=2, color=c0)

    #ax_histy.axhline(y=mean_diff, ls='--', lw=2, color=c0)


    #Set labels
    ax_scatter.set_xlabel(r'N$_{HI}$ [10$^{20}$cm$^{-2}$]', fontsize=18)
    ax_scatter.set_ylabel(r'$\Delta$S [mJy/pixel]', fontsize=18)
    ax_histx.set_ylabel('N', fontsize=18)
    ax_histy.set_xlabel('N', fontsize=18)

    #Add inner title
    t = ds.sdiagnostics.add_inner_title(ax_scatter, ident_string, loc=2, prop=dict(size=16))
    t.patch.set_ec("none")
    t.patch.set_alpha(0.5)

    plt.savefig(output_fname,bbox_inches='tight')
    plt.close()


def plot_spectra_triangle_matrix(source_ID_list, sofia_dir_list, name_base_list, output_name, beam_correction_list=[True], b_maj_px_list=[5.], b_min_px_list=[5.0], v_frame_list=['optical'], color_list=[None], label_list=[''], ident_list=['?']):
    """Plot a triangle matrix of the spectra from different SoFiA runs. The diagonal
    panels shows the spectra of each source. The upper triangle shows the relative
    difference of the respectibe spectras (i,j). While the lower panel simply
    shows two spectras on top of each other.

    Parameters
    ==========
    source_ID_list: list of int
        The SoFiA ID of the sources

    sofia_dir_list: list str
        List of the SoFiA directories to compare

    name_base_list: list of str
        List of the name `output.filename` variable defined in the SoFiA template 
        .par. Basically the base of all file names in the rspective SoFiA dir.

    output_name: str
        The name and full path to the output tiangle plot generated.

    N_optical_pixels: int
        Number of pixels for the optical background image.

    contour_levels: list of float
        List containing the column density contours to drawn in units of 10^(20)
        HI atoms / cm^2.

    beam_correction_list: list of bool, optional
        If True, the spectra is converted from Jy/beam to Jy by normalising with
        the synthesiesd beam area

    b_maj_px_list: list of float, optional
        The major axis of the synthesised beam in pixels.

    b_min_px_list: list of float, optional
        The minor axis of the synthesised beam in pixels.

    v_frame_list: list of str, optional
        The velocity frame used for each spectra. Has to be the same for all.

    color_list: list of colors, optional
        A list defining each sources color. If None is given a random color is
        generated.

    label_list: list of str, optional
        The name/titile used for each source.
    
    ident_list: list of str, optional
        A list containing only a single character. This will be used in the top
        panel to indicate which spectra is subtracted from which. It also used to
        append the label list with this string in parentheses.
    
    Return
    ======
    output_image: file
        The image created
    """

    #Initialise arguments by recursively appending them
    N_sources = len(source_ID_list)
    assert len(sofia_dir_list) == N_sources, 'More or less sources are given than SoFiA directory paths!'

    beam_correction_list = initialise_argument_list(N_sources,beam_correction_list)
    b_maj_px_list = initialise_argument_list(N_sources, b_maj_px_list)
    b_min_px_list = initialise_argument_list(N_sources, b_min_px_list)
    v_frame_list = initialise_argument_list(N_sources, v_frame_list)
    label_list = initialise_argument_list(N_sources, label_list)
    ident_list = initialise_argument_list(N_sources, ident_list)

    #All the velocity frames has to be the same
    if all(vf == v_frame_list[0] for vf in v_frame_list) == False:
        raise ValueError('Different velocity frames are defined for the input spectars!')

    #The name bases might be the same
    if len(name_base_list) != N_sources:
        name_base_list = initialise_argument_list(N_sources, name_base_list)

    #Generate random colors if needed
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color

    #Get the spectra and velocity arrays
    flux_list = []
    velocity_list = []

    for i in range(0,N_sources):

        flux, velocity = ds.sdiagnostics.get_spectra_array(source_ID = source_ID_list[i],
                                        sofia_dir_path = sofia_dir_list[i],
                                        name_base = name_base_list[i],
                                        v_frame = v_frame_list[i],
                                        beam_correction = beam_correction_list[i],
                                        b_maj_px = b_maj_px_list[i],
                                        b_min_px = b_min_px_list[i])

        flux_list.append(flux)
        velocity_list.append(velocity)

    flux_list = np.array(flux_list)
    velocity_list = np.array(velocity_list)
    
    #Get the difference arrays of the spectra for the upper triangle
    #Map the i,j indices to a list as: i *  N_sources + j
    isum = []
    diff_array_list = []
    v_index_list = []
    subtraction_ident_list = []
    
    #for the common diff y-axis
    diff_max = -np.inf
    diff_min = np.inf

    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if i<j:
                #Here I create the difference arrays by subtracting the flux value
                # of array b from the flux of a, using the closest velocity channel
                # value of b. I have the assumption that the end of the spectra is
                # always zero (or beginning). Thus, this closest channel subtraction
                # works if the two arrays have different number of channels, or
                # even if they spanning over different velocity regions.
                # This is a slow but quite robust method.
                
                if np.size(flux_list[i]) >= np.size(flux_list[j]):
                    fa = flux_list[i]
                    va = velocity_list[i]
                    
                    fb = flux_list[j]
                    vb = velocity_list[j]

                    v_index_list.append(True)

                else:
                    fa = flux_list[j]
                    va = velocity_list[j]
                    
                    fb = flux_list[i]
                    vb = velocity_list[i]

                    v_index_list.append(False)

                diff_flux = np.zeros(np.size(fa))
                for k in range(0, np.size(fa)):
                    v_index = np.argmin(np.fabs(np.subtract(va[k],vb)))
                    
                    if np.size(flux_list[i]) >= np.size(flux_list[j]):
                        diff_flux[k] = np.subtract(fa[k], fb[v_index])
                    else: 
                        diff_flux[k] = np.subtract(fb[v_index], fa[k])
 
                if diff_min > np.amin(diff_flux):
                    diff_min = np.amin(diff_flux)

                if diff_max < np.amax(diff_flux):
                    diff_max = np.amax(diff_flux)

                isum.append(int(N_sources * i + j))
                diff_array_list.append(diff_flux)
                
                #Note that we always subtract j from i (indices) !
                subtraction_ident_list.append('({0:s} - {1:s})'.format(
                            ident_list[i], ident_list[j]))
 
    #=== Create the figure
    fig, axes = plt.subplots(figsize=(2 + 4 * N_sources, 2 + 4 * N_sources),
                sharex=True, sharey=True, ncols=N_sources, nrows=N_sources)

    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if i<j:

                if int(N_sources * i + j) in isum:
                    panel_index = np.where(np.array(isum) == int(N_sources * i +j))[0][0]
                    
                    if v_index_list[panel_index]:
                        va = velocity_list[i]
                    else:
                        va = velocity_list[j]

                    ax = axes[i,j].twinx()

                    #set grid
                    ax.grid(c='black', ls='--', alpha=0.5, lw=1.5)
                    axes[i,j].grid(axis='x', lw=1.5, alpha=0.5)

                    #Plot
                    ax.step(va, diff_array_list[panel_index], c='red', linewidth=2., alpha=0.5)
                
                    ax.set_ylim((diff_min,diff_max))
                    
                    #Add inner title
                    t = ds.sdiagnostics.add_inner_title(ax,
                            subtraction_ident_list[panel_index],loc=2,
                            prop=dict(size=16, color='red'))
                    t.patch.set_ec("none")
                    t.patch.set_alpha(1)

                   
                    #set ticks
                    axes[i,j].tick_params(length=0.) #Remove ticks
                    
                    if j != (N_sources - 1):
                        ax.tick_params(length=0.)
                        ax.set(yticklabels=[])
                        
                    else:
                        ax.set_ylabel(r'$\Delta$S', fontsize=18)

            else:
                axes[i,j].step(velocity_list[i],flux_list[i],
                c=color_list[i], linewidth=2.5, alpha=0.8)

                axes[i,j].grid(lw=1.5, alpha=0.5)
                axes[i,j].tick_params(length=0.) #Remove ticks

                if i != j:
                        axes[i,j].step(velocity_list[j], flux_list[j],
                                c=color_list[j], linewidth=2.5, alpha=0.8)

                        axes[i,j].grid(lw=1.5, alpha=0.5)
                        axes[i,j].tick_params(length=0.) #Remove ticks
                
                else:
                    #Add inner title
                    t = ds.sdiagnostics.add_inner_title(axes[i,j], label_list[i] \
                            + ' ({0:s})'.format(ident_list[i]), loc=2, 
                            prop=dict(size=16, color=color_list[i]))
                    t.patch.set_ec("none")
                    t.patch.set_alpha(0.5)

            if i == (N_sources - 1):
                axes[i,j].set_xlabel('v$_{opt}$ [km/s]', fontsize=18)
                axes[i,j].tick_params(axis='x', length=matplotlib.rcParams['xtick.major.size'])

            if j == 0:
                axes[i,j].set_ylabel(r'S [Jy]', fontsize=18)
                axes[i,j].tick_params(axis='y', length=matplotlib.rcParams['xtick.major.size'])

    #Some style settings
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


#=== MAIN ===
if __name__ == "__main__":
    #pass
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    #"""
    #2km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/', 'conventional_imaging/']))

    #Use the column densizty sensitivity of the co-added visibility combination
    # for all the sensitivity maps
    mmap, mmap_wcs, sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
        source_ID = 1,
        sofia_dir_path = sofia_dir_path_list[0],
        name_base = 'beam17_all_',
        b_maj = 30,
        b_min = 30)


    ID_list = [1, 1, 1, 2]
    ident_list = ['V', 'G', 'I', 'C']

    for i in range(0,4):
        for j in range(0,4):
            if j>i:
                diff_ident = '({0:s} - {1:s})'.format(ident_list[i],
                        ident_list[j])

                log.info('Creating sensitivity diff against column density plots for {}...'.format(
                    diff_ident))

                for zoom_name, col_den_binwidth, col_den_lim in zip(['high', 'low'], [1., 0.25], [None, (0., 5)]):
                    plot_flux_density_diff_dependience_on_column_density(source_ID_list=[ID_list[i],ID_list[j]],
                        sofia_dir_list = [sofia_dir_path_list[i], sofia_dir_path_list[j]],
                        name_base_list = ['beam17_all_'],
                        output_fname = working_dir + 'validation/sensitivity_column_density_{0:s}_{1:s}{2:s}_map.pdf'.format(
                            zoom_name, ident_list[i], ident_list[j]),
                        N_optical_pixels = 170,
                        masking_list = [True],
                        mask_sigma_list = [3.5],
                        b_maj_list = [30, 30],
                        b_min_list = [30, 30],
                        b_pa_list = [0, 0],
                        ident_string = diff_ident,
                        col_den_sensitivity_lim_list = [sen_lim],
                        beam_correction = True,
                        b_maj_px_list = [5.],
                        b_min_px_list = [5.],
                        col_den_binwidth = col_den_binwidth,
                        diff_binwidth = 0.005,
                        col_den_lim = col_den_lim)

        
                log.info('..done')

    exit()
    #"""

    """
    #2km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/', 'conventional_imaging/']))

    log.info('Creating mom0 contour triangle plot for 2km baselie results...')
    
    #Use the column densizty sensitivity of the co-added visibility combination
    # for all the sensitivity maps
    mmap, mmap_wcs, sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
        source_ID = 1,
        sofia_dir_path = sofia_dir_path_list[0],
        name_base = 'beam17_all_',
        b_maj = 30,
        b_min = 30)

    plot_mom0_contour_triangle_matrix(source_ID_list=[1, 1, 1, 2],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/mom0_with_uniform_sensitivity_contours.pdf',
            N_optical_pixels = 800,
            contour_levels = [1.6, 2.7, 5.3, 8, 13, 21],
            color_list = [c0, c2, c1, outlier_color],
            label_list = ['co-added visibilities', 'stacked grids', 'stacked images', 'conventional imaging'],
            b_maj_list = [30, 30, 30, 30],
            b_min_list = [30, 30, 30, 30],
            b_pa_list = [0, 0, 0, 0],
            col_den_sensitivity_lim_list = [sen_lim])

    log.info('..done')

    exit()
    #"""


    """
    #2km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/', 'conventional_imaging/']))


    for mom in range(0,3):
        log.info('Creating mom{0:d} contour triangle plot for 2km baselie results...'.format(mom))


        plot_momN_triangle_matrix(moment = mom,
                source_ID_list=[1, 1, 1, 2],
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = ['beam17_all_'],
                output_name = working_dir + 'validation/mom{0:d}_map.pdf'.format(mom),
                N_optical_pixels = 170,
                masking_list = [True],
                mask_sigma_list = [3.0],
                color_list = [c0, c2, c1, outlier_color],
                label_list = ['co-added visibilities', 'stacked grids', 'stacked images', 'conventional imaging'],
                ident_list = ['V', 'G', 'I', 'C'],
                b_maj_list = [30, 30, 30, 30],
                b_min_list = [30, 30, 30, 30],
                b_pa_list = [0, 0, 0, 0])

        
        log.info('..done')


        #Also create the sensitivity map
        if mom == 0:
            log.info('Creating the sensitivity triangle plot...')


            #Use the column densizty sensitivity of the co-added visibility combination
            # for all the sensitivity maps
            mmap, mmap_wcs, sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
                source_ID = 1,
                sofia_dir_path = sofia_dir_path_list[0],
                name_base = 'beam17_all_',
                b_maj = 30,
                b_min = 30)


            plot_momN_triangle_matrix(moment = mom,
                source_ID_list=[1, 1, 1, 2],
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = ['beam17_all_'],
                output_name = working_dir + 'validation/sensitivity_map.pdf'.format(mom),
                N_optical_pixels = 170,
                masking_list = [True],
                mask_sigma_list = [3.0],
                color_list = [c0, c2, c1, outlier_color],
                label_list = ['co-added visibilities', 'stacked grids', 'stacked images', 'conventional imaging'],
                ident_list = ['V', 'G', 'I', 'C'],
                b_maj_list = [30, 30, 30, 30],
                b_min_list = [30, 30, 30, 30],
                b_pa_list = [0, 0, 0, 0],
                col_den_sensitivity_lim_list = [sen_lim, sen_lim, sen_lim, sen_lim],
                sensitivity = True)
        
            log.info('...done')

            #exit()

    exit()
    #"""
    
    #"""
    #6km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/']))


    for mom in range(0,3):
        log.info('Creating mom{0:d} contour triangle plot for 6km baselie results...'.format(mom))

        plot_momN_triangle_matrix(moment = mom,
                source_ID_list=[1, 1, 1],
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = ['beam17_all_'],
                output_name = working_dir + 'validation/mom{0:d}_map.pdf'.format(mom),
                N_optical_pixels = 400,
                masking_list = [True],
                mask_sigma_list = [3.],
                color_list = [c0, c2, c1],
                label_list = ['co-added visibilities', 'stacked grids', 'stacked images'],
                ident_list = ['V', 'G', 'I'],
                b_maj_list = [12, 12, 12],
                b_min_list = [12, 12, 12],
                b_pa_list = [0, 0, 0])

        log.info('..done')
        
        #Also create the sensitivity map
        if mom == 0:
            log.info('Creating the sensitivity triangle plot...')


            #Use the column densizty sensitivity of the co-added visibility combination
            # for all the sensitivity maps
            mmap, mmap_wcs, sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
                source_ID = 1,
                sofia_dir_path = sofia_dir_path_list[0],
                name_base = 'beam17_all_',
                b_maj = 12,
                b_min = 12)


            plot_momN_triangle_matrix(moment = mom,
                source_ID_list=[1, 1, 1],
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = ['beam17_all_'],
                output_name = working_dir + 'validation/sensitivity_map.pdf'.format(mom),
                N_optical_pixels = 400,
                masking_list = [True],
                mask_sigma_list = [3.0],
                color_list = [c0, c2, c1, outlier_color],
                label_list = ['co-added visibilities', 'stacked grids', 'stacked images'],
                ident_list = ['V', 'G', 'I'],
                b_maj_list = [12, 12, 12],
                b_min_list = [12, 12, 12],
                b_pa_list = [0, 0, 0],
                col_den_sensitivity_lim_list = [sen_lim],
                sensitivity = True)
        
            log.info('...done')


    exit()
    #"""
 
    #"""
    #2km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/', 'conventional_imaging/']))

    log.info('Creating spectra triangle plot for 2km baselie results...')


    plot_spectra_triangle_matrix(source_ID_list = [1, 1, 1, 2],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/spectras.pdf',
            beam_correction_list = [True, True, True, False],
            b_maj_px_list = [5.0],
            b_min_px_list = [5.0],
            color_list = [c0, c2, c1, outlier_color],
            label_list = ['co-added visibilities', 'stacked grids', 'stacked images', 'conventional imaging'],
            ident_list = ['V', 'G', 'I', 'C'])

    log.info('...done')

    exit()
    #"""

    #6km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/']))

    log.info('Creating spectra triangle plot for 6km baselie results...')

    plot_spectra_triangle_matrix(source_ID_list = [1, 1, 1],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/spectras.pdf',
            beam_correction_list = [True, True, True],
            b_maj_px_list = [6.0],
            b_min_px_list = [6.0],
            color_list = [c0, c2, c1],
            label_list = ['co-added visibilities', 'stacked grids', 'stacked images'],
            ident_list = ['V', 'G', 'I'])

    log.info('...done')

    exit()

    #"""
    #2km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/', 'conventional_imaging/']))

    log.info('Creating mom0 contour triangle plot for 2km baselie results...')

    plot_mom0_contour_triangle_matrix(source_ID_list=[1, 1, 1, 2],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/mom0_with_contours.pdf',
            N_optical_pixels = 800,
            contour_levels = [1.6, 2.7, 5.3, 8, 13, 21],
            color_list = [c0, c2, c1, outlier_color],
            label_list = ['co-added visibilities', 'stacked grids', 'stacked images', 'conventional imaging'],
            b_maj_list = [30, 30, 30, 30],
            b_min_list = [30, 30, 30, 30],
            b_pa_list = [0, 0, 0, 0])

    log.info('..done')

    exit()
    #"""

    #"""
    #6km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/']))

    log.info('Creating mom0 contour triangle plot for 6km baselie results...')

    plot_mom0_contour_triangle_matrix(source_ID_list=[1, 1, 1],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/mom0_with_contours.pdf',
            N_optical_pixels = 800,
            contour_levels = [1.6, 2.7, 5.3, 8, 13, 21],
            color_list = [c0, c2, c1],
            label_list=['co-added visibilities', 'stacked grids', 'stacked images'])

    log.info('...done')


    exit()
    #"""

    #"""
    #Chiles example SoFiA analysis
    ds.sdiagnostics.create_complementary_figures_to_sofia_output(
        sofia_dir_path = '/home/krozgonyi/Desktop/chiles_example/runSoFiA/',
        name_base = 'chiles_example_',
        N_optical_pixels = 100,
        masking = True,
        mask_sigma = 3,
        contour_levels = [1.6, 2.7, 5.3, 8, 13, 21],
        b_maj = 7,
        b_min = 5,
        b_pa = -45,
        beam_correction = True, 
        b_maj_px = 4, 
        b_min_px = 4,
        v_frame = 'optical')

    exit()
    #"""

    #DINGO pilot beam17 example analysis

    #2km baselines
    stacking_method_sofia_output_list = ['/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/stacked_grids/',
        '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/stacked_images/',
        '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/co_added_visibilities/',
        '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/conventional_imaging/']

    for stacking_sofia_output in stacking_method_sofia_output_list:
        ds.sdiagnostics.create_complementary_figures_to_sofia_output(
            sofia_dir_path = stacking_sofia_output,
            name_base = 'beam17_all_',
            N_optical_pixels = 900,
            masking = True,
            mask_sigma = 3,
            contour_levels = [1.6, 2.7, 5.3, 8, 13, 21],
            b_maj = 30,
            b_min = 30,
            b_pa = 0,
            beam_correction = True, 
            b_maj_px = 5, 
            b_min_px = 5,
            v_frame = 'optical')

    #6 km baselines
    high_res_stacking_method_sofia_output_list = ['/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution/stacked_grids/',
        '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution/stacked_images/',
        '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution/co_added_visibilities/']

    for stacking_sofia_output in high_res_stacking_method_sofia_output_list:
        ds.sdiagnostics.create_complementary_figures_to_sofia_output(
            sofia_dir_path = stacking_sofia_output,
            name_base = 'beam17_all_',
            N_optical_pixels = 900,
            masking = True,
            mask_sigma = 3,
            contour_levels = [1.6, 2.7, 5.3, 8, 13, 21],
            b_maj = 12,
            b_min = 12,
            b_pa = 0,
            beam_correction = True, 
            b_maj_px = 6, 
            b_min_px = 6,
            v_frame = 'optical')

    exit()

