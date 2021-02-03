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
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
 
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

def plot_mom0_contour_triangle_matrix(source_ID_list, sofia_dir_list, name_base_list, output_name, N_optical_pixels=600, contour_levels=[1.6,2.7,5.3,8,13,21], masking_list=[True], mask_sigma_list=[3.0], b_maj_list=[30.], b_min_list=[30.], b_pa_list=[0.], color_list=[None], label_list=['']):
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

    N_optical_pixels: int
        Number of pixels for the optical background image.

    contour_levels: list of float
        List containing the column density contours to drawn in units of 10^(20)
        HI atoms / cm^2.

    masking_list: list of bool
        If True, the respective mom0 maps will be msked

    mask_sigma_list: list of float
        If the mom0 map is masked, pixels below this threshold will be masked.
        The values are given in units of column density sensitivity.

    b_maj_list: list of float
        The major axis of the synthesised beam in arcseconds.

    b_min_list: list of float
        The minor axis of the synthesised beam in arcseconds.

    b_pa_list: list of float
        The position angle of the synthesised beam.

    color_list: list of colors
        A list defining each sources color. If None is given a random color is
        generated.

    label_list: list of str
        The name/titile used for each source.
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
                sofia_dir_list[0], name_base_list[0], N_optical_pixels)

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
                                        b_min = b_min_list[i])


        mom0_map_list.append(mom0_map)
        mom0_wcs_list.append(map_wcs)
        mom0_sen_lim_list.append(map_sen_lim)

    #=== Create the figure
    fig, axes = plt.subplots(figsize=(2 + 4 * N_sources, 2 + 4 * N_sources),
                sharex=True, sharey=True, ncols=N_sources, nrows=N_sources,
                subplot_kw={'projection': optical_wcs})


    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if i<j:
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


def plot_momN_triangle_matrix(moment, source_ID_list, sofia_dir_list, name_base_list, output_name, N_optical_pixels=600, masking_list=[True], mask_sigma_list=[3.0], b_maj_list=[30.], b_min_list=[30.], b_pa_list=[0.], color_list=[None], label_list=['']):
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
    label_list = initialise_argument_list(N_sources,label_list)

    #The name bases might be the same
    if len(name_base_list) != N_sources:
        name_base_list = initialise_argument_list(N_sources, name_base_list)

    #Generate random colors if needed
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color

    #Get the data arrays and coordinate systems for the plots
    #I only get a single (!) background image: this is an empty image corrsponding
    # to the first source sky coordinates

    optical_background, optical_wcs, survey = ds.sdiagnostics.get_optical_image_ndarray(source_ID_list[0],
                sofia_dir_list[0], name_base_list[0], N_optical_pixels=N_optical_pixels,
                survey=None)


    #Mask all pixels
    source_index, catalog_path, cubelet_path_dict, spectra_path = ds.sdiagnostics.get_source_files(source_ID_list[0], sofia_dir_path_list[0], name_base_list[0])
    

    catalog = ds.sdiagnostics.parse_single_table(catalog_path).to_table(use_names_over_ids=True)

    #Get the optical image centre from the SoFiA RA, Dec coordinates of the selected source
    ra = catalog['ra'][source_index]
    dec = catalog['dec'][source_index]
    pos = SkyCoord(ra=ra, dec=dec, unit='deg',equinox='J2000')


    mom_map, map_wcs, map_sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
                                        source_ID = source_ID_list[0],
                                        sofia_dir_path = sofia_dir_list[0],
                                        name_base = name_base_list[0],
                                        masking = masking_list[0],
                                        mask_sigma = mask_sigma_list[0],
                                        b_maj = b_maj_list[0],
                                        b_min = b_min_list[0])


    #print(astropy.wcs.utils.proj_plane_pixel_scales(map_wcs))
    #print(6/3600)

    #exit()

    data_array = np.zeros((N_optical_pixels,N_optical_pixels))

    #Create the WCS
    w = WCS(naxis=2)
    # The negation in the longitude is needed by definition of RA, DEC
    #w.wcs.cdelt = astropy.wcs.utils.proj_plane_pixel_scales(map_wcs) #pixels in arcseconds
    w.wcs.cdelt = [6 / 3600, 6 / 3600]
    w.wcs.crpix = [N_optical_pixels // 2 + 1, N_optical_pixels // 2 + 1]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [pos.ra.deg, pos.dec.deg]
    w.naxis = 2
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0

    #Create file and read in and the nelete it....
    #Remove file if exists
    temp_fits_path='./a.fits'
    if os.path.exists(temp_fits_path):
        os.remove(temp_fits_path)
    
    fits.writeto(filename=temp_fits_path, data=data_array, header=w.to_header(), 
            checksum=True, output_verify='ignore', overwrite=False)
    
    optical_fits = fits.open(temp_fits_path)
    
    #Remove for good
    os.remove(temp_fits_path)

   

    #==============
    
    #Get the moment maps and sensitivities
    mom_map_list = []
    mom_wcs_list = []
    mom_sen_lim_list = []

    transformed_map_list = []

    for i in range(0,N_sources):

        mom_map, map_wcs, map_sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = moment,
                                        source_ID = source_ID_list[i],
                                        sofia_dir_path = sofia_dir_list[i],
                                        name_base = name_base_list[i],
                                        masking = masking_list[i],
                                        mask_sigma = mask_sigma_list[i],
                                        b_maj = b_maj_list[i],
                                        b_min = b_min_list[i])


        mom_map_list.append(mom_map)
        mom_wcs_list.append(map_wcs)
        mom_sen_lim_list.append(map_sen_lim)


        #print(astropy.wcs.utils.skycoord_to_pixel(astropy.wcs.utils.pixel_to_skycoord([0,1],[0,1], map_wcs, origin=0), w, origin=0))
        #print(astropy.wcs.utils.pixel_to_pixel(map_wcs,optical_wcs))
        

        transformed_map = np.zeros((N_optical_pixels,N_optical_pixels))

        x_ref, y_ref = astropy.wcs.utils.skycoord_to_pixel(astropy.wcs.utils.pixel_to_skycoord(0, 0, map_wcs, origin=0), w, origin=0)
    
        x_ref_index = int(x_ref)
        y_ref_index = int(y_ref)

        print(x_ref_index, y_ref_index)

        for i in range(0,np.shape(mom_map)[0]):
            for j in range(0,np.shape(mom_map)[1]):
                #x_ref, y_ref = astropy.wcs.utils.skycoord_to_pixel(astropy.wcs.utils.pixel_to_skycoord(j,i, map_wcs, origin=0), w, origin=0)
    
                #x_ref_index = int(x_ref)
                #y_ref_index = int(y_ref)


                #transformed_map[y_ref_index, x_ref_index] = mom_map[i,j]

                transformed_map[y_ref_index + i, x_ref_index - j] = mom_map[i,j]

        transformed_map = np.flip(transformed_map,axis=1)

        #Mask zero values
        mask = (transformed_map == 0.)
        transformed_map = np.ma.array(transformed_map, mask=mask)


        transformed_map_list.append(transformed_map)


    #print(astropy.wcs.utils.proj_plane_pixel_scales(map_wcs))
    #print(astropy.wcs.utils.proj_plane_pixel_area(map_wcs))


    #All mom map pixels has to be the same scale and cover the same area.

        


    #exit()

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
                #axes[i,j].imshow(optical_background, origin='lower', cmap='Greys')
                #axes[i,j].coords.grid(color='white', alpha=0.5, linestyle='dashed')
 
                if i == j:
                    axes[i,j].imshow(transformed_map_list[i], alpha=1., origin='lower')
                    #transform=axes[i,j].get_transform(mom_wcs_list[i]))


                    #axes[i,j].coords.grid(color='white', alpha=0.5, linestyle='dashed')
                    #axes[i,j].set_axisbelow(True)
                
                ra = axes[i,j].coords[0]
                dec = axes[i,j].coords[1]

                ra.set_ticklabel_visible(False)
                dec.set_ticklabel_visible(False)
            
                if i != j:
                    #axes[i,j].contour(mom0_map_list[j], levels=contour_levels,
                    #        transform=axes[i,j].get_transform(mom0_wcs_list[j]),
                    #        colors=color_list[j], linewidths=1.5, alpha=0.8)

                    axes[i,j].imshow(np.subtract(transformed_map_list[i],transformed_map_list[j]), alpha = 1., origin='lower')
                    #axes[i,j].coords.grid(color='white', alpha=0.5, linestyle='dashed')
                
                else:
                                        #Add inner title
                    t = ds.sdiagnostics.add_inner_title(axes[i,j], label_list[i], loc=2, 
                            prop=dict(size=16,color=color_list[i]))
                    t.patch.set_ec("none")
                    t.patch.set_alpha(0.5)

                    #Add beam ellipse centre is defined as a fraction of the background image size
                    beam_loc_ra = w.array_index_to_world(int(0.05 * N_optical_pixels), int(0.05 * N_optical_pixels)).ra.value
                    beam_loc_dec = w.array_index_to_world(int(0.05 * N_optical_pixels), int(0.05 * N_optical_pixels)).dec.value

                    beam_ellip = Ellipse((beam_loc_ra, beam_loc_dec), b_maj_list[i]/3600, b_min_list[i]/3600,
                            b_pa_list[i], fc='white', ec='white', alpha=1., transform=axes[i,j].get_transform('fk5'))
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

    beam_correction_list: list of bool
        If True, the spectra is converted from Jy/beam to Jy by normalising with
        the synthesiesd beam area

    b_maj_px_list: list of float
        The major axis of the synthesised beam in pixels.

    b_min_px_list: list of float
        The minor axis of the synthesised beam in pixels.

    v_frame_list: list of str
        The velocity frame used for each spectra. Has to be the same for all.

    color_list: list of colors
        A list defining each sources color. If None is given a random color is
        generated.

    label_list: list of str
        The name/titile used for each source.
    
    ident_list: list of str
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
                    t.patch.set_alpha(0.5)

                   
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

    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/', 'conventional_imaging/']))

    log.info('Creating momN contour triangle plot for 2km baselie results...')

    plot_momN_triangle_matrix(moment = 0,
            source_ID_list=[1, 1, 1, 2],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/momN_map.pdf',
            N_optical_pixels = 170,
            masking_list = [True],
            color_list = [c0, c2, c1, outlier_color],
            label_list = ['co-added visibilities', 'stacked grids', 'stacked images', 'conventional imaging'],
            b_maj_list = [30, 30, 30, 30],
            b_min_list = [30, 30, 30, 30],
            b_pa_list = [0, 0, 0, 0])

    log.info('..done')

    exit()
 
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

