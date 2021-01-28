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

def plot_mom0_contour_triangle_matrix(source_ID_list, sofia_dir_list, name_base_list, output_name, N_optical_pixels=600, contour_levels=[3,5,7,9,11], masking_list=[True], mask_sigma_list=[3.0], b_maj_list=[30.], b_min_list=[30.], b_pa_list=[0.], color_list=[None], label_list=['']):
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
                        colors=color_list[i], linewidths=2.5, alpha=0.65)

                #Grid
                #axes[i,j].coords.grid(color='white', alpha=0.5, linestyle='solid', linewidth=1)

                ra = axes[i,j].coords[0]
                dec = axes[i,j].coords[1]

                ra.set_ticklabel_visible(False)
                dec.set_ticklabel_visible(False)

                if i != j:
                    axes[i,j].contour(mom0_map_list[j], levels=contour_levels,
                            transform=axes[i,j].get_transform(mom0_wcs_list[j]),
                            colors=color_list[j], linewidths=2.5, alpha=0.65)

                
                else:
                    #Plot the 3sigma sensitivity limit with red
                    axes[i,j].contour(mom0_map_list[i], levels=np.multiply(np.array([3]), mom0_sen_lim_list[i]),
                            transform=axes[i,j].get_transform(mom0_wcs_list[i]),
                            colors='red', linewidths=2.5, alpha=0.65)

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







#=== MAIN ===
if __name__ == "__main__":
    #pass
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    """
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

    #exit()
    """

    #"""
    #6km baselines
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_sofia_outputs/high_resolution'

    sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/']))

    log.info('Creating mom0 contour triangle plot for 6km baselie results...')

    plot_mom0_contour_triangle_matrix(source_ID_list=[1, 1, 1],
            sofia_dir_list = sofia_dir_path_list,
            name_base_list = ['beam17_all_'],
            output_name = working_dir + 'validation/mom0_with_contours.pdf',
            N_optical_pixels = 900,
            contour_levels = [3, 9, 30, 90],
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

