"""This script is a collection of imaging function calls of svalidation and
sdiagnostics functions for my Thesis. The imaging scripts can be enabled by
switches in the beginning of the main section of the code. Not only the SoFiA output
imaging scripts are called here, but the 3Dbarollo output (kinematic modelling
results) as well. The following images can be generated:
    
    SoFiA outout:

    - [x] RMS -- channel plot for all methods from the dirty maps
    - [x] RMS -- channel plot for all methods from the residual maps
    - [x] nice mom0 and mom1 maps with column density contours (only grid stacking)
    - [x] nice spectra plot (only grid stacking)
    - [x] spectras triangle plot
    - [x] mom0 triangle plot with 3and 5sigma contours and 1 sigma cut
    - [x] mom1 triangle plot with the mom0 contours and cut
    - [x] measured column density diff -- dynamic range plots for all comparisons
    - [x] measured column density diff -- RA/Dec values plots for all comparisons
    - [x?] N_px -- log(N_HI) histogram for all deep imaging method

    3Dbarolo output:

    - [] rotation curves, inclinations..etc plots to compare models
    - [x] measured p-v diagrams triangle plot
    - [x] model p-v diagrams triangle plot
    - [x] residual p-v diagrams triangle plot

    Hybrid plots

    - [] velocity map residuals triangle plot (!)
    - [] channel-by channel moment 0 maps

The implemented plots are marked with [x].
Furthermore, the conventional imaging has not been completed, so for now I use
results from an old (PB corrected) run.
"""
#=== Imports ===
import os, sys
import os, sys
import shutil
import numpy as np
import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

import cmocean

#Import dstack and svalidation
import dstack as ds
import svalidation #import from current directory
import kvalidation #import from current directory

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

#=================
#=== Functions ===
#=================

def observation_setup_plot(source_ID,
    sofia_dir_path,
    name_base,
    centre_ra,
    centre_dec,
    output_name,
    b_maj=30.,
    b_min=30.,
    b_pa=0.,
    rms_halfwidth = None,
    beam_FWHM_halfwidth=None,
    N_optical_pixels=600,
    sigma_mom0_contours=True,
    mom0_contour_levels=[3.,5.,9.,16.,32.],
    color=None, 
    masking=True,
    mask_sigma=3.5):
    """A specific function displaying the observation setup that includes:

        - background optical image
        - The observed galaxy contours
        - The imaged area borders
        - Primary beam FWHM circle
        - Centre rectangular area in which the RMS is measured 

    That is, this function really is specific for the Thesis.

    Parameters
    ==========
    source_ID: int
        The ID of the selected source. IDs are not pythonic; i.e. the first ID is 1.

    sofia_dir_path_list: str
        Full path to the directory where the output of SoFiA saved/generated.
        Has to end with a slash (/)!

    name_base_list: str
      The `output.filename` variable defined in the SoFiA template .par.
      Basically the base of all file names.
      However, it has to end with a lower dash (?): _ !

    centre_ra: float, optional,
        The RA of the field centre in [degrees] !

    centre_dec: float, optional,
        The Dec of the field centre in [degrees] !

    output_name: str
        The full path to the output dir and the output image name

    b_maj: float, optional
        The major axis of the beam in arcseconds

    b_min: float, optional
        The minor axis of the beam in arcseconds

    b_pa: float, optional
        The position angle of the synthesised beam in degrees

    rms_halfwidth: int, optional
        The half-width of the rectangular window used to measure RMS in [arcseconds]

    beam_FWHM_halfwidth: int, optional
        The half of the FWHM of the primary beam in [arcseconds]

    N_optical_pixels: int, optional
        The background image number of pixels (pixelsize is 1 arcsec)

    sigma_mom0_contours: bool, optimal
        If True, the contour levels should be given in terms of sigma, else in
        column density (10^20)

    mom0_contour_levels: list of float, optional
        The list of mom0 contour levels to be drawn
    
    color: color, optional
        The source contour colors
        
    masking: bool, optional
        If True, the SoFiA cube will be masked

    mask_sigma: float, optional
        The sigma which below the SoFiA cube is potentially masked

    Return
    ======
    output_image: file
        The image created
    """
    #Generate a random color if needed
    if color == None:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color

    #Get moment maps and background image
    optical_im, optical_im_wcs, survey_used = ds.sdiagnostics.get_optical_image_ndarray(
        source_ID_list[0], sofia_dir_path_list[0], name_base_list[0],
        N_optical_pixels = N_optical_pixels, spec_centre = True,
        centre_ra = centre_ra, centre_dec = centre_dec)

    col_den_map, mom0_wcs, col_den_sen_lim = ds.sdiagnostics.get_momN_ndarray(0,
            source_ID, sofia_dir_path, name_base, b_maj=b_maj, b_min=b_min,
            masking=masking, mask_sigma=mask_sigma)

    if sigma_mom0_contours:
            mom0_contour_levels = np.array(mom0_contour_levels) * col_den_sen_lim
            log.info('The column density contours are {}'.format(mom0_contour_levels))

    #Create the plot
    fig = plt.figure(1, figsize=(9,9))
    ax = fig.add_subplot(111, projection=optical_im_wcs)
    
    ax.imshow(optical_im.data,origin='lower',cmap='Greys')
    ax.contour(col_den_map, levels=mom0_contour_levels,
        transform=ax.get_transform(mom0_wcs),
        colors=color, linewidths=2.5, alpha=1.0)    
    
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[0].set_axislabel('RA (J2000)', fontsize=16)
    ax.coords[1].set_axislabel('Dec (J2000)', fontsize=16)

    ax.set_aspect('equal', 'box')

    ax.grid(color='white', linewidth=1.5, alpha=0.5, linestyle='dashed')

    #Add Galaxy name
    ax.text(x=0.525, y=0.725, s="NGC7361", fontsize=18, transform=ax.transAxes)

    #Add beam ellipse centre is defined as a fraction of the background image size
    beam_loc_ra = optical_im_wcs.array_index_to_world(
        int(0.025 * N_optical_pixels), int(0.025 * N_optical_pixels)).ra.value
    beam_loc_dec = optical_im_wcs.array_index_to_world(
        int(0.025 * N_optical_pixels), int(0.025 * N_optical_pixels)).dec.value

    beam_ellip = Ellipse((beam_loc_ra, beam_loc_dec),
        b_maj/3600, b_min/3600, b_pa, fc='black', ec='black', alpha=0.75,
        transform=ax.get_transform('fk5'))
    
    ax.add_patch(beam_ellip)

    #Add RMS rectangle
    arcsec2deg = lambda x: x / 3600

    if rms_halfwidth == None:
        #Use 240' window
        rms_halfwidth = 40.*6. #Assuming 6 arsecpixel size and 80 pixels window 

    rms_rectangle = Rectangle((centre_ra - arcsec2deg(rms_halfwidth),
        centre_dec - arcsec2deg(rms_halfwidth)), arcsec2deg(2*rms_halfwidth),
        arcsec2deg(2*rms_halfwidth), fill=None,
        linestyle='-', linewidth=2.5, ec='black', alpha=1.,
        transform=ax.get_transform('fk5'))    

    ax.add_patch(rms_rectangle)

    #Primary beam FWHM
    #Computed by using the formula
    # radian2arcsec(1.09*(0.21/12)*4*np.log10(2))
    # where the parameters are coming from:
    # https://github.com/ATNF/yandasoft/wiki/linmos#alternate-primary-beam-models

    if beam_FWHM_halfwidth == None:
        radian2arcsec = lambda x: x * 206264.80625 #quick and dirty

        beam_FWHM_halfwidth = radian2arcsec(1.09*(0.21/12)*4*np.log10(2)) / 2 

    beam_FWHM_circle = Circle((centre_ra, centre_dec),
        radius = arcsec2deg(beam_FWHM_halfwidth), fill=None,
        linestyle='--', linewidth=2.5, ec='black', alpha=1.,
        transform=ax.get_transform('fk5'))    

    ax.add_patch(beam_FWHM_circle)

    #Add HIPASS beam
    # TO DO: add this as an optional argument...

    HIPASS_beam_FWHM_halfwidth = 14.3 * 60. / 2
    galaxy_centre_ra = 340.574623
    galaxy_centre_dec = -30.057655

    HIPASS_beam_FWHM_circle = Circle((galaxy_centre_ra, galaxy_centre_dec),
        radius = arcsec2deg(HIPASS_beam_FWHM_halfwidth), fill=None,
        linestyle='--', linewidth=2.5, ec='#A7F0F0', alpha=1.,
        transform=ax.get_transform('fk5'))    

    ax.add_patch(HIPASS_beam_FWHM_circle)


    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


#============
#=== MAIN ===
#============
if __name__ == "__main__":
    #pass
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    #=== Define what to plot ===
    #Decide on Wiener_filtering
    filtering = True

    #Decide if deconvolution happened
    dirty = False

    if dirty == True and filtering == False:
        raise ValueError('Only either the filtering or deconvolution can be disabled at a time!')

    #Decide the resolution
    full_res = False #If True the 6km baseline results are plotted

    #Decide if kinematics plots are created
    kinematics = True

    #===
    setup_plot = False

    #Decide on individual figures to make
    rms_plot = False

    col_density_histogram = False
    normalised_col_density_histogram = False

    simple_grid_mom_contour_plots = False
    simple_grid_spectrum_plot = False

    simple_image_and_hipass_spectrum_plot = False
    simple_grid_and_hipass_spectrum_plot = False

    simple_grid_and_image_mom_contour_plots = False
    simple_grid_and_image_spectrum_plot = False

    simple_grid_image_and_hipass_spectrum_plot = False

    spectra_triangle_plot = False
    mom0_triangle_plot = False
    mom1_triangle_plot = False

    diff_scaling_plots = False

    #Kinematics
    profile_curves = False
    angle_curves = False
    pv_data_trinagle_plot = False
    pv_model_trinagle_plot = False
    pv_residual_trinagle_plot = False


    ringdensplot = True

    #=== Setup variables ===

    if not full_res:
        baseline_length = int(2)

        N_opt_px = 130 #Number of optical background pixels in pixels (not in arcsecond)

        if not kinematics:
            #Define environment variables
            if filtering:
                if not dirty:
                    working_dir = '/home/krozgonyi/Desktop/NGC7361_results/SoFiA/2km_baseline_results/'

                    output_dir = working_dir + 'validation/'

                    #sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
                    #    'stacked_grids/', 'stacked_images/']))

                    sofia_dir_path_list = list(map(working_dir.__add__,['baseline_vis_imaging/',
                        'co_added_visibilities/', 'stacked_grids/', 'stacked_images/',
                        'conventional_imaging/']))

                    rms_dir = working_dir + 'measured_RMS/'

                    rms_file_list = list(map(rms_dir.__add__,[
                        'baseline_imaging_rms.dat', 'co_added_visibilities_rms.dat',
                        'stacked_grids_rms.dat', 'stacked_images_rms.dat',
                        'conventional_imaging_rms.dat']))

                    #rms_file_list = list(map(rms_dir.__add__,[
                    #    'baseline_imaging_rms.dat', 'co_added_visibilities_rms.dat',
                    #    'stacked_grids_rms.dat', 'stacked_images_rms.dat']))

                    rms_colors = ['black', c0, c2, c1, outlier_color]
                    rms_labels = ['B', 'V', 'G', 'I', 'C']
                    #rms_colors = ['black', c0, c2, c1]
                    #rms_labels = ['B', 'V', 'G', 'I']
                    rms_ptitle = 'Wiener-filtering and deconvolution'
                    rms_outlabel = 'filtering'
                    #rms_outlabel = 'filtering_without_C'
                    rms_linestyles = ['-']

                    #Define source and imaging parameters

                    #NOTE when only one element is given the imaging code automatically appends
                    # it to the required size, i.e. all deep imaging has the same parameter

                    #source_ID_list = [4, 4, 3]
                    source_ID_list = [6, 4, 4, 3, 1]
                    beam_correction_list = [False, True, True, True, False]
                    #beam_correction_list = [True]
                    #color_list = [c0, c2, c1]
                    #label_list = ['visibilities', 'stacked grids', 'stacked images']
                    #ident_list = ['V', 'G', 'I']
                    color_list = ['black', c0, c2, c1, outlier_color]
                    label_list = ['baseline visibilities', 'visibilities', 'stacked grids',
                                    'stacked images', 'conventional imaging']
                    ident_list = ['B', 'V', 'G', 'I', 'C']

                else:
                    working_dir = '/home/krozgonyi/Desktop/NGC7361_results/SoFiA/dirty_2km_baseline_results/'

                    output_dir = working_dir + 'validation/'

                    rms_dir = working_dir + 'measured_RMS/'

                    rms_file_list = list(map(rms_dir.__add__,[
                        'baseline_imaging_rms.dat', 'co_added_visibilities_rms.dat',
                        'stacked_grids_rms.dat', 'stacked_images_rms.dat']))

                    #rms_file_list = list(map(rms_dir.__add__,[
                    #    'baseline_imaging_rms.dat', 'co_added_visibilities_rms.dat',
                    #    'stacked_grids_rms.dat']))

                    rms_colors = ['black', c0, c2, c1]
                    rms_labels = ['B', 'V', 'G', 'I']
                    #rms_colors = ['black', c0, c2]
                    #rms_labels = ['B', 'V', 'G']
                    rms_ptitle = 'no deconvolution'
                    rms_outlabel = 'dirty'
                    rms_linestyles = ['-', '--', '-.', '-']
                    #rms_linestyles = ['-', '--', '-.']

            else:
                working_dir = '/home/krozgonyi/Desktop/NGC7361_results/\
SoFiA/no_Wiener_filtering_2km_baseline_results/'

                output_dir = working_dir + 'validation/'

                sofia_dir_path_list = list(map(working_dir.__add__,[
                    'co_added_visibilities/', 'stacked_grids/', 'stacked_images/']))

                rms_dir = working_dir + 'measured_RMS/'

                rms_file_list = list(map(rms_dir.__add__,[
                    'baseline_imaging_rms.dat', 'co_added_visibilities_rms.dat',
                    'stacked_grids_rms.dat', 'stacked_images_rms.dat']))

                rms_colors = ['black', c0, c2, c1]
                rms_labels = ['B', 'V', 'G', 'I']
                rms_ptitle = 'no Wiener-filtering'
                rms_outlabel = 'no_filtering'
                rms_linestyles = ['-']
                #rms_linestyles = ['-', '--', '-.', '-']


                #Define source and imaging parameters

                #NOTE when only one element is given the imaging code automatically appends
                # it to the required size, i.e. all deep imaging has the same parameter

                source_ID_list = [7, 3, 1]
                beam_correction_list = [True]
                color_list = [c0, c2, c1]
                label_list = ['visibilities', 'stacked grids', 'stacked images']
                ident_list = ['V', 'G', 'I']
                rms_outlabel = 'no_filtering'
            
            #=== Set common parameters for 2km 
            #List parameters
            name_base_list = ['beam17_all_']
            b_maj_px_list = [5.]
            b_min_px_list = [5.]
            b_maj_list = [30.]
            b_min_list = [30.]
            b_pa_list = [0.]
            masking_list = [True]
            mask_sigma_list = [3.]
            densplot_mask_sigma_list = [0.05]
            contour_levels = [8., 16.]

            mom_triangle_contours = True
            diff_saturation = 24.

            #=== Single valued parameters
            #Select the gridded dataset for the contour plots
            grid_plot_ID = 2
            image_plot_ID = 4

            #For single contour plots
            N_optical_pixels = 600

            #mom0_contour_levels = [0.5, 1.6, 2.7, 5.3, 13] #in column density 10^20
            mom0_contour_levels = [8, 16, 32, 64, 128] #in column density 10^20

            central_vel = 1247 #central vel for mom1 map contours [km/s]
            delta_vel = 16

        #Kinematics
        else:
            working_dir = '/home/krozgonyi/Desktop/NGC7361_results/3Dbarolo/2km_baseline_results/'

            output_dir = working_dir + 'validation/'

            dir_path_list = list(map(working_dir.__add__,['baseline_vis_imaging/',
        'co_added_visibilities/', 'stacked_grids/', 'stacked_images/',
        'conventional_imaging/']))

            #List parameters
            profile_file_name_list = ['densprof.txt']
            pv_profile_file_name_list = ['rings_final2.txt']
            pv_err_profile_file_name_list = ['rings_final1.txt']
            pv_fits_name_base_list = ['DINGO_J224218.09-300323.8',
                'DINGO_J224218.10-300326.8', 'DINGO_J224218.10-300326.8',
                'DINGO_J224218.05-300325.7', 'NONE']

            contour_levels=[1,2,4,8,16,32,64]
            #contour_levels=[4,8,16]
            color_list = ['black', c0, c2, c1, outlier_color]
            label_list = ['baseline visibilities','co-added visibilities',
                        'stacked grids', 'stacked images', 'conventional imaging']
            ident_list = ['B', 'V', 'G', 'I', 'C']
            S_rms_list = [0.00555033, 0.00518113, 0.00517657, 0.00510539, 0.00625409]

            #Single valued parameters
            channelwidth = 4.
            centre_index = 55
            edge_crop = 9
            ring_crop = 3
            inner_ring_crop = 0
            #S_rms=0.00555033 #This is the actual RMS level of the baseline 


    else:
        baseline_length = int(6)
    
        if not kinematics:    
            #Define environment variables
            working_dir = '/home/krozgonyi/Desktop/NGC7361_results/SoFiA//6km_baseline_results/'

            output_dir = working_dir + 'validation/'

            sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
                'stacked_grids/', 'stacked_images/']))

            #Define source and imaging parameters
            source_ID_list = [3, 4, 8]
            name_base_list = ['beam17_all_']
            beam_correction_list = [True, True, True]
            b_maj_px_list = [6.]
            b_min_px_list = [6.]
            b_maj_list = [12.]
            b_min_list = [12.]
            b_pa_list = [0.]
            masking_list = [True]
            mask_sigma_list = [3.]
            densplot_mask_sigma_list = [0.01]
            color_list = [c0, c2, c1]
            label_list = ['visibilities', 'stacked grids', 'stacked images']
            ident_list = ['V', 'G', 'I']
            contour_levels = [8.,16.]

            grid_plot_ID = 1
            image_plot_ID = 2

            #Single valued parameters
            N_opt_px = 420 #Number of optical background pixels in pixels ???
            mom_triangle_contours = True
            diff_saturation = 24.

            #For single contour plots
            N_optical_pixels = 600

            #mom0_contour_levels = [0.5, 1.6, 2.7, 5.3, 13] #in column density 10^20
            mom0_contour_levels = [8, 16, 32, 64, 128] #in column density 10^20

            central_vel = 1250 #central vel for mom1 map contours [km/s]
            delta_vel = 16

            rms_dir = working_dir + 'measured_RMS/'

            rms_file_list = list(map(rms_dir.__add__,[
                'co_added_visibilities_rms.dat',
                'stacked_grids_rms.dat', 'stacked_images_rms.dat']))

            rms_colors = [c0, c2, c1]
            rms_labels = ['V', 'G', 'I']
            rms_ptitle = 'Wiener-filtering and deconvolution'
            rms_outlabel = 'filtering'
            rms_linestyles = ['-']


        else:
            #kinematics plots:
            working_dir = '/home/krozgonyi/Desktop/NGC7361_results/3Dbarolo/6km_baseline_results/'

            output_dir = working_dir + 'validation/'

            dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
                'stacked_grids/', 'stacked_images/']))

            #List parameters
            profile_file_name_list = ['densprof.txt']
            pv_profile_file_name_list = ['rings_final2.txt']
            pv_err_profile_file_name_list = ['rings_final1.txt']
            pv_fits_name_base_list = ['DINGO_J224218.06-300326.6',
                'DINGO_J224218.05-300326.8', 'DINGO_J224217.92-300325.2']

            contour_levels=[1,2,4,8,16,32,64]
            color_list = [c0, c2, c1]
            label_list = ['co-added visibilities',
                        'stacked grids', 'stacked images']
            ident_list = ['V', 'G', 'I']
            S_rms_list = [0.00440574, 0.00441169, 0.00441169]

            #Single valued parameters
            channelwidth = 4.
            centre_index = 75
            edge_crop = 3
            ring_crop = 5
            inner_ring_crop = 1


    #===========================================================================
    #=== Imaging ===
    if not kinematics:
        #The special setup plot
        if setup_plot:
            log.info('Creating observation setup plot...')

            #Use the grid stacking 2km baseline data

            observation_setup_plot(source_ID = source_ID_list[grid_plot_ID],
                sofia_dir_path = sofia_dir_path_list[grid_plot_ID],
                name_base = name_base_list[0],
                centre_ra = 340.591, #Field centre RA in degrees
                centre_dec = -30.41624, #Field centre Dec in degrees
                output_name = output_dir + '{0:d}km_observation_setup.pdf'.format(
                    baseline_length),
                b_maj = b_maj_list[0],
                b_min = b_min_list[0],
                b_pa = b_pa_list[0],
                rms_halfwidth = None,
                beam_FWHM_halfwidth = None,
                N_optical_pixels = 6144, #The whole area imaged
                sigma_mom0_contours = False,
                mom0_contour_levels = [np.sqrt(2.)],
                #mom0_contour_levels = [1.],
                color = color_list[grid_plot_ID], 
                masking = masking_list[0],
                mask_sigma = mask_sigma_list[0])

            log.info('...done')

        if rms_plot:
            log.info('Creating RMS -- channel plot...')

            svalidation.plot_RMS(rmsfile_list = rms_file_list,
                output_fname = output_dir + '{0:d}km_rms_{1:s}.pdf'.format(
                    baseline_length,rms_outlabel),
                label_list = rms_labels,
                color_list = rms_colors,
                linestyle_list = rms_linestyles,
                rest_frame = 'optical',
                region_list=[None,None,None,None,(319,397)],
                ptitle = rms_ptitle)

            log.info('...done')

        if col_density_histogram:
            log.info('Creating column density histograms...')

            svalidation.plot_column_density_histogram(source_ID_list = source_ID_list,
                sofia_dir_path_list = sofia_dir_path_list,
                name_base_list = name_base_list,
                output_fname = output_dir + '{0:d}km_col_den_hist.pdf'.format(
                    baseline_length),
                N_bins = 25,
                #masking = False,
                masking_list = masking_list,
                mask_sigma_list = densplot_mask_sigma_list,
                b_maj_list = b_maj_list,
                b_min_list = b_maj_list,
                color_list = color_list,
                label_list = ident_list,
                col_den_sensitivity_lim_list = [None],
                conver_from_NHI=True,
                pixelsize_list = [30.],
                inclination_list = [70],
                densplot=False,
                N_optical_pixels = N_opt_px,
                bin_edge_list=np.arange(-3,1.5,0.2))

            log.info('...done')

        if normalised_col_density_histogram:
            log.info('Creating normalised column density histograms...')

            svalidation.plot_column_density_histogram(source_ID_list = source_ID_list,
                sofia_dir_path_list = sofia_dir_path_list,
                name_base_list = name_base_list,
                output_fname = output_dir + '{0:d}km_normalised_col_den_hist.pdf'.format(
                    baseline_length),
                N_bins = 25,
                #masking = False,
                masking_list = masking_list,
                mask_sigma_list = densplot_mask_sigma_list,
                b_maj_list = b_maj_list,
                b_min_list = b_maj_list,
                color_list = color_list,
                label_list = ident_list,
                col_den_sensitivity_lim_list = [None],
                conver_from_NHI=True,
                pixelsize_list = [30.],
                inclination_list = [70],
                densplot=True,
                N_optical_pixels = N_opt_px,
                bin_edge_list=np.arange(-3,1.5,0.2))

            log.info('...done')

        if simple_grid_mom_contour_plots:
            log.info('Creating single mom0 and mom1 contour maps...')

            source_ID = source_ID_list[grid_plot_ID]
            sofia_dir_path = sofia_dir_path_list[grid_plot_ID]
            name_base = name_base_list[0]

            output_name = output_dir + '{0:d}km_grid_contour_plots.pdf'.format(
                baseline_length)

            svalidation.simple_moment0_and_moment1_contour_plot(source_ID_list = [source_ID],
                sofia_dir_path_list = [sofia_dir_path],
                name_base_list = [name_base],
                output_name = output_name,
                b_maj_list = b_maj_list,
                b_min_list = b_min_list,
                b_pa_list = b_pa_list,
                N_optical_pixels = N_optical_pixels,
                #sigma_mom0_contours = True,
                sigma_mom0_contours = False,
                #mom0_contour_levels = mom0_contour_levels,
                mom0_contour_levels = [0.64, 1.29, 2.57, 5.15, 10.3],
                central_vel = central_vel,
                delta_vel = delta_vel,
                N_half_contours_mom1 = 7,
                color_list = [color_list[grid_plot_ID]],
                masking_list = masking_list,
                mask_sigma_list = mask_sigma_list)

            log.info('...done')

        if simple_grid_spectrum_plot:
            log.info('Create single spectra plot...')

            output_name = output_dir + '{0:d}km_grid_spectra.pdf'.format(
                baseline_length)

            svalidation.simple_spectra_plot(source_ID_list = [source_ID_list[grid_plot_ID]],
                sofia_dir_path_list = [sofia_dir_path_list[grid_plot_ID]],
                name_base_list = [name_base_list[0]],
                output_name = output_name,
                beam_correction_list = [True],
                color_list = [color_list[grid_plot_ID]],
                b_maj_px_list = [b_maj_px_list[0]],
                b_min_px_list = [b_min_px_list[0]])

            log.info('...done')

        if simple_grid_and_hipass_spectrum_plot:
            log.info('Create single spectra plot including HIPASS spectra...')

            output_name = output_dir + '{0:d}km_grid_and_HIPASS_spectra.pdf'.format(
                baseline_length)

            svalidation.simple_spectra_plot(source_ID_list = [None, source_ID_list[grid_plot_ID]],
                sofia_dir_path_list = [None, sofia_dir_path_list[grid_plot_ID]],
                name_base_list = [name_base_list[0]],
                output_name = output_name,
                beam_correction_list = [False, True],
                color_list = ['#A7F0F0', color_list[grid_plot_ID]],
                b_maj_px_list = [b_maj_px_list[0]],
                b_min_px_list = [b_min_px_list[0]],
                special_flux_list = [
                '/home/krozgonyi/Desktop/NGC7361_results/SoFiA/2km_baseline_results/NGC7361_hipass_spectra.txt',
                None])

            log.info('...done')

        if simple_image_and_hipass_spectrum_plot:
            log.info('Create single spectra plot including HIPASS spectra...')

            output_name = output_dir + '{0:d}km_image_and_HIPASS_spectra.pdf'.format(
                baseline_length)

            svalidation.simple_spectra_plot(source_ID_list = [None, source_ID_list[image_plot_ID]],
                sofia_dir_path_list = [None, sofia_dir_path_list[image_plot_ID]],
                name_base_list = [name_base_list[0]],
                output_name = output_name,
                beam_correction_list = [False, True],
                color_list = ['#A7F0F0', color_list[image_plot_ID]],
                b_maj_px_list = [b_maj_px_list[0]],
                b_min_px_list = [b_min_px_list[0]],
                special_flux_list = [
                '/home/krozgonyi/Desktop/NGC7361_results/SoFiA/2km_baseline_results/NGC7361_hipass_spectra.txt',
                None])

            log.info('...done')            

        if simple_grid_and_image_mom_contour_plots:
            log.info('Creating single mom0 and mom1 contour maps...')

            source_ID = [source_ID_list[grid_plot_ID],\
                        source_ID_list[image_plot_ID]] 
            sofia_dir_path = [sofia_dir_path_list[grid_plot_ID],\
                        sofia_dir_path_list[image_plot_ID]]
            name_base = name_base_list[0]

            mom0_contour_levels = [1, 2, 4, 8, 16]

            output_name = output_dir + '{0:d}km_GI_contour_plots.pdf'.format(
                baseline_length)

            svalidation.simple_moment0_and_moment1_contour_plot(source_ID_list = source_ID,
                sofia_dir_path_list = sofia_dir_path,
                name_base_list = [name_base],
                output_name = output_name,
                b_maj_list = b_maj_list,
                b_min_list = b_min_list,
                b_pa_list = b_pa_list,
                N_optical_pixels = N_optical_pixels,
                sigma_mom0_contours = False,
                #sigma_mom0_contours = True,
                mom0_contour_levels = mom0_contour_levels,
                #mom0_contour_levels = [0.64, 1.29, 2.57, 5.15, 10.3],
                central_vel = central_vel,
                delta_vel = delta_vel,
                N_half_contours_mom1 = 7,
                color_list = [color_list[grid_plot_ID], color_list[image_plot_ID]],
                masking_list = masking_list,
                mask_sigma_list = mask_sigma_list)

            log.info('...done')

        if simple_grid_and_image_spectrum_plot:
            log.info('Create single spectra plot...')

            output_name = output_dir + '{0:d}km_GI_spectra.pdf'.format(
                baseline_length)

            svalidation.simple_spectra_plot(source_ID_list = [source_ID_list[grid_plot_ID],\
                source_ID_list[image_plot_ID]],
                sofia_dir_path_list = [sofia_dir_path_list[grid_plot_ID],\
                sofia_dir_path_list[image_plot_ID]],
                name_base_list = [name_base_list[0]],
                output_name = output_name,
                beam_correction_list = [True, False],
                color_list = [color_list[grid_plot_ID], color_list[image_plot_ID]],
                b_maj_px_list = [b_maj_px_list[0]],
                b_min_px_list = [b_min_px_list[0]])

            log.info('...done')

        if simple_grid_image_and_hipass_spectrum_plot:
            log.info('Create single GI spectra plot including HIPASS spectra...')

            output_name = output_dir + '{0:d}km_GI_and_HIPASS_spectra.pdf'.format(
                baseline_length)

            svalidation.simple_spectra_plot(source_ID_list = [None,
                source_ID_list[grid_plot_ID], source_ID_list[image_plot_ID]],
                sofia_dir_path_list = [None, sofia_dir_path_list[grid_plot_ID],
                sofia_dir_path_list[image_plot_ID]],
                name_base_list = [name_base_list[0]],
                output_name = output_name,
                beam_correction_list = [False, beam_correction_list[grid_plot_ID],
                beam_correction_list[image_plot_ID]],
                color_list = ['#A7F0F0', color_list[grid_plot_ID],
                color_list[image_plot_ID]],
                b_maj_px_list = [b_maj_px_list[0]],
                b_min_px_list = [b_min_px_list[0]],
                special_flux_list = [
                '/home/krozgonyi/Desktop/NGC7361_results/SoFiA/2km_baseline_results/NGC7361_hipass_spectra.txt',
                None, None])

            log.info('...done')        

        if spectra_triangle_plot:
            log.info('Creating spectra triangle plot for {0:d}km \
baseline results...'.format(baseline_length))

            svalidation.plot_spectra_triangle_matrix(source_ID_list = source_ID_list,
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = name_base_list,
                output_name = output_dir + '{0:d}km_spectra_triangle.pdf'.format(
                baseline_length),
                beam_correction_list = beam_correction_list,
                b_maj_px_list = b_maj_px_list,
                b_min_px_list = b_min_px_list,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list)
        
            log.info('...done')

        if mom0_triangle_plot:
            log.info('Creating mom0 triangle plot for {0:d}km \
baseline results...'.format(baseline_length))

            svalidation.plot_momN_triangle_matrix(moment = 0,
                source_ID_list = source_ID_list,
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = name_base_list,
                output_name = output_dir + '{0:d}km_mom0_map_triangle.pdf'.format(
                baseline_length),
                N_optical_pixels = N_opt_px,
                masking_list = masking_list,
                mask_sigma_list = mask_sigma_list,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list,
                b_maj_list = b_maj_list,
                b_min_list = b_min_list,
                b_pa_list = b_pa_list,
                sensitivity=False,
                contours=mom_triangle_contours,
                contour_levels = contour_levels)

            log.info('...done')


        if mom1_triangle_plot:
            log.info('Creating mom1 triangle plot for {0:d}km \
baseline results...'.format(baseline_length))

            svalidation.plot_momN_triangle_matrix(moment = 1,
                source_ID_list = source_ID_list,
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = name_base_list,
                output_name = output_dir + '{0:d}km_mom1_map_triangle.pdf'.format(
                baseline_length),
                N_optical_pixels = N_opt_px,
                masking_list = masking_list,
                mask_sigma_list = mask_sigma_list,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list,
                b_maj_list = b_maj_list,
                b_min_list = b_min_list,
                b_pa_list = b_pa_list,
                sensitivity=False,
                contours=mom_triangle_contours,
                contour_levels = contour_levels,
                diff_saturation = diff_saturation)

            log.info('...done')

        if diff_scaling_plots:

            mmap, mmap_wcs, sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
                source_ID = source_ID_list[0],
                sofia_dir_path = sofia_dir_path_list[0],
                name_base = name_base_list[0],
                b_maj = b_maj_list[0],
                b_min = b_min_list[0])

            for i in range(0,len(ident_list)):
                for j in range(0,len(ident_list)):
                    #if j>i:
                    diff_ident = '({0:s} - {1:s})'.format(ident_list[i],
                            ident_list[j])
        
                    col_den_binwidth = 21
                    col_den_lim = None                        

                    for orientation, orientation_dependence in zip(['col_den', 'RA', 'Dec'], [False, True, True]):
                        log.info('Creating flux diff against {0:s} plots for {1:}...'.format(
                        orientation, diff_ident))

                        svalidation.plot_flux_density_diff_dependience_on_column_density(
                            source_ID_list=[source_ID_list[i],source_ID_list[j]],
                            sofia_dir_list = [sofia_dir_path_list[i], sofia_dir_path_list[j]],
                            name_base_list = name_base_list,
                            output_fname = output_dir + \
                            'scaling_plots/{0:d}km_sensitivity_column_density_{1:s}_{2:s}{3:s}_map.pdf'.format(
                                baseline_length, orientation, ident_list[i], ident_list[j]),
                            N_optical_pixels = N_opt_px,
                            masking_list = masking_list,
                            mask_sigma_list = mask_sigma_list,
                            b_maj_list = b_maj_list,
                            b_min_list = b_min_list,
                            b_pa_list = b_pa_list,
                            ident_string = diff_ident,
                            col_den_sensitivity_lim_list = [sen_lim],
                            beam_correction = beam_correction_list[1],
                            b_maj_px_list = b_maj_px_list,
                            b_min_px_list = b_min_list,
                            col_den_binwidth = col_den_binwidth,
                            diff_binwidth = 0.1,
                            col_den_lim = col_den_lim,
                            sensitivity = True,
                            logbins = True,
                            orientation_dependence = orientation_dependence,
                            orientation = orientation)
            
                        log.info('..done')

    #Kinematics
    else:
        if profile_curves:
            for profile in ['rotation', 'dispersion', 'density']:
                log.info('Creating {0:s} curves for the {1:d}km \
baseline results...'.format(profile, baseline_length))

                if profile != 'density':
                    p_list = pv_profile_file_name_list
                else:
                    p_list = profile_file_name_list

                kvalidation.plot_profile_curves(rot_dir_list = dir_path_list,
                    profile_file_name_list = p_list,
                    profile = profile,
                    ring_crop = ring_crop,
                    inner_ring_crop = inner_ring_crop,
                    color_list = color_list,
                    label_list = ident_list,
                    output_fname = output_dir + '{0:d}km_{1:s}_curves.pdf'.format(
                        baseline_length,profile))

                log.info('...done')

        if angle_curves:
            for angle_profile in ['inclination', 'position_angle']:
                log.info('Creating {0:s} curves for the {1:d}km \
baseline results...'.format(angle_profile, baseline_length))

                p_list = pv_profile_file_name_list

                kvalidation.plot_angle_curves(rot_dir_list = dir_path_list,
                    profile_file_name_list = p_list,
                    profile_error_file_name_list = pv_err_profile_file_name_list,
                    angle_type = angle_profile,
                    ring_crop = ring_crop,
                    inner_ring_crop = inner_ring_crop,
                    color_list = color_list,
                    label_list = ident_list,
                    output_fname = output_dir + '{0:d}km_{1:s}_curves.pdf'.format(
                        baseline_length, angle_profile))

                log.info('...done')

        if pv_data_trinagle_plot:

            log.info('Creating p-v diagram of the raw data for the {0:d}km\
baseline results...'.format(baseline_length))

            kvalidation.plot_pv_diagram_triangle_plot(rot_dir_list = dir_path_list,
                profile_file_name_list = pv_profile_file_name_list,
                pv_fits_name_base_list = pv_fits_name_base_list,
                plot_type = 'data',
                channelwidth = channelwidth,
                centre_index = centre_index,
                edge_crop = edge_crop,
                ring_crop = ring_crop,
                inner_ring_crop = inner_ring_crop,
                S_rms_list = S_rms_list,
                contour_levels = contour_levels,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list,
                output_fname = output_dir + 'pv_diagram_data_triangle_plot.pdf')

            log.info('...done')

        if pv_model_trinagle_plot:

            log.info('Creating p-v diagram of the raw data for the {0:d}km\
baseline results...'.format(baseline_length))

            kvalidation.plot_pv_diagram_triangle_plot(rot_dir_list = dir_path_list,
                profile_file_name_list = pv_profile_file_name_list,
                pv_fits_name_base_list = pv_fits_name_base_list,
                plot_type = 'model',
                channelwidth = channelwidth,
                centre_index = centre_index,
                edge_crop = edge_crop,
                ring_crop = ring_crop,
                inner_ring_crop = inner_ring_crop,
                S_rms_list = S_rms_list,
                contour_levels = contour_levels,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list,
                output_fname = output_dir + 'pv_diagram_model_triangle_plot.pdf')

            log.info('...done')

        if pv_residual_trinagle_plot:

            log.info('Creating p-v diagram of the raw data for the {0:d}km\
baseline results...'.format(baseline_length))

            kvalidation.plot_pv_diagram_triangle_plot(rot_dir_list = dir_path_list,
                profile_file_name_list = pv_profile_file_name_list,
                pv_fits_name_base_list = pv_fits_name_base_list,
                plot_type = 'residual',
                channelwidth = channelwidth,
                centre_index = centre_index,
                edge_crop = edge_crop,
                ring_crop = ring_crop,
                inner_ring_crop = inner_ring_crop,
                S_rms_list = S_rms_list,
                contour_levels = contour_levels,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list,
                output_fname = output_dir + 'pv_diagram_residual_triangle_plot.pdf')

            log.info('...done')

        if ringdensplot:
            log.info('Plot integrated flux density map with ring models...')


            fits_path = dir_path_list[1] + '/maps/DINGO_J224218.10-300326.8_0mom.fits'

            kvalidation.plot_3Dbarolo_fits_map(fits_path = fits_path,
                rot_dir = dir_path_list[1],
                profile_file_name = pv_profile_file_name_list[0],
                output_fname = output_dir + '{0:d}km_ringdensplot.pdf'.format(
                        baseline_length),
                N_optical_pixels = N_opt_px)


            log.info('...done')