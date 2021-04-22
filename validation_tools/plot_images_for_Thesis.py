"""This script is a collection of imaging function calls of svalidation and
sdiagnostics functions for my Thesis. The imaging scripts can be enabled by
switches in the beginning of the main section of the code. Not only the SoFiA output
imaging scripts are called here, but the 3Dbarollo output (kynematic modelling
results) as well. The following images can be generated:
    
    SoFiA outout:

    - [] RMS -- channel plot for all methods from the dirty maps
    - [] RMS -- channel plot for all methods from the residual maps
    - [] nice mom0 and mom1 maps with column density contours (only grid stacking)
    - [] nice spectra plot (only grid stacking)
    - [x] spectras triangle plot
    - [x] mom0 triangle plot with 3and 5sigma contours and 1 sigma cut
    - [x] mom1 triangle plot with the mom0 contours and cut
    - [] measured -- column density flux density plots for all comparisons
    - [] N_px -- log(N_HI) histogram for all deep imaging method

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

#============
#=== MAIN ===
#============
if __name__ == "__main__":
    #pass
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    #=== Define what to plot ===

    #Decide on Wiener_filtering
    filtering = False

    #Decide the resolution
    full_res = False #If True the 6km baseline results are plotted

    #Decide if kinematics plots are created
    kinematics = False

    #Decide on individual figures to make
    spectra_triangle_plot = True
    mom0_triangle_plot = True
    mom1_triangle_plot = True

    #Kinematics
    profile_curves = False
    pv_data_trinagle_plot = True
    pv_model_trinagle_plot = True
    pv_residual_trinagle_plot = True


    #=== Setup variables ===

    if not full_res:
        baseline_length = int(2)

        if not kinematics:
            #Define environment variables
            if filtering:
                working_dir = '/home/krozgonyi/Desktop/NGC7361_results/SoFiA/2km_baseline_results/'

                output_dir = working_dir + 'validation/'

                #sofia_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
                #    'stacked_grids/', 'stacked_images/']))

                sofia_dir_path_list = list(map(working_dir.__add__,['baseline_vis_imaging/',
                    'co_added_visibilities/', 'stacked_grids/', 'stacked_images/',
                    'conventional_imaging/']))

                #Define source and imaging parameters

                #NOTE when only one element is given the imaging code automatically appends
                # it to the required size, i.e. all deep imaging has the same parameter

                #source_ID_list = [4, 4, 3]
                source_ID_list = [6, 4, 4, 3, 3]
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
                working_dir = '/home/krozgonyi/Desktop/NGC7361_results/\
SoFiA/no_Wiener_filtering_2km_baseline_results/'

                output_dir = working_dir + 'validation/'

                sofia_dir_path_list = list(map(working_dir.__add__,[
                    'co_added_visibilities/', 'stacked_grids/', 'stacked_images/']))

                #Define source and imaging parameters

                #NOTE when only one element is given the imaging code automatically appends
                # it to the required size, i.e. all deep imaging has the same parameter

                source_ID_list = [7, 3, 1]
                beam_correction_list = [True]
                color_list = [c0, c2, c1]
                label_list = ['visibilities', 'stacked grids', 'stacked images']
                ident_list = ['V', 'G', 'I']
            
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
            contour_levels = [8., 16.]

            #Single valued parameters
            N_opt_px = 130 #Number of optical background pixels in pixels ???
            mom_triangle_contours = True
            diff_saturation = 24.

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
            source_ID_list = [4, 3, 4]
            name_base_list = ['beam17_all_']
            beam_correction_list = [True, True, True]
            b_maj_px_list = [6.]
            b_min_px_list = [6.]
            b_maj_list = [12.]
            b_min_list = [12.]
            b_pa_list = [0.]
            masking_list = [True]
            mask_sigma_list = [3.]
            color_list = [c0, c2, c1]
            label_list = ['visibilities', 'stacked grids', 'stacked images']
            ident_list = ['V', 'G', 'I']
            contour_levels = [16.]

            #Single valued parameters
            N_opt_px = 420 #Number of optical background pixels in pixels ???
            mom_triangle_contours = True
            diff_saturation = 24.


        else:
            #kinematics plots:
            working_dir = '/home/krozgonyi/Desktop/NGC7361_results/3Dbarolo/6km_baseline_results/'

            output_dir = working_dir + 'validation/'

            dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
                'stacked_grids/', 'stacked_images/']))

            #List parameters
            profile_file_name_list = ['densprof.txt']
            pv_profile_file_name_list = ['rings_final2.txt']
            pv_fits_name_base_list = ['DINGO_J224218.04-300319.2',
                'DINGO_J224218.05-300319.0', 'DINGO_J224219.24-300310.2']

            contour_levels=[1,2,4,8,16,32,64]
            color_list = [c0, c2, c1]
            label_list = ['co-added visibilities',
                        'stacked grids', 'stacked images']
            ident_list = ['B', 'V', 'G', 'I']
            S_rms_list = [0.0044004, 0.00439989, 0.00443053]

            #Single valued parameters
            channelwidth = 4.
            centre_index = 75
            edge_crop = 3
            ring_crop = 6


    #===========================================================================
    #=== Imaging ===
    if not kinematics:
        if spectra_triangle_plot:
            log.info('Creating spectra triangle plot for {0:d}km \
baseline results...'.format(baseline_length))

            svalidation.plot_spectra_triangle_matrix(source_ID_list = source_ID_list,
                sofia_dir_list = sofia_dir_path_list,
                name_base_list = name_base_list,
                output_name = output_dir + 'spectra_triangle.pdf',
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
                output_name = output_dir + 'mom0_map_triangle.pdf',
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
                output_name = output_dir + 'mom1_map_triangle.pdf',
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
                    color_list = color_list,
                    label_list = label_list,
                    output_fname = output_dir + '{0:s}_curves.pdf'.format(profile))

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
                S_rms_list = S_rms_list,
                contour_levels = contour_levels,
                color_list = color_list,
                label_list = label_list,
                ident_list = ident_list,
                output_fname = output_dir + 'pv_diagram_residual_triangle_plot.pdf')

            log.info('...done')