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

def plot_RMS(rmsfile_list,
            output_fname,
            label_list=['?'],
            color_list=[None],
            rest_frame='optical',
            ptitle=None):
    """Create a plot of RMS -- channel from a list of output .dat files created
    by cimRMS.

    Parameters
    ==========
    rmsfile_list: list of str
        The list containing the files created by cimRMS

    output_fname: str
        Name and full path of the images created.

    label list, list of strings, optional
        A string for each RMS output, that is displayed as a legend on the plot

    color_list: list of strings, optional
        The color for each RMS output on the plot

    rest_frame: str, optional
        The rest frame in whic the x axis is displayed valid are optical and
        frequency. The latter is expect to be provided in [Hz] and the code
        automatically converts it to [GHz]

    Return
    ======
    output_image: file
        The image created

    """
    if rest_frame not in ['optical', 'frequency']:
        raise ValueError('Invalid rest frame for spectral axis is given! \
Only optical and frequency are supported.')

    N_sources = len(rmsfile_list)

    label_list = svalidation.initialise_argument_list(N_sources, label_list)

    #Generate random colors if needed
    color_list = svalidation.initialise_argument_list(N_sources, color_list)
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color

    #Create plot
    fig = plt.figure(1, figsize=(8,5))
    ax = fig.add_subplot(111)

    if ptitle is not None:
        plt.title('{0:s}'.format(ptitle), fontsize=21, loc='left')

    lines = []

    for i in range(0,N_sources):
        freq, rms = np.genfromtxt(rmsfile_list[i], skip_header=4, delimiter=',',
                    usecols=(0,1), unpack=True)

        #Convert RMS from Jy/beam to mJy/beam
        rms = np.multiply(rms,1000)

        if rest_frame == 'frequency':
            #Convert frequency from Hz to GHz
            freq = np.multiply(freq,1e-9)

        else:
            #Convert frequncy from Hz to km/s
            freq = np.array([ds.sdiagnostics.get_velocity_from_freq(i) for i in freq])


        lines.append(plt.step(freq, rms, color=color_list[i], lw=2.5, alpha=0.8,
            label='{0:s}'.format(label_list[i]))[0])

        #print(lines[0][0].get_label())
        #print(type(lines[0][0]))

    if rest_frame == 'optical':
        ax.set_xlabel(r'V$_{opt}$ [km/s]', fontsize=18)
    else:
        ax.set_xlabel(r'$\nu$ [GHz]', fontsize=18)

    ax.set_ylabel(r'RMS [mJy/beam]', fontsize=18)

    labels = [l.get_label() for l in lines]
    
    legend0 = ax.legend(lines, labels, loc='center right',
        bbox_to_anchor= (1.05, 1.0), ncol=2, borderaxespad=0, frameon=True,
        fontsize=16, framealpha=1, fancybox=True, labelspacing=0.1,
        handletextpad=0.3, handlelength=1., columnspacing=0.5)

    legend0.get_frame().set_linewidth(2);
    legend0.get_frame().set_edgecolor('black');

    plt.savefig(output_fname,bbox_inches='tight')
    plt.close()

def plot_column_density_histogram(source_ID,
    sofia_dir,
    name_base,
    output_fname,
    masking=True,
    mask_sigma=3.0,
    b_maj=30.,
    b_min=30.,
    b_pa=0.,
    color='black',
    col_den_sensitivity_lim=None,
    conver_from_NHI=True,
    pixelsize=5.,
    inclination=0.):
    """This is a simple function generating the N pixel, column density histogram
    for a single given mom0 map.

    Parameters
    ==========

    source_ID: int
        The SoFiA ID of the source

    sofia_dir: str
        The SoFiA directory path

    name_base: list of str
        The name `output.filename` variable defined in the SoFiA template 
        .par. Basically the base of all file names in the respective SoFiA dir.

    output_name: str
        The name and full path to the output triangle plot generated.

    masking: optional
        If True, the mom0 map will be masked

    mask_sigma: float, optional
        If the mom0 map is masked, pixels below this threshold will be masked.
        The values are given in units of column density sensitivity.

    b_maj: float, optional
        The major axis of the synthesised beam in arcseconds.

    b_min: float, optional
        The minor axis of the synthesised beam in arcseconds.

    b_pa: float, optional
        The position angle of the synthesised beam.

    color: str, optional
        The color of the histogram

    col_den_sensitivity_lim: float, optional
        The column density sensitivity for each SoFiA output provided by
        the user, rather than using the by default value computed from the SoFiA RMS value.
        Given in units of 10^20 HI /cm^2
    
    convert_from_NHI: bool, optional
        If True, the x axis will be converted from N_HI [10^20 1/cm^2] to
        N_HI in [M_sun/pc^2] by using the following formulae:

        :math:

    pixelsize: float, optional
        The pixelsize in arcseconds

    inclination: float, optional
        The inclination  in gedreed (!) used for the inclination correction.

        The correction is cos(inclination)

    Return
    ======
    output_image: file
        The image created
 
    """
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u

    #Get the mom map
    mom_map, map_wcs, map_sen_lim = ds.sdiagnostics.get_momN_ndarray(moment = 0,
                            source_ID = source_ID,
                            sofia_dir_path = sofia_dir,
                            name_base = name_base,
                            masking = masking,
                            mask_sigma = mask_sigma,
                            b_maj = b_maj,
                            b_min = b_min,
                            col_den_sensitivity = col_den_sensitivity_lim)

    #Flatten the array and perform unit conversion if needed
    col_den_array = mom_map.flatten()

    #convert from [10^20 1/cm^2] to [M_sun / pc^2]
    if conver_from_NHI:
        #Get the redshift of the source
        source_index, catalog_path, cubelet_path_dict, spectra_path = \
                    ds.sdiagnostics.get_source_files(source_ID,
                    sofia_dir, name_base)

        freq, z = ds.sdiagnostics.get_freq_and_redshift_from_catalog(catalog_path,
                        source_index)

        #Get the pixel size in pc => we actually dont need this
        """
        #see: https://stackoverflow.com/questions/56279723/how-to-convert-arcsec-to-mpc-using-python-astropy
        cosmo = FlatLambdaCDM(H0=67.7, Om0=0.307) 

        #TO DO: Check the cosmology parameters!!!
        # and add them as optional parameters to the function!

        d_A = cosmo.angular_diameter_distance(z=z) # in [Mpc] !!

        theta = pixelsize*u.arcsec 

        distance_Mpc = (theta * d_A).to(u.Mpc, u.dimensionless_angles())

        px_dist_in_pc = distance_Mpc.to(u.pc).value
        
        print(px_dist_in_pc)
        #"""

        #Now convert the cm^2 to pc^2
        # 1pc^2 equals (3.0857)^2 times 10^36

        area_conv_factor = np.power((3.0857), 2)
        #area_conv_factor = np.power((3.), 2)

        #Convert the mass from number of HI atoms to solar masses
        # The number of HI atoms is given in 10^20 atoms unit
        # 1 M solar is 1.98847 times 10^30 kg
        # and one HI in kg is the HI isotope mass times the atomic mass (inverted)
        # (1.007825 * 1.660539) times 10^-27 kg
        # That is,  under a 1cm^2 area 1 M solar is given by
        # (1.98847 * (1.007825 * 1.660539)) times 10^57 HI atoms
        #

        mass_conversion_factor = 0.1 * (1.98847 * (1.007825 * 1.660539))
        #mass_conversion_factor = 0.1 * (2. * (1. * 1.7))

        #Conversion
        c_factor = mass_conversion_factor * area_conv_factor
        #c_factor = mass_conversion_factor

        col_den_array = np.multiply(col_den_array, c_factor)

        #TO DO check this step!

        #Correction for inclination
        inc_corr = np.cos((np.pi * (inclination / 180)))

        col_den_array = np.multiply(col_den_array, inc_corr)

        #col_den_array[col_den_array == 0] = None

        #col_den_array = np.multiply(col_den_array,0.55)

    import copy

    col_den_array = copy.deepcopy(np.ma.compressed(col_den_array))

    unique, counts = np.unique(col_den_array, return_counts=True)
    #unique, counts = np.unique(np.log10(col_den_array), return_counts=True)

    #print(np.sort(np.array(counts).flatten())[-10:-1])

    result = np.column_stack((unique, counts)) 

    print(np.amax(col_den_array))

    #=== Create the plot ===
    fig = plt.figure(1, figsize=(8,5))
    ax = fig.add_subplot(111)

    #Get the histogram
    if not conver_from_NHI:
        ax.hist(col_den_array,
            bins=np.logspace(np.log10(np.amin(col_den_array)),\
                np.log10(np.amax(col_den_array)), 100),
            histtype='stepfilled', rwidth=0.8,
            linewidth=2, color=color)

        ax.set_xscale("log")

    else:
        ax.hist(np.log10(col_den_array),
            bins=np.linspace(np.amin(np.log10(col_den_array)),\
                np.amax(np.log10(col_den_array)), 100),
            histtype='stepfilled', rwidth=0.8,
            linewidth=2, color=color)

        #ax.set_yscale("log")
    
    ax.set_ylabel(r'N [pixel]', fontsize=18)

    if conver_from_NHI:
        ax.set_xlabel(r'log$\Sigma_{HI}$ [M$_\odot$/pc$^2$]', fontsize=18)
    else:
        ax.set_xlabel(r'N$_{HI}$ [10$^{20}$/cm$^2$]', fontsize=18)

    plt.savefig(output_fname,bbox_inches='tight')
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

    #Decide the resolution
    full_res = False #If True the 6km baseline results are plotted

    #Decide if kinematics plots are created
    kinematics = False

    #Decide on individual figures to make
    rms_plot = False

    col_density_histogram = False

    simple_grid_mom_contour_plots = False
    simple_grid_spectrum_plot = False

    simple_grid_and_hipass_spectrum_plot = False

    simple_grid_and_image_mom_contour_plots = False
    simple_grid_and_image_spectrum_plot = False

    spectra_triangle_plot = False
    mom0_triangle_plot = False
    mom1_triangle_plot = False

    diff_scaling_plots = False

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

                rms_dir = working_dir + 'measured_RMS/'

                rms_file_list = list(map(rms_dir.__add__,[
                    'baseline_imaging_rms.dat', 'co_added_visibilities_rms.dat',
                    'stacked_grids_rms.dat', 'stacked_images_rms.dat']))

                rms_colors = ['black', c0, c2, c1]
                rms_labels = ['B', 'V', 'G', 'I']
                rms_ptitle = 'Wiener-filtering and deconvolution'
                rms_outlabel = 'filtering'

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
                working_dir = '/home/krozgonyi/Desktop/NGC7361_results/\
SoFiA/no_Wiener_filtering_2km_baseline_results/'

                output_dir = working_dir + 'validation/'

                sofia_dir_path_list = list(map(working_dir.__add__,[
                    'co_added_visibilities/', 'stacked_grids/', 'stacked_images/']))

                rms_dir = working_dir + 'measured_RMS/'

                rms_file_list = list(map(rms_dir.__add__,[
                    'co_added_visibilities_rms.dat',
                    'stacked_grids_rms.dat', 'stacked_images_rms.dat']))

                rms_colors = [c0, c2, c1]
                rms_labels = ['V', 'G', 'I']
                rms_ptitle = 'no Wiener-filtering'

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
            contour_levels = [8., 16.]

            N_opt_px = 130 #Number of optical background pixels in pixels ???
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

            central_vel = 1246 #central vel for mom1 map contours [km/s]
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
        if rms_plot:
            log.info('Creating RMS -- channel plot...')


            plot_RMS(rmsfile_list = rms_file_list,
                output_fname = output_dir + 'rms_{0:s}.pdf'.format(rms_outlabel),
                label_list = rms_labels,
                color_list = rms_colors,
                rest_frame = 'optical',
                ptitle = rms_ptitle)

            log.info('...done')

        if col_density_histogram:
            log.info('Creating column density histograms...')

            plot_column_density_histogram(source_ID = source_ID_list[grid_plot_ID],
                sofia_dir = sofia_dir_path_list[grid_plot_ID],
                name_base = name_base_list[0],
                output_fname = output_dir + 'col_den_hist.pdf',
                masking = masking_list[0],
                mask_sigma = mask_sigma_list[0],
                #mask_sigma = 1,
                b_maj = b_maj_list[0],
                b_min = b_maj_list[0],
                b_pa = b_pa_list[0],
                color = color_list[grid_plot_ID],
                col_den_sensitivity_lim = None,
                conver_from_NHI=True,
                pixelsize=30.,
                inclination=70)

            log.info('...done')

        if simple_grid_mom_contour_plots:
            log.info('Creating single mom0 and mom1 contour maps...')

            source_ID = source_ID_list[grid_plot_ID]
            sofia_dir_path = sofia_dir_path_list[grid_plot_ID]
            name_base = name_base_list[0]

            output_name = output_dir + 'grid_contour_plots.pdf'

            svalidation.simple_moment0_and_moment1_contour_plot(source_ID_list = [source_ID],
                sofia_dir_path_list = [sofia_dir_path],
                name_base_list = [name_base],
                output_name = output_name,
                b_maj_list = b_maj_list,
                b_min_list = b_min_list,
                b_pa_list = b_pa_list,
                N_optical_pixels = N_optical_pixels,
                sigma_mom0_contours = True,
                mom0_contour_levels = mom0_contour_levels,
                central_vel = central_vel,
                delta_vel = delta_vel,
                N_half_contours_mom1 = 7,
                color_list = [color_list[grid_plot_ID]],
                masking_list = masking_list,
                mask_sigma_list = mask_sigma_list)

            log.info('...done')

        if simple_grid_spectrum_plot:
            log.info('Create single spectra plot...')

            output_name = output_dir + 'grid_spectra.pdf'

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

            output_name = output_dir + 'grid_and_HIPASS_spectra.pdf'

            svalidation.simple_spectra_plot(source_ID_list = [None, source_ID_list[grid_plot_ID]],
                sofia_dir_path_list = [None, sofia_dir_path_list[grid_plot_ID]],
                name_base_list = [name_base_list[0]],
                output_name = output_name,
                beam_correction_list = [False, True],
                color_list = ['lightgrey', color_list[grid_plot_ID]],
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

            output_name = output_dir + 'GI_contour_plots.pdf'

            svalidation.simple_moment0_and_moment1_contour_plot(source_ID_list = source_ID,
                sofia_dir_path_list = sofia_dir_path,
                name_base_list = [name_base],
                output_name = output_name,
                b_maj_list = b_maj_list,
                b_min_list = b_min_list,
                b_pa_list = b_pa_list,
                N_optical_pixels = N_optical_pixels,
                sigma_mom0_contours = False,
                mom0_contour_levels = mom0_contour_levels,
                central_vel = central_vel,
                delta_vel = delta_vel,
                N_half_contours_mom1 = 7,
                color_list = [color_list[grid_plot_ID], color_list[image_plot_ID]],
                masking_list = masking_list,
                mask_sigma_list = mask_sigma_list)

            log.info('...done')

        if simple_grid_and_image_spectrum_plot:
            log.info('Create single spectra plot...')

            output_name = output_dir + 'GI_spectra.pdf'

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
        
                    col_den_binwidth = 20
                    col_den_lim = None                        

                    for orientation, orientation_dependence in zip(['col_den', 'RA', 'Dec'], [False, True, True]):
                        log.info('Creating flux diff against {0:s} plots for {1:}...'.format(
                        orientation, diff_ident))

                        svalidation.plot_flux_density_diff_dependience_on_column_density(source_ID_list=[source_ID_list[i],source_ID_list[j]],
                            sofia_dir_list = [sofia_dir_path_list[i], sofia_dir_path_list[j]],
                            name_base_list = name_base_list,
                            output_fname = output_dir + 'scaling_plots/sensitivity_column_density_{0:s}_{1:s}{2:s}_map.pdf'.format(
                                orientation, ident_list[i], ident_list[j]),
                            N_optical_pixels = N_opt_px,
                            masking_list = masking_list,
                            mask_sigma_list = mask_sigma_list,
                            b_maj_list = b_maj_list,
                            b_min_list = b_min_list,
                            b_pa_list = b_pa_list,
                            ident_string = diff_ident,
                            col_den_sensitivity_lim_list = [sen_lim],
                            beam_correction = beam_correction_list[0],
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