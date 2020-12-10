"""Functions to plot spectra
"""
#=== Imports ===
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import logging

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

#=== Functions ===
def get_RMS_and_spectral_array_from_CIM(cim_path,output_spectra_name,all_dim=False,chan=None,chan_max=None):
    """Create a spectra txt file containing channel frequencies and the measured RMS of the corresponding channel
    of the input Cim.

    This code is builds upon existing `dstack` routines and creates two arrays, which it saves to a file:

    rms_arr: numpy.ndarray
        RMS values for ecah channel
    spectral_arr: numpy.ndarray
        The frequency values for each array

    Note, that the polarisation axis index is by default set to 0!


    Parameters
    ==========
    cim_path: str
        The path to the CASAimage

    output_spectra_name: str
        Full path and name to the output spectra file created

    all_dim: bool, optional
        If True, the RMS will be computed for all channels and polarizations in the image cube
        Note that, this can be **very slow** even when parallelised!
        If set to True the `chan_max` param will be ignored!

    chan: int, optional
        Index of the channel in the image cube

    chan_max: int, optional
        Index of the upper channel limit if the RMS is computed only on subset of the imagechannels.
        Has to be larger than the :chan: variable, and smaéller than the number of channels.
        Can be negative using python indexing scheme, but it can result in an indexing error if specified poorly!

    Return
    ======
    spectra_name: file
        Contains two columns: channel and RMS. The header informs the user of the units.
    """
    rms_arr, rms_unit = ds.cim.measure_CIM_RMS(cim_path,all_dim=all_dim,chan=chan,chan_max=chan_max,return_dim=True)
    
    #Set teh full channel range for the spectral axis array
    #if all_dim == True:     
    #    chan = 0
    #    chan_max = -1

    spectral_arr, spectral_unit = ds.cim.get_CIM_spectral_axis_array(cim_path,chan=chan,chan_max=chan_max)

    np.savetxt(output_spectra_name,np.column_stack((spectral_arr,rms_arr)), header='Freq [{0:s}] RMS [{1:s}]'.format(spectral_unit,rms_unit))


def plot_spectra_list(input_spectra_file_list, beam_size_list, flux_scaling_list, boundary_list, color_list, label_list, output_name):
    """This is a one-fit-for-all type spectra plotting function to quickly compare spectras.

    The input files should be 2 columns:
        - column containing spectral window avalues
        - column containing slux values

    The output is a S[Jy] -- \nu [GHz] plot, so the data might need to be converted to the proper dimensions.

    If the measured flux density unit is geven in Jy/beam a correction for the beam is needed,
    which for a Gaussian beam is a division by the beam area:

    pi * a * b / (4 * ln(2))

    were a and b are the minor and major axis in pixels.

    However, the current code only works with a circula-Gaussian beam, here a = b !

    Furthermore, if the uit is mJy for example a scaling for Jy is needed.

    Some level of styling is possible and currently not optional but mandatory.

    TO DO:
        - add frequency scaling posibility

    Parameters
    ==========
    input_spectra_file_list: list
        List of paths for the spectra files

    beam_size_list: list
        List containing the (average) beam axis size in pixels. If no beam correction needed, set it to `None`

    flux_scaling_list: list
        Flux scaling values if the flux unit is not given in Jy. Else set to 1.

    boundary_list: list
        Indices of the lower and upper boundaries of the spectral arrays to be shown. 
        Values has to be touples containing both indices. If the full array should be shown set it to (0,-1)!

    color_list: list
        Color list of the corresponding spectra

    label_list: list
        Lables of the spectras

    output_name: str
        Output name of the plot created.

    Return
    ======
    Spectra plot
    """

    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(111)

    #Loop through spectra list
    for i in range(0,len(input_spectra_file_list)):
        spectra_data = np.genfromtxt(input_spectra_file_list[i])

        #Beam correction from Jy=beam to Jy
        if beam_size_list[i] != None:
            spectra_data[:,1] /=  np.pi * beam_size_list[i] * beam_size_list[i] / (4 * np.log(2))

        ax1.step(spectra_data[boundary_list[i][0]:boundary_list[i][1],0],
                spectra_data[boundary_list[i][0]:boundary_list[i][1],1] * flux_scaling_list[i],
                lw=3, c=color_list[i], label=label_list[i], alpha=1)

    ax1.legend(fontsize=16,loc='best')

    ax1.set_xlabel(r'$\mathcal{V}$ [GHz]', fontsize=18)
    ax1.set_ylabel(r'S [Jy]', fontsize=18)

    plt.savefig(output_name,bbox_inches='tight')


def plot_spectra_list_diff_triangle_matrix(input_spectra_file_list, beam_size_list, flux_scaling_list, boundary_list, color_list, label_list, output_name):
    """

    """

    fig, axes = plt.subplots(figsize=(2+4*len(input_spectra_file_list),2+4*len(input_spectra_file_list)),
                            sharex=True, sharey=True,
                            ncols=len(input_spectra_file_list), nrows=len(input_spectra_file_list))

    for i in range(0,len(input_spectra_file_list)):
        for j in range(0,len(input_spectra_file_list)):
            if i<j:
                axes[i, j].axis('off')
            else:
                spectra_data = np.genfromtxt(input_spectra_file_list[i])
                #Beam correction from Jy=beam to Jy
                if beam_size_list[i] != None:
                    spectra_data[:,1] /=  np.pi * beam_size_list[i] * beam_size_list[i] / (4 * np.log(2))

                axes[i, j].grid()

                if i != j:
                    second_spectra_data = np.genfromtxt(input_spectra_file_list[j])
                    #Beam correction from Jy=beam to Jy
                    if beam_size_list[j] != None:
                        second_spectra_data[:,1] /=  np.pi * beam_size_list[j] * beam_size_list[j] / (4 * np.log(2))


                    axes[i, j].step(second_spectra_data[boundary_list[j][0]:boundary_list[j][1],0],
                            second_spectra_data[boundary_list[j][0]:boundary_list[j][1],1] * flux_scaling_list[j],
                            lw=3, c=color_list[j], label=label_list[j], alpha=1)
                else:
                    #axes[i, j].legend(fontsize=16,loc='best')
                    axes[i, j].set_title(label_list[i],fontsize=18)


                #Plot first spectra
                axes[i, j].step(spectra_data[boundary_list[i][0]:boundary_list[i][1],0],
                    spectra_data[boundary_list[i][0]:boundary_list[i][1],1] * flux_scaling_list[i],
                    lw=3, c=color_list[i], label=label_list[i], alpha=1)

                if i == len(input_spectra_file_list)-1:
                    axes[i, j].set_xlabel(r'$\mathcal{V}$ [GHz]', fontsize=18)

                if j == 0:
                    axes[i, j].set_ylabel(r'S [Jy]', fontsize=18)

    #Some style settings
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    plt.savefig(output_name,bbox_inches='tight')


#=== MAIN ===
if __name__ == "__main__":
    #Examples

    plot_spectra_list_diff_triangle_matrix(['/home/krozgonyi/Desktop/beam17_results/spectras/co_added_visibility_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/stacked_image_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/stacked_grid_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/traditional_combined_nights_spectra.txt'],
                        [5,5,5,5],
                        [1,1,1,1000],
                        #[1,1.038,1,1000],
                        [(58,158),(58,158),(58,158),(6377,6477)],
                        #[(0,-1),(0,-1),(0,-1),(6377-58,6477+58)],
                        [c0,c1,c2,outlier_color],
                        ['Co added visibilities',
                        'Stacked images',
                        'Stacked grids',
                        'Conventional imaging'],
                        '/home/krozgonyi/Desktop/beam17_results/spectras/all_comparision_matrix.pdf')

    exit()

    plot_spectra_list(['/home/krozgonyi/Desktop/beam17_results/spectras/rms_co_added_visibility_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_image_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_grid_spectra.txt'],
                        [1,1,1],
                        [1000/np.sqrt(7),1000,1000],
                        #[1000/np.sqrt(7),1.038*1000,1000],
                        #[1000,1000,1000],
                        [(0,-1),(0,-1),(0,-1)],
                        [c0,c1,c2],
                        ['Co added visibilities',
                        'Stacked images',
                        'Stacked grids'],
                        '/home/krozgonyi/Desktop/beam17_results/spectras/all_comparision.pdf')

    exit()


    #get_RMS_and_spectral_array_from_CIM('/home/krozgonyi/Desktop/beam17_results/noise_cube/co_added_visibilities/image.deep.restored',
    #                                    '/home/krozgonyi/Desktop/beam17_results/noise_cube/rms_spectra.txt',
    #                                    all_dim=True)

    get_RMS_and_spectral_array_from_CIM('/home/krozgonyi/Desktop/beam17_results/co_added_visibilities/image.deep.restored',
                                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_co_added_visibility_spectra.txt',
                                        all_dim=False, chan=0, chan_max=70)

    get_RMS_and_spectral_array_from_CIM('/home/krozgonyi/Desktop/beam17_results/stacked_grids/image.deep.restored',
                                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_grid_spectra.txt',
                                        all_dim=False, chan=0, chan_max=70)

    get_RMS_and_spectral_array_from_CIM('/home/krozgonyi/Desktop/beam17_results/stacked_images/image.restored.deep',
                                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_image_spectra.txt',
                                        all_dim=False, chan=0, chan_max=70)

    #exit()


    
    plot_spectra_list(['/home/krozgonyi/Desktop/beam17_results/spectras/co_added_visibility_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/stacked_image_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/stacked_grid_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/traditional_combined_nights_spectra.txt'],
                        [5,5,5,5],
                        [1,1,1,1000],
                        [(58,158),(58,158),(58,158),(6377,6477)],
                        [c0,c1,c2,c3],
                        ['Co added visibilities',
                        'Stacked images',
                        'Stacked grids',
                        'Conventional imaging'],
                        '/home/krozgonyi/Desktop/beam17_results/spectras/all_comparision.pdf')