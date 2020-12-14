"""Functions to plot spectra
"""
#=== Imports ===
import numpy as np
import sys, os
import random

import matplotlib
import matplotlib.pyplot as plt

import configparser
import logging

import dstack as ds

import ast #For literal evaluation

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


def plot_spectra_list(input_spectra_file_list, output_name, label_list, beam_size_list=[None], flux_scaling_list=[1], frequency_scaling_list=[1], boundary_list=[(0,-1)], color_list=[None], flux_label=r'S [Jy]', frequency_label=r'$\mathcal{V}$ [GHz]'):
    """This is a one-fit-for-all type spectra plotting function to quickly compare spectras. The spectras are plotted onto the same plot.

    The input files should be 2 columns:
        - column containing spectral window avalues
        - column containing slux values

    The output is a S[Jy] -- \nu [GHz] plot by default, so the data might need to be converted to the proper dimensions or
    the user has to change the labels accordingly! This is possible using arguments. Note, that all spectras needs to be on the same scale!

    If the measured flux density unit is geven in Jy/beam a correction for the beam is needed,
    which for a Gaussian beam is a division by the beam area:

    pi * a * b / (4 * ln(2))

    were a and b are the minor and major axis in pixels.

    However, the current code only works with a circula-Gaussian beam, here a = b !

    Note, that the default argument parameters are recursively appended with the last value, that can be a surce of weird behaviour of this function!
    This can be used to help the user: if all spectra needs the same correction for example it is adequate to provide a list conatining a single value for the correction!

    Parameters
    ==========
    input_spectra_file_list: list
        List of paths for the spectra files

    output_name: str
        Output name of the plot created.

    label_list: list
        The list containing of the names of all the input spectra.

    beam_size_list: list, optional
        List containing the (average) beam axis size in pixels. If no beam correction needed, set it to `None`, which is the default setting.

    flux_scaling_list: list, optional
        Flux scaling values if the flux unit is not given in Jy, or the user wants a different unit. Else set to 1, which is the default value..

    frequency_scaling_list: list, optional
        Scaling factors for the frequency axes if the input frequencies are not in GHz, or the user wants a different unit. Else set to 1, which is the default value.

    boundary_list: list, optional
        Indices of the lower and upper boundaries of the spectral arrays to be shown. 
        Values has to be touples containing both indices. If the full array should be shown set it to (0,-1), which is the default.

    color_list: list, optional
        Color list of the corresponding spectra. By default random colours are generated!

    flux_label_list: str, optional
        Common flux label of the spectras (y axis). [Jy] by default.

    frequency_label_list: str, optional
        Common frequency lables of the spectras (x axis). [GHz] by default.

    Return
    ======
    Spectra plot svaed as `output_name`
    """
    def initialise_argument_list(required_list_length,argument_list):
        """Child-function recursively appending the argument lists if needed.

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

    #Initialise arguments by recursively appending them
    N_spectras = len(input_spectra_file_list)
    beam_size_list = initialise_argument_list(N_spectras,beam_size_list)
    flux_scaling_list = initialise_argument_list(N_spectras,flux_scaling_list)
    frequency_scaling_list = initialise_argument_list(N_spectras,frequency_scaling_list)
    boundary_list = initialise_argument_list(N_spectras,boundary_list)
    color_list = initialise_argument_list(N_spectras,color_list)

    #Generate random colors if needed
    for i in range(0,len(color_list)):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color

    #=== The plot
    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(111)

    #Loop through spectra list
    for i in range(0,len(input_spectra_file_list)):
        spectra_data = np.genfromtxt(input_spectra_file_list[i])

        #Beam correction from Jy=beam to Jy
        if beam_size_list[i] != None:
            spectra_data[:,1] /=  np.pi * beam_size_list[i] * beam_size_list[i] / (4 * np.log(2))

        ax1.step(spectra_data[boundary_list[i][0]:boundary_list[i][1],0] * frequency_scaling_list[i],
                spectra_data[boundary_list[i][0]:boundary_list[i][1],1] * flux_scaling_list[i],
                lw=3, c=color_list[i], label=label_list[i], alpha=1)

    ax1.legend(fontsize=16,loc='best')

    ax1.set_xlabel(frequency_label, fontsize=18)
    ax1.set_ylabel(flux_label, fontsize=18)

    plt.savefig(output_name,bbox_inches='tight')


def plot_spectra_list_diff_triangle_matrix(input_spectra_file_list, beam_size_list, flux_scaling_list, frequency_scaling_list, boundary_list, color_list, label_list, output_name):
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

def run_spectral_analysis(parset_path, selected_sections):
    """This function is a wrapper around all functions defined in this script to run them usin parameters et files by using ConfigParser.

    The goal is to create command-line applications that can be incorporated to snakemake pipelines and provide some quick analytical plots for quality check.

    However, now they exists as just a separate scripts. This warpper makes sure, all functions can be called from a parameterfile.

    The parameter file has to have the foillowing structure:

    A unique [Section name] for each 'task' the user wants to run. Each section calls one function from this script. The first parameter of the
    section has to be in the following form:
    
    `function = function_name_from_this_script`

    then each parameter of thet function needs to be given as follows:

    `parameter_name = parameter_value_as_a_literal_python_string`

    These parameters are evaluated to the proper python classes using the `ast` module and then passed to the corresponding function.
    It is crucal to provide the parameters in the appropriate form `ast`can evaluate!

    Parameters
    ==========
    parset_path: str
        Path and name of the input parset file, whic stucture is given above.

    selected_sections: list
        List of unique sections which will be evaluated

    Return
    ======
    Run functions given by each selected section with the section's parameters, respectively.
    """
    def convert_section_to_dict(parset_path, single_section):
        """This child-function defines the literal evaluation of the ConfigParser parameters into a dictionary,
        which get passed to the respective function.

        Parameters
        ==========
        parset_path: str
            Path and name of the input parset file.

        single_section: str
            The seleted section to evaluate

        Return
        ======
        params_dict: dict
            A dictionary conatining the arguments for the respective function of the selected section.
        """
        config = configparser.ConfigParser()
        config.read(parset_path)

        log.info('Running section {0:s}'.format(single_section))

        params_dict = dict(config.items(single_section))
        del params_dict['function']

        for k, v in params_dict.items():
            params_dict[k] = ast.literal_eval(params_dict[k])

        return params_dict


    assert os.path.exists(parset_path), 'Spectra analysis parset does not exist!'

    log.info('Using parset {0:s}'.format(parset_path))

    config = configparser.ConfigParser()
    config.read(parset_path)

    #Get sections and loop through the selected sections and call the respective functions
    available_sections = config.sections()

    for sec in selected_sections:
        if sec not in available_sections:
            raise ValueError('Section {0:s} devined by te user is not in the provided input parameter set file!'.format(sec))

        #Check for all functions in thes script and call them!
        if config.get(sec,'function') == 'get_RMS_and_spectral_array_from_CIM':
            get_RMS_and_spectral_array_from_CIM(**convert_section_to_dict(parset_path,sec))
        elif config.get(sec,'function') == 'plot_spectra_list':
            plot_spectra_list(**convert_section_to_dict(parset_path,sec))
        elif config.get(sec,'function') == 'plot_spectra_list_diff_triangle_matrix':
            plot_spectra_list_diff_triangle_matrix(**convert_section_to_dict(parset_path,sec))
        else:
            raise ValueError('The given function {0:s} is not defined!'.format(config.get(sec,'function'))) 

#=== MAIN ===
if __name__ == "__main__":
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))



    run_spectral_analysis('./spectra_quck_and_dirty_analyisis.in',
                            selected_sections=['SimpleSpectraTestPlot'])


    exit()


    #Examples

    plot_spectra_list_diff_triangle_matrix(['/home/krozgonyi/Desktop/beam17_results/spectras/co_added_visibility_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/stacked_image_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/stacked_grid_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/traditional_combined_nights_spectra.txt'],
                        [5,5,5,5],
                        [1,1,1,1000],
                        #[0.0012622857250140181,0.0012197129794263414,0.001262288513992514,1000*0.0012197129794263414],
                        #[1,1.0349,1,1000],
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
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_grid_spectra.txt'],
                        [1,1],
                        [1,1],
                        [(0,-1),(0,-1)],
                        [c0,c2],
                        ['Co added visibilities',
                        'Stacked grids'],
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_vis_grid_comparision.pdf')

    exit()

    plot_spectra_list(['/home/krozgonyi/Desktop/beam17_results/spectras/rms_co_added_visibility_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_image_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_stacked_grid_spectra.txt'],
                        [1,1,1],
                        [1,1,1],
                        #[1000/np.sqrt(7),1.038*1000,1000],
                        #[1000,1000,1000],
                        [(0,-1),(0,-1),(0,-1)],
                        [c0,c1,c2],
                        ['Co added visibilities',
                        'Stacked images',
                        'Stacked grids'],
                        '/home/krozgonyi/Desktop/beam17_results/spectras/rms_all_comparision.pdf')

    exit()




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


    plot_spectra_list_diff_triangle_matrix(['/home/krozgonyi/Desktop/beam17_results/quick_and_dirty_visibility_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/old_tapering/spectras/stacked_image_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/old_tapering/spectras/stacked_grid_spectra.txt',
                        '/home/krozgonyi/Desktop/beam17_results/old_tapering/spectras/traditional_combined_nights_spectra.txt'],
                        [5,5,5,5],
                        [1000,1,1,1000],
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