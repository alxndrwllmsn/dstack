"""These functions are used to compare the kynematics of the sources foud 
by SoFiA in different pipeline setups and modelled in GIPSY.

Therefore, this code is really similar to `sdiagnostics`, but work son the GIPSY
output of a source to compare kinematics models.

The code compare rotation curves, p-v diagrams and density profiles.

We used these diagnostics jointly with the `sdiagnostics` code to validate the
grid stacking pipeline.
"""
#=== Imports ===
import os, sys
import shutil
import numpy as np
import logging
import warnings
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

from astropy.io import fits

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

    This function is copied from `svalidation`, it should rather be imported
    from somwhere! TO DO: put this to sdiagnostics maybe.

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
        
        #Lazy error handeling (does not matter if input list is longer)
        if len(argument_list) > required_list_length:
            warnings.warn('Argument list {} is longer than the number of sources provided!'.format(argument_list))
            return argument_list

    return argument_list

def plot_profile_curves(gipsy_dir_list, profile_file_name_list, output_fname, profile='rotation', label_list=['?'], color_list=[None]):
    """This is a simppe function to plot different models fit on different deep
    imaging methods. The main curves this function can produce:

    - rotation curves
    - dispersion curves
    - density profiles

    NOTE, that GIPSY generates the fit and this code just overplot the output of
    differet runs!

    Parameters
    ==========

    gipsy_dir_list: list of str
        The full path of each GIPSY output which under the fit text files live.

    profile_file_name_list: lost of strings
        The name of the profile file generated by GIPSY. As several fit cycles are possible,
        this can be te first of the final cycle. Also, the density profiles are
        outputted to a different file than the rotation curves. This si the file name
        that will be used for generating the plots!

    output_fname: str
        Name and full path of the images created.

    profile: str, optional
        The type of the profile plot generated. Currently, the following profiles
        are supported: rotation, dispersion and density

    label list, list of strings, optional
        A string for each GIPSY output, that is displayed a s alegend on the plot

    color_list: list of strings, optional
        The color for each GIPSY output on tha plot

    Retun
    =====
    output_image: file
        The image created
 
    """
    #Check the available profiles
    assert profile in ['rotation', 'dispersion', 'density'], 'Invalid profile image requested!'

    N_sources = len(gipsy_dir_list)

    profile_file_name_list = initialise_argument_list(N_sources, profile_file_name_list)
    label_list = initialise_argument_list(N_sources, label_list)

    #Generate random colors if needed
    color_list = initialise_argument_list(N_sources, color_list)
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color
    

    #Create plot
    fig = plt.figure(1, figsize=(8,5))
    ax = fig.add_subplot(111)

    lines = []

    for i in range(0,N_sources):
        profilefit_file_path = gipsy_dir_list[i] + profile_file_name_list[i]


        if profile in ['rotation', 'dispersion']:
            #The rotation curve
            rad, vrot, srot = np.genfromtxt(profilefit_file_path, skip_header=1,
                                    usecols=(1,2,3), unpack=True) 

            #compute the upper and lower errors
            verr_l, verr_h, serr_l, serr_h = np.genfromtxt(profilefit_file_path,
                    skip_header=1, usecols=(13,14,15,16), unpack=True)

            if profile == 'rotation':   
                lines.append(ax.errorbar(rad, vrot, yerr=[verr_l, -verr_h], fmt='o', capsize=2,
                    elinewidth=1.5, markersize=7.5, color=color_list[i], alpha=0.75,
                    label='{0:s}'.format(label_list[i])))
            elif profile == 'dispersion':
                lines.append(ax.errorbar(rad, srot, yerr=[serr_l, -serr_h], fmt='o', capsize=2,
                    elinewidth=1.5, markersize=7.5, color=color_list[i], alpha=0.75,
                    label='{0:s}'.format(label_list[i])))

        else:
            rad_sd, surfdens, sd_err = np.genfromtxt(profilefit_file_path,
                    usecols=(0,3,4), unpack=True)

            lines.append(ax.errorbar(rad_sd, surfdens, yerr=sd_err, fmt='o', capsize=2,
                    elinewidth=1.5, markersize=7.5, color=color_list[i], alpha=0.75,
                    label='{0:s}'.format(label_list[i])))
     
    if profile == 'rotation':
        ax.set_ylabel(r'v$_{rot}$ [km/s]', fontsize=18)
    elif profile == 'dispersion':
        ax.set_ylabel(r'$\sigma_{rot}$ [km/s]', fontsize=18)
    else:
        ax.set_ylabel(r'$\Sigma$ [Jy/pixel $\times$ km/s]', fontsize=18) #Unit depends on the GIPSY file!
    
    ax.set_xlabel('Radius [arcsec]', fontsize=18)
    ax.grid()

    if profile == 'rotation':
        #Add legend
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right', fontsize=16)

        #Add inner title
        t = ds.sdiagnostics.add_inner_title(ax, 'Rotation curves', loc=2, prop=dict(size=18))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
    elif profile == 'dispersion':
        #Add legend
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left', fontsize=16)

        #Add inner title
        t = ds.sdiagnostics.add_inner_title(ax, 'Dispersion curves', loc=1, prop=dict(size=18))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    else:
        #Add legend
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left', fontsize=16)

        #Add inner title
        t = ds.sdiagnostics.add_inner_title(ax, 'Density curves', loc=1, prop=dict(size=18))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    plt.savefig(output_fname,bbox_inches='tight')
    plt.close()


def plot_pv_diagram_triangle_plot(gipsy_dir_list, pv_fits_name_base_list, profile_file_name_list, output_fname, color_list=[None], label_list=['?'] ):
    """

    """
    N_sources = len(gipsy_dir_list)

    profile_file_name_list = initialise_argument_list(N_sources, profile_file_name_list)
    label_list = initialise_argument_list(N_sources, label_list)

    #Generate random colors if needed
    color_list = initialise_argument_list(N_sources, color_list)
    for i in range(0,N_sources):
        if color_list[i] == None:
            color_list[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF)) #Generate random HEX color
 
    #NOTE only creating the plot along the major axis of the galaxy!i

    #Get the data arrays and crop them to the same size 
    xy_min = [np.inf, np.inf]
    edge_size = 10

    for i in range(0,N_sources):
        model_fits = gipsy_dir_list[i] + 'pvs/' + pv_fits_name_base_list[i] + 'mod_pv_a_local.fits'

        image_maj = fits.open(model_fits)[0].data
    
        for j, length in zip(range(0,2), np.shape(image_maj)):
            if length < xy_min[j]:
                xy_min[j] = length

    common_size = np.subtract(np.array(xy_min),edge_size)

    #Get the PV diagram axis settings
    vsys_list = []

    for i in range(0,N_sources):
        profilefit_file_path = gipsy_dir_list[i] + profile_file_name_list[i]

        sys_vel = np.genfromtxt(profilefit_file_path,skip_header=1,usecols=(11),unpack=True)

        for vs in sys_vel:
            vsys_list.append(vs)


    #Get the mean systemnatics velocity
    # NOTE this is computed across rthe different input images, and so
    # the systematics velocity of the different methods should not differ mutch!

    vsys = np.nanmean(np.array(vsys_list))

    print(vsys)


    #Get the coordinates based on the GIPSY code, butask Tristan for why some
    #parameters are hardcoded...


    exit()





    #head = [image_maj[0].header]

    #Get pixel values
    #crpixpv = head[0]['CRPIX1']
    #cdeltpv = head[0]['CDELT1']
    #crvalpv = head[0]['CRVAL1']
    #xminpv, xmaxpv = np.floor(crpixpv - 1 - 50), np.ceil(crpixpv - 1 + 50)
    
    #if xminpv < 0: xminpv = 0
    #if xmaxpv >= head[0]['NAXIS1']: xmaxpv = head[0]['NAXIS1']-1


    #Get the data array
    #data_maj = image_maj[0].data[17:85,int(xminpv):int(xmaxpv)+1]
    #data_maj = image_maj[0].data

    cont = 0.00392164
    v = np.array([1,2,4,8,16,32,64])*cont

    #=== Create the figure
    fig, axes = plt.subplots(figsize=(2 + 4 * N_sources, 2 + 4 * N_sources),
                sharex=True, sharey=True, ncols=N_sources, nrows=N_sources)


    for i in range(0,N_sources):
        for j in range(0,N_sources):
            if i<j:
                axes[i,j].set_axis_off() #This does not work with projection
               
            else:
                if i == j:
                    data_fits = gipsy_dir_list[i] + 'pvs/' + pv_fits_name_base_list[i] + '_pv_a.fits'
                    model_fits = gipsy_dir_list[i] + 'pvs/' + pv_fits_name_base_list[i] + 'mod_pv_a_local.fits'
                    uncut_model_maj = fits.open(model_fits)[0].data

                    dsize = np.shape(uncut_model_maj)
                    x_lim = (dsize[0] - common_size[0]) / 2
                    y_lim = (dsize[1] - common_size[1]) / 2

                    model_maj = uncut_model_maj[int(np.floor(x_lim)):int(-np.ceil(x_lim)),
                                            int(np.floor(y_lim)):int(-np.ceil(y_lim))]


                    data_maj = fits.open(data_fits)[0].data[int(np.floor(x_lim)):int(-np.ceil(x_lim)),
                                            int(np.floor(y_lim)):int(-np.ceil(y_lim))]

                    axes[i,j].contour(data_maj,v,origin='lower',
                                linewidths=1.5, colors=outlier_color)
                    
                    axes[i,j].contour(model_maj,v,origin='lower', alpha=0.75,
                                linewidths=1.5,colors=color_list[i])

                if i != j:
                    #Difference maps
                    pass
            
            if i == (N_sources - 1):
                axes[i,j].set_xlabel('Offset [arcsec]', fontsize=18)
                axes[i,j].tick_params(axis='x', length=matplotlib.rcParams['xtick.major.size'])

            if j == 0:
                axes[i,j].set_ylabel(r'$\Delta v_{los}}$ [km/s]', fontsize=18)
                axes[i,j].tick_params(axis='y', length=matplotlib.rcParams['xtick.major.size'])

    #Some style settings
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    plt.savefig(output_fname, bbox_inches='tight')
    plt.close()




#=== MAIN ===
if __name__ == "__main__":
    #pass
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    #PV diagram triangle plot
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_rotation_curve_resuls/'

    gipsy_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/']))

    plot_pv_diagram_triangle_plot(gipsy_dir_list = gipsy_dir_path_list,
            profile_file_name_list = ['ringlog2.txt'],
            pv_fits_name_base_list = ['DINGO_J224218.10-300330.0', 'DINGO_J224218.09-300329.7', 'DINGO_J224218.04-300329.9'],
            color_list = [c0, c2, c1],
            output_fname = working_dir + 'validation/pv_diagram_triangle_plot.pdf')



    exit()

    #Rotattion different profile curves
    working_dir = '/home/krozgonyi/Desktop/quick_and_dirty_rotation_curve_resuls/'

    gipsy_dir_path_list = list(map(working_dir.__add__,['co_added_visibilities/',
        'stacked_grids/', 'stacked_images/']))

    profile_list = ['rotation', 'dispersion', 'density']
    profile_file_names = [['ringlog2.txt'], ['ringlog2.txt'], ['densprof.txt']]

    for i in range(0,3):
        log.info('Creating {0:s} curves plot...'.format(profile_list[i]))

        plot_profile_curves(gipsy_dir_list = gipsy_dir_path_list,
            profile_file_name_list = profile_file_names[i],
            profile = profile_list[i],
            label_list = ['co added visibilities', 'stacked grids', 'stacked images'],
            color_list = [c0,c2,c1],
            output_fname = working_dir + 'validation/{0:s}_curves.pdf'.format(
            profile_list[i]))

        log.info('...done')


