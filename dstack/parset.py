"""
This module is a wrapper around parsets used in YandaSoft.
The code takes template parset(s) and creates custom parsets for pipelines such as the gridded DINGO pipeline.
Currently not everything is supported and the wrapper is not complete, i.e. the user can break the wrapper if not careful!
"""

__all__ = ['check_preconditioner_suppoort', 'create_parset_mapping', 'Parset',
            'check_parameter_and_Imager_compatibility', 'check_parameter_and_Preconditioner_compatibility']

import os
import warnings

#A global variable defining the ``YandaSoft`` imagers supported for the template parset
#the dstack is an extra option to create custom templates
_SUPPORTED_IMAGERS = ['Cimager', 'Cdeconvolver', 'dstack']
_SUPPORTED_SOLVERS = ['Clean']
_SUPPORTED_GRIDDER_NAMES = ['Box', 'SphFunc', 'WProject', 'WStack']
_SUPPORTED_PRECONDITIONERS = ['Wiener','GaussianTaper']

#Some default parameters
_DEFAULT_IMAGER = 'Cimager'
_DEFAULT_IMAGE_NAMES = 'image.dstack.test'
_DEFAULT_GRIDDER_NAME = 'WProject'
_DEFAULT_PRECONDITIONER = []

#TO DO:
#======
"""
Create a _preconditioners attribute to the Parset class.
Initialise as an empty list and fill up from the template parset
Or have the user-defined list which is cross-matched against the template parset
Make a function where the user can modify this value
When saving a parset only use the lines compatible with the preonditioners defined

Create a function that prints out what is supported
"""

#=== Functions ===
def check_preconditioner_suppoort(preconditioners=_DEFAULT_PRECONDITIONER):
    """Check if the list of preconditioners given is supported

    Parameters
    ==========
    preconditioners: list
        A list containing the preconditioners to test

    Returns
    =======
    allowded: bool
        True, if the given list contains only allowed preconditioners,
        and False if not
    """
    if preconditioners == []:
        return True
    else:
        for preconditioner in preconditioners:
            if preconditioner not in _SUPPORTED_PRECONDITIONERS:
                return False
    return True

def create_parset_mapping(image_names=_DEFAULT_IMAGE_NAMES, gridder_name=_DEFAULT_GRIDDER_NAME):
    """This function creates a mapping between the ``dstack`` and ``YandaSoft`` parset variables.
    The imager used is not included in the ``YandaSoft`` variables. However, when reading in a template parset
    using the ``__init__()`` method of the Parset class, the imager name have to be included in the parset file.
    This solution gives more flexibility with reading and creating parsets. The list of supported imagers:

    - Cimager
    - Cdeconvolver

    And for reading the template parset, a special imager name is allowed: dstack (not allowed for creating parsets)

    This function however, just defines the mapping between the ``dstack`` and ``YandaSoft`` and thus can be used to
    overwrite an existing mapping of a ``Parset``. This is useful when a ``Parset`` has already created from a template,
    but we want to modify some values and create a parset that uses a different imager.

    Currently not all parset parameters are supported.
    e.g. only the Clean solvers are supported in the current version of ``dstcak``

    Parameters
    ==========
    image_names: str
        The ``Names`` parameter of ``YandaSoft`` imager task. The code *currelntly not supporting a list of Names only a single Name can be used*!

    gridder_name: str
        The ``gridder`` parameter of ``YandaSoft`` imager task.

    Returns
    =======
    parset_mapping: dict
        The dictionary that is the mapping between the ``dstack`` and ``YandaSoft`` parset variables
    inverse_parset_mapping: dict
        Inverse dictionary of ``parset_mapping``
    """
    parset_mapping = {
        'dataset': 'dataset',
        'grid' : 'grid',
        'psfgrid' : 'psfgrid',
        'pcf' : 'pcf',
        'imagetype' : 'imagetype',
        'nworkergroups' : 'nworkergroups',
        'nchanpercore' : 'nchanpercore',
        'Channels' : 'Channels',
        'Frequencies' : 'Frequencies',
        'beams' : 'beams',
        'nwriters' : 'nwriters',
        'freqframe' : 'freqframe',
        'singleoutputfile' : 'singleoutputfile',
        'solverpercore' : 'solverpercore',
        'datacolumn' : 'datacolumn',
        'sphfuncforpsf' : 'sphfuncforpsf',
        'calibrate' : 'calibrate',
        'Cscalenoise' : 'calibrate.scalenoise',
        'Callowflag' : 'calibrate.allowflag',
        'Cignorebeam' : 'calibrate.ignorebeam',
        'gainsfile' : 'gainsfile',
        'residuals' : 'residuals',
        'restore' : 'restore',
        'nUVWMachines' : 'nUVWMachines',
        'uvwMachineDirTolerance' : 'uvwMachineDirTolerance',
        'gridder' : 'gridder',
        'MaxUV' : 'MaxUV',
        'MinUV' : 'MinUV',
        'usetmpfs' : 'usetmpfs',
        'tmpfs' : 'tmpfs',
        'rankstoringcf' : 'rankstoringcf',
        'visweights' : 'visweights',
        'VMFSreffreq' : 'visweights.MFS.reffreq',
        'solver' : 'solver',
        'ncycles' : 'ncycles',
        'sensitivityimage' : 'sensitivityimage',
        'Scutoff' : 'sensitivityimage.cutoff',
        'channeltolerance' : 'channeltolerance',
        'dumpgrids' : 'dumpgrids',
        'memorybuffers' : 'memorybuffers',
        'Ireuse' : 'Images.reuse',
        'Ishape' : 'Images.shape',
        'Icellsize' : 'Images.cellsize',
        'IwriteAtMajorCycle' : 'Images.writeAtMajorCycle',
        'INames' : 'Images.Names',
        'INnchan' : 'Images.{0:s}.nchan'.format(image_names),
        'INfrequency' : 'Images.{0:s}.frequency'.format(image_names),
        'INdirection' : 'Images.{0:s}.direction'.format(image_names),
        'INtangent' : 'Images.{0:s}.tangent'.format(image_names),
        'INewprojection' : 'Images.{0:s}.ewprojection'.format(image_names),
        'INshape' : 'Images.{0:s}.shape'.format(image_names),
        'INcellsize' : 'Images.{0:s}.cellsize'.format(image_names),
        'INnfacets' : 'Images.{0:s}.nfacets'.format(image_names),
        'INpolarisation' : 'Images.{0:s}.polarisation'.format(image_names),
        'INnterms' : 'Images.{0:s}.nterms'.format(image_names),
        'INfacetstep' : 'Images.{0:s}.facetstep'.format(image_names),
        'Gpadding' : 'gridder.padding',
        'Galldatapsf' : 'gridder.alldatapsf',
        'Goversampleweight' : 'gridder.oversampleweight',
        'GMaxPointingSeparation' : 'gridder.MaxPointingSeparation',
        'Gsnapshotimaging' : 'gridder.snapshotimaging',
        'GSwtolerance' : 'gridder.snapshotimaging.wtolerance',
        'GSclipping' : 'gridder.snapshotimaging.clipping',
        'GSweightsclipping' : 'gridder.snapshotimaging.weightsclipping',
        'GSreprojectpsf' : 'gridder.snapshotimaging.reprojectpsf',
        'GScoorddecimation' : 'gridder.snapshotimaging.coorddecimation',
        'GSinterpmethod' : 'gridder.snapshotimaging.interpmethod',
        'GSlongtrack' : 'gridder.snapshotimaging.longtrack',
        'Gbwsmearing' : 'gridder.bwsmearing',
        'GBchanbw' : 'gridder.bwsmearing.chanbw',
        'GBnsteps' : 'gridder.bwsmearing.nsteps',
        'Gparotation' : 'gridder.parotation',
        'GPangle' : 'gridder.parotation.angle',
        'Gswappols' : 'gridder.swappols',
        'GNwmax' : 'gridder.{0:s}.wmax'.format(gridder_name),
        'GNwmaxclip' : 'gridder.{0:s}.wmaxclip'.format(gridder_name),
        'GNnwplanes' : 'gridder.{0:s}.nwplanes'.format(gridder_name),
        'GNwstats' : 'gridder.{0:s}.wstats'.format(gridder_name),
        'GNalpha' : 'gridder.{0:s}.alpha'.format(gridder_name),
        'GNwsampling' : 'gridder.{0:s}.wsampling'.format(gridder_name),
        'GNWexponent' : 'gridder.{0:s}.wsampling.exponent'.format(gridder_name),
        'GNWnwplanes50' : 'gridder.{0:s}.wsampling.nwplanes50'.format(gridder_name),
        'GNWexport' : 'gridder.{0:s}.wsampling.export'.format(gridder_name),
        'GNcutoff' : 'gridder.{0:s}.cutoff'.format(gridder_name),
        'GNCabsolute' : 'gridder.{0:s}.cutoff.absolute'.format(gridder_name),
        'GNoversample' : 'gridder.{0:s}.oversample'.format(gridder_name),
        'GNmaxsupport' : 'gridder.{0:s}.maxsupport'.format(gridder_name),
        'GNlimitsupport' : 'gridder.{0:s}.limitsupport'.format(gridder_name),
        'GNvariablesupport' : 'gridder.{0:s}.variablesupport'.format(gridder_name),
        'GNoffsetsupport' : 'gridder.{0:s}.offsetsupport'.format(gridder_name),
        'GNtablename' : 'gridder.{0:s}.tablename'.format(gridder_name),
        'GNusedouble' : 'gridder.{0:s}.usedouble'.format(gridder_name),
        'GNsharecf' : 'gridder.{0:s}.sharecf'.format(gridder_name),
        'Cverbose' : 'solver.Clean.verbose',
        'Ctolerance' : 'solver.Clean.tolerance',
        'Cweightcutoff' : 'solver.Clean.weightcutoff',
        'Ccweightcutoff' : 'solver.Clean.weightcutoff.clean',
        'Calgorithm' : 'solver.Clean.algorithm',
        'Cscales' : 'solver.Clean.scales',
        'Cniter' : 'solver.Clean.niter',
        'Cgain' : 'solver.Clean.gain',
        'Cspeedup' : 'solver.Clean.speedup',
        'Cpadding' : 'solver.Clean.padding',
        'Clogevery' : 'solver.Clean.logevery',
        'Csaveintermediate' : 'solver.Clean.saveintermediate',
        'CBpsfwidth' : 'solver.Clean.psfwidth',
        'CBdetectdivergence' : 'solver.Clean.detectdivergence',
        'CBorthogonal' : 'solver.Clean.orthogonal',
        'CBdecoupled' : 'solver.Clean.decoupled',
        'Tminorcycle' : 'threshold.minorcycle',
        'Tmajorcycle' : 'threshold.majorcycle',
        'Tmasking' : 'threshold.masking',
        'PNames' : 'preconditioner.Names',
        'Ppreservecf' : 'preconditioner.preservecf',
        'PWnoisepower' : 'preconditioner.Wiener.noisepower',
        'PWnormalise' : 'preconditioner.Wiener.normalise',
        'PWrobustness' : 'preconditioner.Wiener.robustness',
        'PWtaper' : 'preconditioner.Wiener.taper',
        'PRrobustness' : 'Robust.robustness',
        'PGaussianTaper' : 'preconditioner.GaussianTaper',
        'PGTisPsfSize' : 'preconditioner.GaussianTaper.isPsfSize',
        'PGTtolerance' : 'preconditioner.GaussianTaper.tolerance',
        'Rbeam' : 'restore.beam',
        'Rbeamcutoff' : 'restore.beam.cutoff',
        'Requalise' : 'restore.equalise',
        'Rupdateresiduals' : 'restore.updateresiduals',
        'RbeamReference' : 'restore.beamReference'
    }

    inverse_parset_mapping = {v: k for k, v in parset_mapping.items()}

    return parset_mapping, inverse_parset_mapping

def check_parameter_and_Imager_compatibility(parset_param, imager=_DEFAULT_IMAGER):
    """This function defines which parameters the supported imagers are not compatible with.

    This check returns False if the given parameter is not compatible with the imager

    This function is only called when a parset is written to disc. The in-memory parest
    can be really messed up for sake of flexibility

    Parameters
    ==========
    parset_param: str
        The ``dstack`` parset parameter variable name

    imager: str
        Imager to test against

    Returns
    =======
    Compatibility: bool
        True if the parameter is allowed with the given imager,
        and False if not
    """
    assert imager in _SUPPORTED_IMAGERS, 'Imager {0:s} is not supported!'.format(imager)

    if imager == 'Cimager':
        forbidden_params = ['grid','psfgrid','pcf']

        if parset_param in forbidden_params:
            return False
        else:
            return True
    elif imager == 'Cdeconvolver':
        forbidden_params = ['dataset']

        if parset_param in forbidden_params:
            return False
        else:
            return True
    else:
        #Everything goes with the dstack imager :O
        return True

def check_parameter_and_Preconditioner_compatibility(parset_param, preconditioners=_DEFAULT_PRECONDITIONER):
    """This function defines which parameters the preconditioners used are not compatible with.

    This check returns False if the given parameter is not compatible with the preconditioner

    This function is only called when a parset is written to disc. The in-memory parest
    can be really messed up for sake of flexibility 

    Parameters
    ==========
    parset_param: str
        The ``dstack`` parset parameter variable name

    preconditioners: list
        This is the list of preconditioners used

     Returns
    =======
    Compatibility: bool
        True if the parameter is allowed with the given preconditioner(s),
        and False if not   
    """
    General_forbidden_params = ['PRrobustness']
    Wiener_forbidden_params = ['PWnoisepower', 'PWnormalise', 'PWrobustness', 'PWtaper']
    GaussianTaper_forbidden_params = ['PGTisPsfSize', 'PGTtolerance']

    if preconditioners == []:
        #The Ppreservecf parameter is allowed due to the misterious ways YandaSoft works...
        if parset_param in General_forbidden_params or \
        parset_param in Wiener_forbidden_params or \
        parset_param in GaussianTaper_forbidden_params:
            return False
        else:
            return True
    elif 'Wiener' not in preconditioners:
        if parset_param in Wiener_forbidden_params:
            return False
        else:
            return True
    elif 'GaussianTaper' not in preconditioners:
        if parset_param in GaussianTaper_forbidden_params:
            return False
        else:
            return True
    else:
        assert check_preconditioner_suppoort(preconditioners), 'The preconditioner given is not allowed!'
        return True

#=== CLASSES ===
class Parset(object):
    """Creates an in-memory dictionary of a ``YandaSoft`` parameterset.

    It can be initialized as an empty dictionary and then the user can define
    *every* parameter. Or a template parset can be used to initialize the
    imaging parameters. To build different imaging parsets for pipelines,
    the variation of the two methods is advised.

    The template parset has to have each parameter in a new line starting with
    the ``YandaSoft`` parameter e.g. ``Cimgaer.dataset``, which obviously starts
    with the imager name. See the list of imagers supported in  ``create_parset_mapping()``
    The parameters value is the string after = and a backspace until the end of the line.

    Lines starting with ``#`` are skipped.

    The mapping between ``dstack`` and ``YandaSoft`` variables is defined by
    the dictionary created with ``create_parset_mapping`.

    The mapping is an attribute of the ``Parset`` class.

    The mapping is based on the ``ASKAPSOFT`` documentation.
        - `imager <https://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/calim/imager.html>`_

    Keyword Arguments
    =================
    template_path: str or None
        Full path to a template parset file, which can be used to 
        initialize the parset parameters

    imager: str
        Have to be a selected ``YandaSoft`` Imager task. When a parset file is created
        this attribute is used to define the imager

    image_names: str
        The ``Names`` parameter of the parset. The template parset has to have this ``Names``
        parameters (if used), when read in. Nevertheless, this can be changed on the fly, and
        parsets with different names can be created. Use the ``update_parset_mapping()`` function
        for this.

    gridder_name: str
        The ``gridder`` parameter of ``YandaSoft`` imager task.

    """
    def __init__(self, template_path=None, imager=_DEFAULT_IMAGER, image_names=_DEFAULT_IMAGE_NAMES, gridder_name=_DEFAULT_GRIDDER_NAME):
        object.__setattr__(self, "_parset", {})

        assert imager in _SUPPORTED_IMAGERS, 'Imager {0:s} is not supported!'.format(imager)
        self._imager = imager
        self._image_names = image_names
        assert gridder_name in _SUPPORTED_GRIDDER_NAMES, 'Gridder {0:s} is not supported!'.format(gridder_name)
        self._gridder_name = gridder_name

        pm, ipm = create_parset_mapping(image_names=self._image_names,gridder_name=self._gridder_name)
        object.__setattr__(self,"_mapping", pm)
        object.__setattr__(self,"_inverse_mapping", ipm)

        if template_path != None:
            assert os.path.exists(template_path), 'Template parset does not exist!'

            with open(template_path, 'r') as f:
                    for line in f.readlines():
                        if line[0] == '#' or line.split() == []:
                            continue
                        
                        name = line.split()[0]

                        #Check if the name starts with a string from the list of supported imagers
                        if list(filter(name.startswith, _SUPPORTED_IMAGERS)) != []:
                            name = '.'.join(name.split('.')[1:])#Name without imager
                            value = line[line.index('= ')+2:].rstrip()#get rid of line separators with .rstrip()

                            assert name in self._inverse_mapping, \
                            'Can not interpret the parameter {0:s} given in the parset {1:s}!'.format(name,template_path)
                            
                            #Some consistency check
                            if self._inverse_mapping[name] == 'INames':
                                assert self._image_names in value, \
                                'Parsed created with different image names ({0:s}) from that defined in the template parset {1:s}!'.format(
                                    self._image_names,template_path)

                            if self._inverse_mapping[name] == 'gridder':
                                assert self._gridder_name in value, \
                                'Parsed created with different gridder name ({0:s}) from that defined in the template parset {1:s}!'.format(
                                    self._gridder_name,template_path)

                            if self._inverse_mapping[name] == 'solver':
                                assert value in  _SUPPORTED_SOLVERS, \
                                'The solver defined in the template {0:s} is not supported!'.format(template_path)

                            self._parset[self._inverse_mapping[name]] = value
                        else:
                            warnings.warn('Invalid parset parameter: {0:s} as the imager used is {1:s}! (parameter skipped)'.format(
                            name,self._imager,template_path))


    def __setattr__(self, name, value):
        """Add a new key and a corresponding value to the ``Parset``

        Parameters
        ==========
        name: str
            Name of the parameter. Have to be in ``self._mapping`` keys.

        value: str (or almost anything)
            The value of the parameter. When the ``Parset`` is saved
            the value will be converted to a string.

        Returns
        =======
        :obj:`Parset`
            The ``_parset`` dict is appended with the ``name`` and ``value``
        """
        custom_attributes = ['_imager','_image_names', '_gridder_name','_mapping','_inverse_mapping']
        if name in custom_attributes:
            object.__setattr__(self, name, value)
        else:
            self._parset[self._inverse_mapping[self._mapping[name]]] = value

    def __repr__(self):
        """Return the ``_parset`` dict as a line-by line string of keys and values.
        """
        lines = 'Imager: {0:s}\n'.format(self._imager)
        lines += 'Image Names: {0:s}\n'.format(self._image_names)
        lines += 'Parset parameters:\n{\n'
        for key, value in self._parset.items():
            lines += '\t{} = {}\n'.format(key, value)
        lines += '}'
        return lines

    def update_parset_mapping(self, image_names=_DEFAULT_IMAGE_NAMES, gridder_name=_DEFAULT_GRIDDER_NAME):
        """Update the mapping used between the ``dstack`` and ``YandaSoft`` parset variables.
        It also updates the parameters defined. Therefore, ths function supposed to be used when
        one wants to change the attributes affecting the mapping, as this keeps everything consistent.

        Parematers
        ==========
        image_names: str
            The ``Names`` parameter of the parset. The template parset has to have this ``Names``
            parameters (if used), when read in. Nevertheless, this can be changed on the fly, and
            parsets with different names can be created. Use the ``update_parset_mapping()`` function
            for this.

        gridder_name: str
            The ``gridder`` parameter of ``YandaSoft`` imager task.

        Returns
        =======
        :obj:`Parset`
            With updated mapping attributes
        """
        self._image_names = image_names
        self._gridder_name = gridder_name
        pm, ipm = create_parset_mapping(image_names=self._image_names,
                                        gridder_name=self._gridder_name)
        self._mapping = pm
        self._inverse_mapping = ipm

    def update_imager(self, imager=_DEFAULT_IMAGER):
        """Go-to routine when updating the imager.

        Parameters
        ==========
        imager: str
            Have to be a selected ``YandaSoft`` Imager task. When a parset file is created
            this attribute is used to define the imager

        Returns
        =======
        :obj:`Parset`
            With updated imager attribute
        """
        assert imager in _SUPPORTED_IMAGERS, 'Imager {0:s} is not supported!'.format(imager)
        self._imager = imager

    def save_parset(self, output_path, parset_name, overwrite=True):
        """Save the in-memory ``Parset`` to ``output_path/parset_name``.
        The saved parset can be fed into ``YandaSoft``

        Parameters
        ==========
        output_path: str
            Full path to the folder in which the parset will be saved

        parset_name:
            Name of the parset file created

        overwrite: bool
            If True, then the parset will be overwritten if existed

        Returns
        ========
        Parset file: Parset file readable by ``YandaSoft``
            Create the parset file at ``output_path/parset_name``

        """
        parset_path = os.path.join(output_path, parset_name)

        if os.path.isfile(parset_path) and overwrite == False:
            assert os.path.isfile(parset_path), \
            'The parset file {0:s} exists and the parameter overwrite is set to false!'.format(parset_path)

        with open(parset_path, 'w') as f:
            for key in self._parset.keys():
                if check_parameter_and_Imager_compatibility(key, self._imager):
                    print('{0:s}.{1:s} = {2:s}'.format(self._imager,self._mapping[key],str(self._parset[key])),file=f)
                else:
                    continue


if __name__ == "__main__":    

    parset = Parset(template_path='/home/krozgonyi/Desktop/test_parset.in',imager='dstack',image_names='image.test',gridder_name='WProject')
    #parset.set_parameters_from_template('/home/krozgonyi/Desktop/test_parset.in')

    print(parset)

    exit()

    #parset._imager = 'Cimager'

    #parset.INfacetstep = 1

    #parset.update_parset_mapping(image_names='dstack')

    parset.update_imager('Cdeconvolver')

    #exit()

    #print(parset)

    #parset.dataset = 'a.ms'
    #parset.nUVWMachines = 0.2

    parset.save_parset('/home/krozgonyi/Desktop','test_parset.in')

