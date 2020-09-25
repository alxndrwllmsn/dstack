"""
This module is a wrapper around parsets used in YandaSoft.
The code takes template parset(s) and creates custom parsets for pipelines such as the gridded DINGO pipeline.
"""

__all__ = ['create_parset_mapping', 'Parset']

import os
import warnings

#=== CREATE GLOBALS ===
def create_parset_mapping(imager='Cimager',image_names='test'):
    
    global _IMAGER 
    global _IMAGE_NAME
    global _PARSET_MAPPING
    global _INVERSE_PARSET_MAPPING

    #Define the imager used
    _IMAGER = imager

    #Define the Image names used
    _IMAGE_NAME = image_names

    #Define the mapping between ``dstcak`` and ``Yandasoft`` parset variables. This is the only place where the mapping is defined.
    _PARSET_MAPPING = {
        'imagetype' : '{0:s}.imagetype'.format(_IMAGER),
        'dataset': '{0:s}.dataset'.format(_IMAGER),
        'nworkergroups' : '{0:s}.nworkergroups'.format(_IMAGER),
        'nchanpercore' : '{0:s}.nchanpercore'.format(_IMAGER),
        'Channels' : '{0:s}.Channels'.format(_IMAGER),
        'Frequencies' : '{0:s}.Frequencies'.format(_IMAGER),
        'beams' : '{0:s}.beams'.format(_IMAGER),
        'nwriters' : '{0:s}.nwriters'.format(_IMAGER),
        'freqframe' : '{0:s}.freqframe'.format(_IMAGER),
        'singleoutputfile' : '{0:s}.singleoutputfile'.format(_IMAGER),
        'solverpercore' : '{0:s}.solverpercore'.format(_IMAGER),
        'datacolumn' : '{0:s}.datacolumn'.format(_IMAGER),
        'sphfuncforpsf' : '{0:s}.sphfuncforpsf'.format(_IMAGER),
        'calibrate' : '{0:s}.calibrate'.format(_IMAGER),
        'Cscalenoise' : '{0:s}.calibrate.scalenoise'.format(_IMAGER),
        'Callowflag' : '{0:s}.calibrate.allowflag'.format(_IMAGER),
        'Cignorebeam' : '{0:s}.calibrate.ignorebeam'.format(_IMAGER),
        'gainsfile' : '{0:s}.gainsfile'.format(_IMAGER),
        'residuals' : '{0:s}.residuals'.format(_IMAGER),
        'restore' : '{0:s}.restore'.format(_IMAGER),
        'Rbeam' : '{0:s}.restore.beam'.format(_IMAGER),
        'Rbeamcutoff' : '{0:s}.restore.beam.cutoff'.format(_IMAGER),
        'Requalise' : '{0:s}.restore.equalise'.format(_IMAGER),
        'Rupdateresiduals' : '{0:s}.restore.updateresiduals'.format(_IMAGER),
        'nUVWMachines' : '{0:s}.nUVWMachines'.format(_IMAGER),
        'uvwMachineDirTolerance' : '{0:s}.uvwMachineDirTolerance'.format(_IMAGER),
        'gridder' : '{0:s}.gridder'.format(_IMAGER),
        'MaxUV' : '{0:s}.MaxUV'.format(_IMAGER),
        'MinUV' : '{0:s}.MinUV'.format(_IMAGER),
        'usetmpfs' : '{0:s}.usetmpfs'.format(_IMAGER),
        'tmpfs' : '{0:s}.tmpfs'.format(_IMAGER),
        'rankstoringcf' : '{0:s}.rankstoringcf'.format(_IMAGER),
        'visweights' : '{0:s}.visweights'.format(_IMAGER),
        'VMFSreffreq' : '{0:s}.visweights.MFS.reffreq'.format(_IMAGER),
        'solver' : '{0:s}.solver'.format(_IMAGER),
        'ncycles' : '{0:s}.ncycles'.format(_IMAGER),
        'sensitivityimage' : '{0:s}.sensitivityimage'.format(_IMAGER),
        'Scutoff' : '{0:s}.sensitivityimage.cutoff'.format(_IMAGER),
        'channeltolerance' : '{0:s}.channeltolerance'.format(_IMAGER),
        'dumpgrids' : '{0:s}.dumpgrids'.format(_IMAGER),
        'memorybuffers' : '{0:s}.memorybuffers'.format(_IMAGER),
        'Ireuse' : '{0:s}.Images.reuse'.format(_IMAGER),
        'Ishape' : '{0:s}.Images.shape'.format(_IMAGER),
        'Icellsize' : '{0:s}.Images.cellsize'.format(_IMAGER),
        'IwriteAtMajorCycle' : '{0:s}.Images.writeAtMajorCycle'.format(_IMAGER),
        'INames' : '{0:s}.Images.Names'.format(_IMAGER),
        'INnchan' : '{0:s}.Images.{1:s}.nchan'.format(_IMAGER,_IMAGE_NAME),
        'INfrequency' : '{0:s}.Images.{1:s}.frequency'.format(_IMAGER,_IMAGE_NAME),
        'INdirection' : '{0:s}.Images.{1:s}.direction'.format(_IMAGER,_IMAGE_NAME),
        'INtangent' : '{0:s}.Images.{1:s}.tangent'.format(_IMAGER,_IMAGE_NAME),
        'INewprojection' : '{0:s}.Images.{1:s}.ewprojection'.format(_IMAGER,_IMAGE_NAME),
        'INshape' : '{0:s}.Images.{1:s}.shape'.format(_IMAGER,_IMAGE_NAME),
        'INcellsize' : '{0:s}.Images.{1:s}.cellsize'.format(_IMAGER,_IMAGE_NAME),
        'INnfacets' : '{0:s}.Images.{1:s}.nfacets'.format(_IMAGER,_IMAGE_NAME),
        'INpolarisation' : '{0:s}.Images.{1:s}.polarisation'.format(_IMAGER,_IMAGE_NAME),
        'INnterms' : '{0:s}.Images.{1:s}.nterms'.format(_IMAGER,_IMAGE_NAME),
        'INfacetstep' : '{0:s}.Images.{1:s}.facetstep'.format(_IMAGER,_IMAGE_NAME)
    }

    #The inverse mapping i.e. map ``Yandasoft`` to ``dstack`` variables
    _INVERSE_PARSET_MAPPING = {v: k for k, v in _PARSET_MAPPING.items()}

#=== CLASSES ===
class Parset(object):
    """Creates an in-memory dictionary of a ``YandaSoft`` parameterset.

    It can be initialized as an empty dictionary and then the user can define
    *every* parameter. Or a template parset can be used to initialize the
    imaging parameters. To build different imaging parsets for pipelines,
    the variation of the two methods is advised.

    The template parset has to have each parameter in a new line starting with
    the ``YandaSoft`` parameter e.g. ``Cimgaer.dataset`` and the parameters value
    is the string after = and a backspace until the end of the line.

    Lines starting with ``#`` are skipped.

    The mapping between ``dstack`` and ``YandaSoft`` variables is defined by
    the ``_PARSET_MAPPING`` global dictionary.
    
    The mapping is based on the ``ASKAPSOFT`` documentation.
        - `imager <https://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/calim/imager.html>`_


    Keyword Arguments
    =================
    template_path: str or None
        Full path to a template parset file, which can be used to 
        initialize the parset parameters

    """
    def __init__(self,template_path=None,imager='Cimager',image_names='test'):
        object.__setattr__(self, "_parset", {})

        create_parset_mapping(imager=imager,image_names=image_names)

        if template_path != None:
            assert os.path.exists(template_path), 'Template parset does not exist!'

            with open(template_path, 'r') as f:
                    for line in f.readlines():
                        if line[0] == '#' or line.split() == []:
                            continue
                        else:
                            name = line.split()[0]

                            if name.startswith(_IMAGER):
                                value = line[line.index('= ')+2:].rstrip()#get rid of line separators with .rstrip()

                                assert name in _INVERSE_PARSET_MAPPING, \
                                'Can not interpret the parameter {0:s} given in the parset {1:s}!'.format(name,template_path)
                                
                                #Check if the parameter is allowed to use jointly with the _IMAGER used

                                self._parset[_PARSET_MAPPING[_INVERSE_PARSET_MAPPING[name]]] = value
                            else:
                                warnings.warn('Invalid parset parameter: {0:s} as the imager used is {1:s}! (parameter skipped)'.format(
                                name,_IMAGER))


    def __setattr__(self, name, value):
        """Add a new key and a corresponding value to the ``Parset``

        Parameters
        ==========
        name: str
            Name of the parameter. Have to be in ``_PARSET_MAPPING`` keys.

        value: str (or almost anything)
            The value of the parameter. When the ``Parset`` is saved
            the value will be converted to a string.

        Returns
        =======
        :obj:`Parset`
            The ``_parset`` dict is appended with the ``name`` and ``value``
        """
        self._parset[_PARSET_MAPPING[name]] = value

    def __repr__(self):
        """Return the ``_parset`` dict as a line-by line string of keys and values.
        """
        lines = '{\n'
        for key, value in self._parset.items():
            lines += '{} = {}\n'.format(key, value)
        lines += '}'
        return lines

    def save_parset(self,output_path,parset_name):
        """Save the in-memory ``Parset`` to ``output_path/parset_name``.
        The saved parset can be fed into ``YandaSoft``

        If the file already exist, it gets deleted! 

        Parameters
        ==========
        
        output_path: str
            Full path to the folder in which the parset will be saved

        parset_name:
            Name of the parset file created

        Returns
        ========
        Parset file: Parset file readable by ``YandaSoft``
            Create the parset file at ``output_path/parset_name``

        """
        parset_path = os.path.join(output_path, parset_name)

        if os.path.isfile(parset_path):
            warnings.warn('Parset file {0:s} existed and got overwritten now!'.format(parset_path))
            os.remove(parset_path) 

        with open(parset_path, 'w') as f:
            for key in self._parset.keys():

                #Check if key is is allowed to use jointly with the _IMAGER used

                print('{0:s} = {1:s}'.format(key,str(self._parset[key])),file=f)



if __name__ == "__main__":    

    #print(_PARSET_MAPPING)
    #print(_INVERSE_PARSET_MAPPING)

    parset = Parset(template_path='/home/krozgonyi/Desktop/test_parset.in',imager='Cimager',image_names='image.test')
    #parset.set_parameters_from_template('/home/krozgonyi/Desktop/test_parset.in')

    print(parset)

    exit()

    parset.dataset = 'a.ms'
    parset.nUVWMachines = 0.2

    print(parset._parset)

    exit()

    parset.save_parset('/home/krozgonyi/Desktop','test_parset.in')

    #exit()

    print(parset._parset)
