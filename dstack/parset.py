"""
This module is a wrapper around parsets used in YandaSoft.
The code takes template parset(s) and creates custom parsets for pipelines such as the gridded DINGO pipeline.
"""

import os

#Define the imager used 
_IMAGER = 'Cimager'

#Define the mapping between ``dstcak`` and ``Yandasoft`` parset variables. This is the only place where the mapping is defined.
_PARSET_MAPPING = {
    'dataset': '{0:s}.dataset'.format(_IMAGER),
    'nUVWMachines' : '{0:s}.nUVWMachines'.format(_IMAGER)
}

#The inverse mapping i.e. map ``Yandasoft`` to ``dstack`` variables
_INVERSE_PARSET_MAPPING = {v: k for k, v in _PARSET_MAPPING.items()}

class Parset(object):
    def __init__(self):
        object.__setattr__(self, "_parset", {})

    def __setattr__(self, name, value):
        self._parset[_PARSET_MAPPING[name]] = value

    def set_parameters_from_template(self,template_path):
        with open(template_path, 'r') as f:
            for line in f.readlines():
                #name = line[:line.index(' ')]
                name = line.split()[0]
                value = line[line.index('= ')+2:].rstrip()#get rid of line separators with .rstrip()
                #value = line.split()[-1]
                #Raises an error if name is not in _INVERSE_PARSET_MAPPING
                self._parset[_PARSET_MAPPING[_INVERSE_PARSET_MAPPING[name]]] = value

    def save_parset(self,parset_output,parset_name):
        parset_path = os.path.join(parset_output, parset_name)
        with open(parset_path, 'w') as f:
            for key in self._parset.keys():
                print('{0:s} = {1:s}'.format(key,str(self._parset[key])),file=f)






if __name__ == "__main__":    

    #print(_PARSET_MAPPING)
    #print(_INVERSE_PARSET_MAPPING)

    parset = Parset()
    parset.set_parameters_from_template('/home/krozgonyi/Desktop/test_parset.in')

    parset.dataset = 'a.ms'
    parset.nUVWMachines = 0.2

    print(parset._parset)

    exit()

    parset.save_parset('/home/krozgonyi/Desktop','test_parset.in')

    #exit()

    print(parset._parset)

    exit()

    import configparser

    def create_parset_from_template(template_path,parset_output,parset_name,imager_type):
        """This is the core functon to set up parsets for pipelines.
        
        My idea of creating pipelines, is to use a template parset that contains all imaging setting
        used for all data sets (e.g days or beams) and only define the differences between the imaging runs.

        Therefore, the wrappers around the ``YandaSoft`` parsets should be flexible. I try to
        achieve this flexibility by using a parset template file that can be bare-bone
        containing only some variables and the rest defined by the user possibly individually for
        each imaging step. Or the parset template can be an almost fully fledged parameterset file,
        where the user defines only a few custom settings for each imaging run.

        This code currently works only with the ``imager`` and ``cdeconvolve-mpi`` imagers used in ``YandaSoft``

        The template parset should have the following structure:

        [Dataset]



        Parameters
        ==========
        template_path: str
            Full path to the parset template used to create the parset file

        parset_output: str
            Full path to the folder in which the parset will be saved

        parset_name: str
            Name of the parset (should have the extension .in)

        imager_type: str
            Name of the imager tha parset is created for. Either ``imager`` or ``Cdeconvolver``

        Returns
        =======
        Parameter set file: ``YandaSoft`` parset file
            Create the parset file at ``parset_output/parset_name``


        """
        assert os.path.exists(template_path), 'Template parset does not exist!'