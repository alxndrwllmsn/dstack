"""
Unit testing for the parset module using the unittest module
The test libraries are not part of the module!
Hence, they needs to be handled separately for now.
"""

"""TO DO:
Come up some extra clever unittesting to increase
the robustness of the tests as the Parset class is
tricsky to test up to high-level with minimal code
and with a kinda general test parset...
"""

import os
import unittest
import configparser
import copy

import dstack as ds

#Setup the parset file for the unittest
global _PARSET
global _TEST_DIR

_PARSET = './unittest_all.in'
#Working on UNIX systems as it creates a stacked grid at /var/tmp
_TEST_DIR = '/var/tmp'

def setup_Parset_unittest(parset_path):
    """For a general unittesting a parset file is used to define the actual YandaSoft parset to test the
    dstack.parset functions against. Thus, any valid parset file can be used for unittesting provided by the user.
    
    The code uses the configparser package to read the config file

    The config section has to be [Parset]

    The parset has to contain all variavbles and respective values this function returns.

    Note that the Parset class is really tricky to test!

    Parameters
    ==========
    parset_path: str
        Full path to a parset file defining specific values which can be used for unittesting.
        Hence, local datasets can be used for unittesting.

    Returns
    =======
    ParsetPath: str
        The full path (and name) of a valid YandaSoft parset used for unittesting.
        The parset have to have both the 'Wiener' and 'GaussianTaper' preconditioners
        defined for testing the preconditioning compatibility of the ``dstack`` wrapper.
    ImageNames: str
        The ``image_names`` parameter used to create the ``Parset`` object for testing.
        The ``Names`` parameter of the parset defined by ``ParsetPath`` has to be equal this
        value defined in the unittesting parset.

    GridderName: str
        Similarly to the ``ImageNames`` variable, the gridder name used in the input parset
        file and so the gridder name used in creating the mapping.
    """
    assert os.path.exists(parset_path), 'Test parset does not exist!'

    config = configparser.ConfigParser()
    config.read(parset_path)

    ParsetPath = config.get('Parset','ParsetPath')
    ImageNames = config.get('Parset','ImageNames')
    GridderName = config.get('Parset','GridderName')

    return ParsetPath, ImageNames, GridderName

class TestParset(unittest.TestCase):
    ParsetPath, ImageNames, GridderName = setup_Parset_unittest(_PARSET)

    def test_save_parset(self):
        output_parset_name = 'test_parset.in'

        initial_parset = ds.parset.Parset(template_path=self.ParsetPath,
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        initial_parset.save_parset(output_path=_TEST_DIR, parset_name=output_parset_name)

        saved_parset = ds.parset.Parset(template_path='{0:s}/{1:s}'.format(_TEST_DIR,output_parset_name),
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        assert len(initial_parset._parset) == len(saved_parset._parset), \
        'The two dictionaries containing the parameters of the template parset and the one created for unittesting have different number of elements!'
        for k, v in initial_parset._parset.items():
            assert saved_parset._parset[k] == v, 'The saved parest is different from the template, that saved in key {0:s}'.format(k)

    def test_sort_parset(self):
        parset = ds.parset.Parset(template_path=self.ParsetPath,
                                    imager='dstack',image_names=self.ImageNames,
                                    gridder_name=self.GridderName)

        assert parset.check_if_parset_sorted() == True, 'Parset created from template is not sorted!'
        #Remove the first element and add again
        first_key, first_value = list(parset._parset.items())[0]
        del parset._parset[first_key]
        parset._parset[first_key] = first_value
        assert parset.check_if_parset_sorted() == False, 'Parset parameters remain sorted after a key moved around (removed and added again)!'
        parset.sort_parset()
        assert parset.check_if_parset_sorted() == True, 'The sort_parset() function does not sort the parset!'

    def test_preconditioning(self):
        output_parset_name = 'test_parset_preconditioning.in'

        initial_parset = ds.parset.Parset(template_path=self.ParsetPath,
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        preconditioners_required = ['Wiener', 'GaussianTaper']
        preconditioner_parameters_for_testing = [ds.parset._WIENER_FORBIDDEN_PARAMS, ds.parset._GAUSSIANTAPER_FORBIDDEN_PARAMS]
        for p, param in zip(preconditioners_required,preconditioner_parameters_for_testing):
            assert p in initial_parset._preconditioner, \
            'The test parset is not configured correctly, the preconditioner {0:s} is missing from it!'.format(p)

            #Test removing parsets
            initial_parset.remove_preconditioner(p)
            assert p not in initial_parset._preconditioner, \
            'Can not remove preconditioner {0:s} from the Parset!'.format(p)

            for prec_param in param:
                assert ds.parset.check_parameter_and_Preconditioner_compatibility(prec_param, initial_parset._preconditioner) == False, \
                'The parameter {0:s} should not be compatible with the Parset, when the preconditioning {1:s} removed!'.format(prec_param,p)

            #Test adding parsets
            initial_parset.add_preconditioner(p)
            assert p in initial_parset._preconditioner, \
            'Can not add preconditioner {0:s} from the Parset!'.format(p)

            for prec_param in param:
                assert ds.parset.check_parameter_and_Preconditioner_compatibility(prec_param, initial_parset._preconditioner) == True, \
                'The parameter {0:s} should be compatible with the Parset, when the preconditioning {1:s} added!'.format(prec_param,p)

        #Check compatibility when no parset is added
        initial_parset._preconditioner = []

        for param in preconditioner_parameters_for_testing:
            for prec_param in param:
                assert ds.parset.check_parameter_and_Preconditioner_compatibility(prec_param, initial_parset._preconditioner) == False, \
                'The parameter {0:s} should not be compatible with the Parset, when no preconditioner is defined!'.format(prec_param)

        #Save the parset and test if the preconditioning name is set correctly
        initial_parset.save_parset(output_path=_TEST_DIR, parset_name=output_parset_name)
        saved_parset = ds.parset.Parset(template_path='{0:s}/{1:s}'.format(_TEST_DIR,output_parset_name),
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        assert saved_parset._preconditioner == [] and saved_parset._parset['PNames'] == [], \
        'Failed to properly save the preconditioners used!'

    def test_gridding_compatibility(self):
        output_parset_name = 'test_parset_gridding_compatibility.in'

        initial_parset = ds.parset.Parset(template_path=self.ParsetPath,
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        gridders_to_test = ds.parset._SUPPORTED_GRIDDER_NAMES #['Box', 'SphFunc', 'WStack', 'WProject'] the order is important!
        gridder_parameters_to_test = [ds.applications.argflatten([ds.parset._COPLANAR_FORBIDDEN_PARAMS + ds.parset._NON_ANTIALIASING_FORBIDDEN_PARAMS]),
                                    ds.parset._COPLANAR_FORBIDDEN_PARAMS, ds.parset._NON_ANTIALIASING_FORBIDDEN_PARAMS,
                                    ds.applications.argflatten([ds.parset._COPLANAR_FORBIDDEN_PARAMS + ds.parset._NON_ANTIALIASING_FORBIDDEN_PARAMS])]

        for i, gridder in zip(range(len(gridders_to_test)),gridders_to_test):
            initial_parset.update_gridder(gridder)
            if gridder != 'WProject':
                for param in gridder_parameters_to_test[i]:
                    assert ds.parset.check_parameter_and_Gridder_compatibility(param, initial_parset._gridder_name) == False, \
                    'The parameter {0:s} should not be compatible with the Parset, when {1:s} gridder is used!'.format(
                    param,initial_parset._gridder_name)
            else:
                for param in gridder_parameters_to_test[i]:
                    assert ds.parset.check_parameter_and_Gridder_compatibility(param, initial_parset._gridder_name) == True, \
                    'WProject should be compatible with all supported gridder parameters including {0:s}!'.format(param)

        #Save the parset and test if the gridder name is set correctly
        initial_parset.update_gridder('Box')
        initial_parset.save_parset(output_path=_TEST_DIR, parset_name=output_parset_name)
        saved_parset = ds.parset.Parset(template_path='{0:s}/{1:s}'.format(_TEST_DIR,output_parset_name),
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name='Box')

        assert saved_parset._gridder_name == 'Box', 'Failed to properly save the updated gridder!'

    def test_image_names_param_consistency(self):
        output_parset_name = 'test_image_names_param_consistency.in'

        initial_parset = ds.parset.Parset(template_path=self.ParsetPath,
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        #Only checking with one example parameter from the dict:
        Ishape_to_test = '[x,x]'
        INshape_to_test = '[y,y]'

        #The choosen parameter is Ishape and INshape
        initial_parset.add_parset_parameter('Ishape',Ishape_to_test)
        initial_parset.add_parset_parameter('INshape',INshape_to_test)

        #Save the parset
        initial_parset.save_parset(output_path=_TEST_DIR, parset_name=output_parset_name)

        saved_parset = ds.parset.Parset(template_path='{0:s}/{1:s}'.format(_TEST_DIR,output_parset_name),
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        assert saved_parset._parset['Ishape'] == saved_parset._parset['INshape'], 'The saved parset {0:s}/{1:s} have ambigous Ishape and INshape!'.format(
                _TEST_DIR,output_parset_name)

        assert saved_parset._parset['INshape'] == Ishape_to_test, 'The saved parset {0:s}/{1:s} have ambigous Ishape and INshape!'.format(
                _TEST_DIR,output_parset_name)

        del saved_parset

        #Now the other way around
        initial_parset.save_parset(output_path=_TEST_DIR, parset_name=output_parset_name, use_image_names=True)

        saved_parset = ds.parset.Parset(template_path='{0:s}/{1:s}'.format(_TEST_DIR,output_parset_name),
                                        imager='dstack',image_names=self.ImageNames,
                                        gridder_name=self.GridderName)

        assert saved_parset._parset['Ishape'] == saved_parset._parset['INshape'], 'The saved parset {0:s}/{1:s} have ambigous Ishape and INshape!'.format(
                _TEST_DIR,output_parset_name)

        assert saved_parset._parset['Ishape'] == INshape_to_test, 'The saved parset {0:s}/{1:s} have ambigous Ishape and INshape!'.format(
                _TEST_DIR,output_parset_name)

if __name__ == "__main__":
    unittest.main()