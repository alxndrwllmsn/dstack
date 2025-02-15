"""
dstack
========
Python package to stack images and grids for the DINGO survey
Author(s): Kristof Rozgonyi (@rstofi)
"""
from . import applications, cim, cgrid, parset, msutil, sdiagnostics, fitsutil, \
				sourceutil, pipelineutil, miscutil

#from .version import version as __version__

__name__ = "dstack"
__author__ = ["Kristof Rozgonyi (@rstofi)"]

__all__ = ['applications', 'cim', 'cgrid', 'parset', 'msutil', 'sdiagnostics',
			'fitsutil', 'sourceutil', 'pipelineutil', 'miscutil']
