"""Utility functions for Snakemake pipelines using dstack
"""

__all__ = ['read_TSF_dsconfig']

import sys, os
import numpy as np
import logging

import configparser

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def read_TSF_dsconfig(config_path):
    """Wrapper for reading data from config files created for sources by `dstack`
    for targeted source finding using the `initTSF` tool.

    Parameters
    ==========

    config_path: str
        Full path to a configuration file to be read
    
    Returns
    =======
    Converts the config file to `python` variables. For the details see:
    `dstack.applications.initTSF`

    """

    config = configparser.ConfigParser()
    config.read(config_path)

    #Get envintoent variables
    fitspath = str(config.get('ENV','fitspath'))
    dsID = str(config.get('ENV','dsID'))

    #Get the Image variables
    b_maj = float(config.get('IMG','b_maj'))
    b_min = float(config.get('IMG','b_min'))
    b_pa = float(config.get('IMG','b_pa'))
    d_px = float(config.get('IMG','d_px'))
    b_maj_px = float(config.get('IMG','b_maj_px'))
    b_min_px = float(config.get('IMG','b_min_px'))
    

    source_ID = str(config.get('SOURCE','ID'))
    RA = float(config.get('SOURCE','RA'))
    Dec = float(config.get('SOURCE','Dec'))
    freq = float(config.get('SOURCE','Freq'))
    region_string = str(config.get('SOURCE','Region'))

    return fitspath, dsID, b_maj, b_min, b_pa, d_px, b_maj_px, b_min_px, source_ID, RA, Dec, freq, region_string

#=== MAIN ===
if __name__ == "__main__":
    pass
