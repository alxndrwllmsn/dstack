"""Suite of utility functions useful in general (i.e. conversions and such)

TO DO: move some functions here to avoid duplication of code!
"""

__all__ = ['deg2arcsec']

import numpy as np
import logging

import dstack as ds

#=== Setup logging ===
log = logging.getLogger(__name__)

#=== Functions ===
def deg2arcsec(d):
	"""
	"""
	return d * 3600


#=== MAIN ===
if __name__ == "__main__":
    pass