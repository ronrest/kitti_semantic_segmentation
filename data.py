from __future__ import print_function, division, unicode_literals
import os
import glob
import numpy as np
import scipy
from scipy import misc
import pickle


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Given a file path, create the necesary parent path structure if needed """
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, file):
    """ Saves an object as a pickle file to the desired file path """
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=2) # compatible with py2.7 and 3.x


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Opens a pickle file and returns the object """
    with open(file, mode = "rb") as fileObj:
        return pickle.load(fileObj)

