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


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_data_dict(data_dir, img_size=[25, 83]):
    """ data_road = directory named `data_road` containing
                    `testing` and `training` subdirectories
    """
    print("Creating data dictionary")
    print("- Using data at:", data_dir)
    # np.array([375, 1242])/15  # array([ 25. ,  82.8])
    # img_size = [25, 83]

    # Directories
    imgs_dir = os.path.join(data_dir, "training/image_2")
    labels_dir = os.path.join(data_dir, "training/gt_image_2")

    print("- Getting list of files")
    # Only get the label files for road (not lane)
    label_files = glob.glob(os.path.join(labels_dir, "*_road_*.png"))

    # Create corresponding list of training image files
    img_files = list(map(lambda f: os.path.basename(f).replace("_road", ""), label_files))
    img_files = list(map(lambda f: os.path.join(imgs_dir, f), img_files)) # absolute path

    n_samples = len(img_files)
    print("- Encountered {} samples".format(n_samples))
    est_filesize = (n_samples*np.prod(img_size)*(3+1))/1e6
    print("- Estimated output filesize: {:0.3f} MB + overhead".format(est_filesize))

    data = {}
    data["X_train"] = np.empty([n_samples]+img_size+[3], dtype=np.uint8)
    data["Y_train"] = np.empty([n_samples]+img_size, dtype=np.uint8)

    print("- Processing image files")
    for i in range(n_samples):
        label_img = scipy.misc.imread(label_files[i])
        input_img = scipy.misc.imread(img_files[i])

        # PRERPOCESS THE IMAGES
        label_img = scipy.misc.imresize(label_img, img_size)
        input_img = scipy.misc.imresize(input_img, img_size)

        # PROCESSING LABEL IMAGE
        # Only one channel, (1=road, 0=not road)
        non_road_class = np.array([255,0,0])
        label_img = (1-np.all(label_img==non_road_class, axis=2, keepdims=False)).astype(np.uint8)

        # Place the images into the data arrays
        data["X_train"][i] = input_img
        data["Y_train"][i] = label_img

    print("- Shuffling the data")
    np.random.seed(seed=128)
    ids = list(np.random.permutation(n_samples))
    data["X_train"] = data["X_train"][ids]
    data["Y_train"] = data["Y_train"][ids]

    print("- Done!")
    return data


