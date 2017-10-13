from __future__ import print_function, division, unicode_literals
import os
import glob
import numpy as np
import scipy
from scipy import misc
import pickle

id2label = ["non-road", "road"]
label2id = {val:id for id,val in enumerate(id2label)}

# ==============================================================================
#                                                                 MAYBE_MAKE_DIR
# ==============================================================================
def maybe_make_dir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ==============================================================================
#                                                                     GET_PARDIR
# ==============================================================================
def get_pardir(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


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
#                                                                       FILE2STR
# ==============================================================================
def file2str(file):
    """ Takes a file path and returns the contents of that file as a string."""
    with open(file, "r") as textFile:
        return textFile.read()


# ==============================================================================
#                                                                       STR2FILE
# ==============================================================================
def str2file(s, file, append=True, sep="\n"):
    """ Takes a string and saves it to a file. By default it appends to end of
        file.
    """
    # Setup mode (append, or replace)
    mode = "a" if append else "w"

    # Add on newline if append is selected
    if append and (sep != ""):
        s = sep + s

    # Ensure parent directory and necesary file structure exists
    maybe_make_pardir(file)

    with open(file, mode=mode) as textFile:
        textFile.write(s)


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_data_dict(data_dir, img_size=[25, 83]):
    """ Given the path to the root directory of the KITTI road dataset,
        it creates a dictionary of the data as numpy arrays.

        data_dir = directory containing `testing` and `training` subdirectories

        returns a dictionary with the keys:

        data["X_train"] = numpy array of input images (0-255 uint8)
        data["Y_train"] = numpy array of label images (uint8)
                          (pixel value representing class label)
    """
    print("Creating data dictionary")
    print("- Using data at:", data_dir)

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


if __name__ == '__main__':
    data_dir = "/path/to/data_road" # Path to the kitti road dataset
    data_dir = "/home/ronny/TEMP/kitti_road_data/data_road"
    pickle_file = "data.pickle"

    # How to chose dim sizes (for architectures that use SAME padding):
    # To allow up to 3 downsamples, pick multiples of 8   eg []
    # To allow up to 4 downsamples, pick multiples of 16
    # To allow up to 5 downsamples, pick multiples of 32  eg [32, 96]

    img_size = [299, 299]
    pickle_file = "data_299x299.pickle"

    data = create_data_dict(data_dir=data_dir, img_size=img_size)
    obj2pickle(data, pickle_file)
